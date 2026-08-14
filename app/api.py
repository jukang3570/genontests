"""FastAPI 엔드포인트와 애플리케이션 의존성을 구성하는 모듈."""

import os
import asyncio
import hashlib
import re
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app.answers import AnswerService, create_answer_service
from app.classifier import IntentClassifier, create_classifier
from app.config import Settings
from app.csv_trace import TraceRecorder, create_trace_recorder
from app.graph import MasterIntentGraph, MasterResult
from app.history import ChatHistoryStore, create_history_store
from app.hitl_store import (
    HitlStateNotFoundError,
    HitlStateStore,
    HitlStateStoreUnavailableError,
    create_hitl_state_store,
)
from app.models import (
    ChatRequest,
    ChatResponse,
    StreamingChatRequest,
    StreamingUser,
)
from app.mcp.client import McpToolExecutor, create_mcp_tool_executor
from app.observability import (
    configure_logging,
    log_context,
    log_failure_diagnostic,
    logger,
    timed,
)
from app.prompt_loader import PromptBundleLoader
from app.subagents.router import SubagentRouter, create_subagent_router
from app.streaming import build_action_event, encode_sse


def create_app(
    *,
    settings: Settings | None = None,
    classifier: IntentClassifier | None = None,
    history_store: ChatHistoryStore | None = None,
    hitl_store: HitlStateStore | None = None,
    subagent_router: SubagentRouter | None = None,
    mcp_executor: McpToolExecutor | None = None,
    trace_recorder: TraceRecorder | None = None,
    answer_service: AnswerService | None = None,
) -> FastAPI:
    """설정, 프롬프트, LLM, Redis, 그래프를 조립해 FastAPI 앱을 만든다.

    테스트에서는 가짜 분류기와 메모리 저장소를 주입할 수 있다. 운영 실행에서는
    실제 GenOS 분류기, Redis 대화 이력 저장소, 일반 Redis HITL 저장소가 생성된다.
    LangGraph Checkpointer는 생성하거나 연결하지 않는다.
    """

    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    settings = settings or Settings.from_env()
    prompt = PromptBundleLoader().load(settings.prompt_version)
    classifier = classifier or create_classifier(settings, prompt)
    history_store = history_store or create_history_store(settings)
    hitl_store = hitl_store or create_hitl_state_store(settings)
    subagent_router = subagent_router or create_subagent_router(settings)
    mcp_executor = mcp_executor or create_mcp_tool_executor(settings)
    answer_service = answer_service or create_answer_service(settings)
    trace_recorder = trace_recorder or create_trace_recorder(
        enabled=settings.csv_trace_enabled,
        directory=settings.csv_trace_dir,
        project_code=settings.project_code,
    )

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        """그래프 생성과 두 Redis 연결의 종료 수명주기를 관리한다."""

        # 그래프는 Checkpointer 없이 한 번 컴파일한다. HTTP 요청 사이에 유지할
        # HITL 상태는 그래프 내부가 아니라 일반 Redis 저장소가 직접 관리한다.
        app_instance.state.graph = MasterIntentGraph(
            classifier,
            history_store,
            hitl_store,
            settings.history_limit,
            subagent_router,
            mcp_executor,
            trace_recorder,
        )
        logger.info(
            "======== 애플리케이션 시작 | 프로젝트=%s | 프롬프트버전=%s | "
            "이력저장소=%s | HITL저장소=%s | HITL활성=%s | "
            "서브에이전트=%s | MCP=%s | Reranking=%s | "
            "Reranking모델=%s | LLM자동재시도=%d회 | "
            "MCP자동재시도=%d회 | LangGraph노드재시도=0회 | "
            "Checkpointer=사용안함",
            settings.project_code,
            prompt.version,
            settings.history_backend,
            settings.hitl_state_backend,
            hitl_store.enabled,
            subagent_router.registered_codes(),
            settings.mcp_backend,
            settings.reranking_enabled,
            settings.reranking_model,
            settings.llm_max_retries,
            settings.mcp_max_retries,
        )
        try:
            yield
        finally:
            # 두 저장소는 기본적으로 같은 Redis 서버를 사용하지만 연결 풀은
            # 각각 생성되므로 애플리케이션 종료 시 모두 닫아 준다.
            await history_store.aclose()
            await hitl_store.aclose()
            await mcp_executor.aclose()
            await answer_service.aclose()
            logger.info("======== 애플리케이션 종료")

    app = FastAPI(
        title="Master Agent Chat API",
        version="0.4.0",
        description=(
            "WAS 사용자 정보와 간결한 SSE 출력을 사용하는 마스터·서브에이전트 "
            "및 일반 Redis HITL API"
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """그래프 진입 전 입력 계약 오류도 필드·위치·수정 방법을 기록한다."""

        request_id = request.headers.get("x-request-id", str(uuid4()))
        log_failure_diagnostic(
            stage="FastAPI 요청 본문 검증",
            code_location="app/api.py:request_validation_error_handler",
            exc=exc,
            likely_cause=(
                "필수 필드 누락, 타입·길이·패턴 오류 또는 null 허용 범위 위반"
            ),
            corrective_action=(
                "로그의 validation_errors loc 필드와 app/models.py의 "
                "요청 경로에 연결된 Pydantic 스키마와 비교해 호출 본문을 수정하세요."
            ),
            retry_count=0,
            context={
                "method": request.method,
                "path": request.url.path,
                "validation_errors": exc.errors(),
            },
        )
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(
                {
                    "error": {
                        "code": "REQUEST_VALIDATION_ERROR",
                        "message": "요청 입력 형식이 올바르지 않습니다.",
                        "request_id": request_id,
                        "details": exc.errors(),
                    }
                }
            ),
        )

    @app.get("/tester", include_in_schema=False)
    @timed("의도분류 테스트 화면")
    async def intent_tester() -> FileResponse:
        """브라우저에서 사용할 단일 HTML 테스트 화면을 반환한다."""

        html_path = (
            Path(__file__).resolve().parents[1]
            / "static"
            / "intent_tester.html"
        )
        logger.info(
            "======== 의도분류 테스트 화면 요청 | 파일=%s",
            html_path,
        )
        return FileResponse(html_path, media_type="text/html")

    @app.get("/v1/metadata")
    @timed("채팅 메타데이터 API")
    async def metadata() -> dict[str, Any]:
        """활성 프롬프트에서 읽은 테스트용 에이전트 코드와 모드를 반환한다."""

        logger.info(
            "======== 채팅 메타데이터 요청 | 에이전트개수=%d",
            len(prompt.agent_codes),
        )
        return {
            "prompt_version": prompt.version,
            # manifest에서 동적으로 읽은 값이므로 HTML에 코드를 하드코딩하지 않는다.
            "agent_codes": list(prompt.agent_codes),
            "chat_history_store": settings.history_backend,
            "hitl_state_store": settings.hitl_state_backend,
            "subagent_codes": list(subagent_router.registered_codes()),
            "mcp_backend": settings.mcp_backend,
            "reranking": {
                "enabled": settings.reranking_enabled,
                "serving_id": settings.reranking_serving_id,
                "model": settings.reranking_model,
                "top_n": settings.reranking_top_n,
            },
            "stream_endpoint": "v1/chat/stream",
            "retry_policy": {
                "llm_max_retries": settings.llm_max_retries,
                "mcp_max_retries": settings.mcp_max_retries,
                "langgraph_node_retries": 0,
            },
        }

    @app.get("/health")
    @timed("상태 확인 API")
    async def health() -> dict[str, str]:
        """서버 상태, 프롬프트 버전, HITL 저장 방식을 반환한다."""

        logger.info("======== 상태 확인 요청")
        return {
            "status": "ok",
            "prompt_version": prompt.version,
            "chat_history_store": settings.history_backend,
            "hitl_state_store": settings.hitl_state_backend,
            "langgraph_checkpointer": "disabled",
            "mcp_backend": settings.mcp_backend,
            "reranking_enabled": str(settings.reranking_enabled).lower(),
            "reranking_model": settings.reranking_model,
            "llm_max_retries": str(settings.llm_max_retries),
            "mcp_max_retries": str(settings.mcp_max_retries),
            "langgraph_node_retries": "0",
        }

    @app.get("/v1/chat/history")
    @timed("에이전트별 대화이력 조회 API")
    async def get_chat_history(
        user_id: str = Query(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9_-]+$",
            description="WAS 사용자 정보의 user.id",
        ),
        session_id: str = Query(min_length=1, max_length=200),
        agent_code: str = Query(min_length=1, max_length=100),
        limit: int = Query(default=100, ge=1, le=500),
        authorization: str = Header(...),
    ) -> dict[str, Any]:
        """사원·session·agent_code가 모두 같은 대화만 반환한다.

        로컬 테스트에서는 프론트가 직접 호출한다. 운영에서는 WAS가 인증된
        ``user.id``를 ``user_id`` 쿼리로 넣고 Bearer 토큰을 전달해야 한다.
        Redis 장애 시 저장소 정책에 따라 빈 메시지 배열을 정상 반환한다.
        """

        _require_bearer_token(authorization)
        normalized_code = _validate_history_agent_code(
            agent_code,
            prompt.agent_codes,
        )
        logger.info(
            "======== 에이전트별 대화이력 조회 요청 | 사용자=%s | "
            "session_id=%s | 에이전트=%s | 저장소=%s",
            user_id,
            session_id,
            normalized_code,
            settings.history_backend,
        )
        messages = await history_store.get_recent(
            user_id,
            session_id,
            normalized_code,
            limit,
            include_metadata=True,
        )
        logger.info(
            "======== 에이전트별 대화이력 조회 완료 | 사용자=%s | "
            "session_id=%s | 에이전트=%s | 메시지개수=%d",
            user_id,
            session_id,
            normalized_code,
            len(messages),
        )
        return {
            "project_code": settings.project_code,
            "backend": settings.history_backend,
            "user_id": user_id,
            "session_id": session_id,
            "agent_code": normalized_code,
            "count": len(messages),
            "messages": messages,
        }

    @app.delete("/v1/chat/history")
    @timed("대화이력 범위 삭제 API")
    async def delete_chat_history(
        user_id: str = Query(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9_-]+$",
            description="WAS 사용자 정보의 user.id",
        ),
        session_id: str = Query(min_length=1, max_length=200),
        agent_code: str | None = Query(
            default=None,
            min_length=1,
            max_length=100,
            description=(
                "지정하면 해당 에이전트만 삭제하고, 생략하면 session 전체를 삭제"
            ),
        ),
        authorization: str = Header(...),
    ) -> dict[str, Any]:
        """에이전트 지정 시 해당 이력만, 미지정 시 session 전체를 삭제한다."""

        _require_bearer_token(authorization)
        normalized_code = (
            _validate_history_agent_code(agent_code, prompt.agent_codes)
            if agent_code is not None
            else None
        )
        logger.info(
            "======== 대화이력 삭제 요청 | 사용자=%s | "
            "session_id=%s | 에이전트=%s | 전체세션삭제=%s",
            user_id,
            session_id,
            normalized_code or "미지정",
            normalized_code is None,
        )
        if normalized_code is None:
            deleted = await history_store.delete_session(
                user_id,
                session_id,
            )
        else:
            deleted = await history_store.delete_agent_history(
                user_id,
                session_id,
                normalized_code,
            )
        logger.info(
            "======== 대화이력 삭제 완료 | 사용자=%s | "
            "session_id=%s | 에이전트=%s | 전체세션삭제=%s | 삭제메시지=%d",
            user_id,
            session_id,
            normalized_code or "미지정",
            normalized_code is None,
            deleted,
        )
        return {
            "project_code": settings.project_code,
            "backend": settings.history_backend,
            "user_id": user_id,
            "session_id": session_id,
            "agent_code": normalized_code,
            "scope": "SESSION" if normalized_code is None else "AGENT",
            "deleted_message_count": deleted,
        }

    @app.get("/v1/tester/history", include_in_schema=False)
    @timed("목업 대화이력 조회 API")
    async def tester_history(
        employee_id: str = Query(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9_-]+$",
        ),
        session_id: str = Query(min_length=1, max_length=200),
        agent_code: str = Query(min_length=1, max_length=100),
    ) -> dict[str, Any]:
        """목업에서 현재 사원·대화·에이전트 범위의 이력을 조회한다.

        저장소 전체를 노출하지 않고 실제 멀티턴 분류에서 사용하는 get_recent와
        같은 범위 조건을 적용한다. 운영용 업무 API가 아니라 테스트 화면의
        진단 기능이므로 OpenAPI 문서에서는 숨긴다.
        """

        normalized_code = agent_code.upper()
        if normalized_code not in prompt.agent_codes:
            logger.info(
                "======== 목업 이력 조회 거절 | 미등록에이전트=%s",
                normalized_code,
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "등록되지 않은 agent_code입니다.",
                    "allowed_codes": list(prompt.agent_codes),
                },
            )

        logger.info(
            "======== 목업 대화이력 조회 요청 | 사원번호=%s | "
            "session_id=%s | 에이전트=%s | 저장소=%s",
            employee_id,
            session_id,
            normalized_code,
            settings.history_backend,
        )
        messages = await history_store.get_recent(
            employee_id,
            session_id,
            normalized_code,
            settings.history_limit,
            include_metadata=True,
        )
        return {
            "backend": settings.history_backend,
            "scope": {
                "employee_id": employee_id,
                "session_id": session_id,
                "agent_code": normalized_code,
            },
            "count": len(messages),
            "messages": messages,
        }

    @app.get("/v1/tester/sessions", include_in_schema=False)
    @timed("목업 전체 대화 목록 API")
    async def tester_sessions() -> dict[str, Any]:
        """현재 프로젝트 저장소의 모든 session_id를 목업에 반환한다."""

        sessions = await history_store.list_sessions()
        logger.info(
            "======== 목업 전체 대화 목록 반환 | 프로젝트=%s | "
            "저장소=%s | 개수=%d",
            settings.project_code,
            settings.history_backend,
            len(sessions),
        )
        return {
            "project_code": settings.project_code,
            "backend": settings.history_backend,
            "count": len(sessions),
            "sessions": sessions,
        }

    @app.delete("/v1/tester/sessions", include_in_schema=False)
    @timed("목업 대화 삭제 API")
    async def delete_tester_session(
        employee_id: str = Query(
            min_length=1,
            max_length=100,
            pattern=r"^[A-Za-z0-9_-]+$",
        ),
        session_id: str = Query(min_length=1, max_length=200),
    ) -> dict[str, Any]:
        """선택한 사원·session의 모든 에이전트 대화이력을 삭제한다."""

        logger.info(
            "======== 목업 대화 삭제 요청 | 프로젝트=%s | 사원번호=%s | "
            "session_id=%s | 저장소=%s",
            settings.project_code,
            employee_id,
            session_id,
            settings.history_backend,
        )
        deleted_messages = await history_store.delete_session(
            employee_id,
            session_id,
        )
        return {
            "project_code": settings.project_code,
            "backend": settings.history_backend,
            "employee_id": employee_id,
            "session_id": session_id,
            "deleted_message_count": deleted_messages,
        }

    @app.post("/chat")
    @timed("GenOS 코드서빙 워크플로우 채팅")
    async def code_serving_chat(body: dict[str, Any]) -> dict[str, Any]:
        """GenOS 워크플로우 규격을 내부 채팅 그래프에 연결한다."""

        question = body.get("question")
        if not isinstance(question, str):
            question = ""
        question = question.strip()
        if question == "__verify__":
            return {"code": 0, "data": {"text": "verified"}}
        if not question:
            return {
                "code": 0,
                "data": {"text": "[ERROR] question is empty"},
            }

        raw_override_config = body.get("overrideConfig")
        override_config = (
            raw_override_config
            if isinstance(raw_override_config, dict)
            else {}
        )
        session_id = _code_serving_identifier(
            override_config.get("session_id")
            or override_config.get("sessionId"),
            prefix="session",
        )
        employee_id = _code_serving_identifier(
            override_config.get("employee_id")
            or override_config.get("employeeId")
            or override_config.get("user_id")
            or override_config.get("userId"),
            prefix="codeserving",
        )
        raw_agent_code = (
            override_config.get("agent_code")
            or override_config.get("agentCode")
        )
        agent_code = (
            raw_agent_code.strip().upper()
            if isinstance(raw_agent_code, str) and raw_agent_code.strip()
            else None
        )
        if agent_code is not None and agent_code not in prompt.agent_codes:
            return {
                "code": 0,
                "data": {
                    "text": (
                        "[ERROR] 등록되지 않은 agent_code입니다: "
                        f"{agent_code}"
                    )
                },
            }

        thread_id = str(uuid4())
        request_id = f"{settings.project_code}:{uuid4()}"
        with log_context(
            request_id=request_id,
            session_id=session_id,
            thread_id=thread_id,
            user_id=employee_id,
        ):
            logger.info(
                "======== 코드서빙 워크플로우 질문 도착 | 질문길이=%d | "
                "선택에이전트=%s | 전달이력개수=%d",
                len(question),
                agent_code or "선택하지 않음",
                len(body.get("history", []))
                if isinstance(body.get("history"), list)
                else 0,
            )
            graph: MasterIntentGraph = app.state.graph
            result = await graph.start(
                thread_id=thread_id,
                employee_id=employee_id,
                session_id=session_id,
                message=question,
                frontend_agent_code=agent_code,
            )

            if result.status == "INPUT_REQUIRED":
                interrupt = result.interrupt or {}
                prompt_text = interrupt.get("message")
                if not isinstance(prompt_text, str) or not prompt_text.strip():
                    prompt_text = "추가 입력이 필요합니다."
                return {"code": 0, "data": {"text": prompt_text}}

            prepared = await answer_service.prepare(result)
            answer_parts: list[str] = []
            async for token in prepared.tokens:
                if token:
                    answer_parts.append(token)
            answer = "".join(answer_parts).strip()
            if not answer:
                answer = "조회 결과가 없습니다. 잠시 후 다시 시도해 주세요."

            if (
                result.status == "PASS"
                and result.classification.agent_code is not None
            ):
                await history_store.append_message(
                    employee_id=employee_id,
                    session_id=session_id,
                    agent_code=result.classification.agent_code,
                    role="assistant",
                    content=answer,
                    message_id=str(uuid4()),
                    metadata={"renderables": prepared.renderables},
                )

            return {"code": 0, "data": {"text": answer}}

    @app.post("/v1/chat", response_model=ChatResponse)
    @timed(
        "채팅 요청 전체 처리",
        expected_exceptions=(HTTPException,),
    )
    async def chat(body: ChatRequest) -> ChatResponse:
        """신규 질문 또는 Redis HITL 입력을 동일한 API에서 처리한다."""

        graph: MasterIntentGraph = app.state.graph

        # thread_id가 있으면 신규 질문이 아니라 이전 INPUT_REQUIRED 응답에 대한
        # 후속 입력이다. Redis에 저장된 hitl_type과 상태를 기준으로 LangGraph가
        # 검증 Edge를 선택하므로 프론트가 분기 코드를 따로 판단할 필요가 없다.
        if body.is_hitl_continuation:
            assert body.thread_id is not None
            assert body.hitl_input is not None
            logger.info(
                "======== 통합 채팅 요청 도착 | 유형=HITL재진입 | "
                "thread_id=%s | 입력=%s",
                body.thread_id,
                body.hitl_input,
            )
            try:
                result = await graph.resume(
                    thread_id=body.thread_id,
                    value=body.hitl_input,
                )
            except HitlStateNotFoundError as exc:
                # TTL 만료, 잘못된 thread_id 또는 이미 승인되어 DEL된 상태이다.
                logger.info(
                    "======== HITL 재진입 거절 | Redis HITL 상태 없음 | "
                    "thread_id=%s",
                    body.thread_id,
                )
                raise HTTPException(
                    status_code=404,
                    detail={
                        "code": "HITL_STATE_NOT_FOUND",
                        "message": (
                            "이어 갈 HITL 상태를 찾을 수 없습니다. 상태가 "
                            "만료됐거나, 이미 처리됐거나, thread_id가 "
                            "올바르지 않을 수 있습니다."
                        ),
                        "thread_id": body.thread_id,
                        "action": "START_NEW_CHAT",
                    },
                ) from exc
            except HitlStateStoreUnavailableError as exc:
                raise _hitl_store_unavailable() from exc

            logger.info(
                "======== 통합 채팅 처리 결과 | 유형=HITL재진입 | "
                "thread_id=%s | 상태=%s",
                body.thread_id,
                result.status,
            )
            return _response(result)

        # Pydantic의 요청 모드 검증을 통과했으므로 신규 질문의 message와
        # employee_id는 반드시 존재한다. frontend_agent_code는 사용자가
        # 에이전트를 선택하지 않은 경우 None일 수 있다.
        assert body.message is not None
        assert body.employee_id is not None
        frontend_code = (
            body.frontend_agent_code.upper()
            if body.frontend_agent_code is not None
            else None
        )
        if (
            frontend_code is not None
            and frontend_code not in prompt.agent_codes
        ):
            logger.info(
                "======== 요청 검증 실패 | 알 수 없는 에이전트코드=%s",
                frontend_code,
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "등록되지 않은 frontend_agent_code입니다.",
                    "allowed_codes": list(prompt.agent_codes),
                },
            )

        # thread_id는 한 번의 채팅/HITL 흐름을 식별한다. session_id는
        # 같은 사원이 동일 에이전트와 이어 가는 여러 질문을 하나로 묶는다.
        thread_id = str(uuid4())
        session_id = body.session_id or thread_id
        logger.info(
            "======== 통합 채팅 요청 도착 | 유형=신규질문 | "
            "thread_id=%s | 사원번호=%s | session_id=%s | "
            "프론트에이전트=%s",
            thread_id,
            body.employee_id,
            session_id,
            frontend_code or "선택하지 않음",
        )

        try:
            result = await graph.start(
                thread_id=thread_id,
                employee_id=body.employee_id,
                session_id=session_id,
                message=body.message,
                frontend_agent_code=frontend_code,
            )
        except HitlStateStoreUnavailableError as exc:
            raise _hitl_store_unavailable() from exc

        logger.info(
            "======== 통합 채팅 처리 결과 | 유형=신규질문 | "
            "thread_id=%s | 상태=%s",
            thread_id,
            result.status,
        )
        return _response(result)

    @app.post("/v1/chat/stream")
    @timed("스트리밍 채팅 요청 접수")
    async def stream_chat(
        body: StreamingChatRequest,
        authorization: str = Header(...),
    ) -> StreamingResponse:
        """최종 WAS 연계 요청을 받아 간결한 SSE 이벤트로 처리한다.

        외부 응답에는 프론트 동작에 필요한 식별자·메시지·답변·action만 보낸다.
        마스터/서브에이전트 분류, 시나리오, 파라미터와 MCP 세부 결과는 모두
        request_id 문맥이 붙은 서버 로그로 추적한다.
        """

        if not authorization.casefold().startswith("bearer ") or not (
            authorization[7:].strip()
        ):
            raise HTTPException(
                status_code=401,
                detail={
                    "code": "INVALID_AUTHORIZATION",
                    "message": "Authorization 헤더에 Bearer 토큰이 필요합니다.",
                },
            )
        access_token = authorization[7:].strip()

        # 명세상 null을 허용하는 식별자는 서버에서 안전하게 보완한 뒤 그 값을
        # SSE로 즉시 반환한다. 이후 같은 응답 안에서는 보완된 값만 사용한다.
        session_id = body.session_id or str(uuid4())
        thread_id = body.thread_id or str(uuid4())
        normalized_endpoint = (body.endpoint or settings.project_code).casefold()
        if normalized_endpoint != settings.project_code:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_ENDPOINT",
                    "message": "이 서비스에서 처리할 수 없는 endpoint입니다.",
                    "allowed_endpoint": settings.project_code,
                },
            )
        normalized_agent = (
            body.agent_code.upper()
            if body.agent_code is not None
            else None
        )
        if (
            normalized_agent is not None
            and normalized_agent not in prompt.agent_codes
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "INVALID_AGENT_CODE",
                    "message": "등록되지 않은 agent_code입니다.",
                    "allowed_codes": list(prompt.agent_codes),
                },
            )

        request_id = f"{settings.project_code}:{uuid4()}"
        employee_id = _resolve_employee_id(body.user, session_id)
        # MCP에서 필요한 요청 정보를 한 객체로 전달한다. access_token은 그래프
        # 실행 중에만 사용하고 Redis HITL/대화이력에는 저장하지 않는다.
        request_context = {
            "access_token": access_token,
            "endpoint": normalized_endpoint,
            "recruitment_org_type_code": (
                _derive_recruitment_org_type_code(
                    body.user.id if body.user is not None else None
                )
            ),
            "user": (
                body.user.model_dump(mode="json")
                if body.user is not None
                else {"id": None, "deptcode": None, "deptname": None}
            ),
        }
        user_message_id = str(uuid4())
        assistant_message_id = str(uuid4())
        with log_context(
            request_id=request_id,
            session_id=session_id,
            thread_id=thread_id,
            user_id=employee_id,
        ):
            logger.info(
                "======== 요청 도착 | 유형=%s | endpoint=%s | "
                "선택에이전트=%s | 인증=Bearer확인완료",
                "HITL재진입" if body.is_hitl_continuation else "신규질문",
                normalized_endpoint,
                normalized_agent,
            )
            logger.info(
                "======== 사용자 정보 | 사번=%s | 부서코드=%s | 부서명=%s | "
                "유치조직구분코드=%s | 익명내부식별자=%s",
                body.user.id if body.user else None,
                body.user.deptcode if body.user else None,
                body.user.deptname if body.user else None,
                request_context["recruitment_org_type_code"],
                body.user is None or body.user.id is None,
            )
            logger.info(
                "======== 사용자 질문 | 길이=%d | 내용=%s",
                len(body.message),
                body.message,
            )
            logger.info(
                "======== 사용자 추가입력 | 개수=%d | 값=%s",
                len(body.human_input or []),
                [
                    item.model_dump(mode="json")
                    for item in (body.human_input or [])
                ],
            )

        async def event_stream():
            with log_context(
                request_id=request_id,
                session_id=session_id,
                thread_id=thread_id,
                user_id=employee_id,
            ):
                started_at = perf_counter()
                logger.info("======== SSE 시작 | 이벤트=request_id")
                yield encode_sse("request_id", request_id)
                logger.info("======== SSE 식별자 전달 | 이벤트=session_id")
                yield encode_sse("session_id", session_id)
                logger.info("======== SSE 식별자 전달 | 이벤트=thread_id")
                yield encode_sse("thread_id", thread_id)

                initial_content: Any = body.message
                if body.is_hitl_continuation:
                    initial_content = {
                        "message": body.message,
                        "humanInput": [
                            item.model_dump(mode="json")
                            for item in (body.human_input or [])
                        ],
                    }
                initial_messages = [
                    {
                        "role": "user",
                        "id": user_message_id,
                        "content": initial_content,
                    },
                    {
                        "role": "assistant",
                        "id": assistant_message_id,
                        "parentMessageId": user_message_id,
                        "content": "",
                        "metadata": {"renderables": []},
                    },
                ]
                logger.info(
                    "======== SSE 메시지 트리 전달 | 이벤트=messages | 메시지개수=%d",
                    len(initial_messages),
                )
                yield encode_sse("messages", initial_messages)

                current_stage = "LANGGRAPH_HITL_RESUME" if (
                    body.is_hitl_continuation
                ) else "LANGGRAPH_NEW_CHAT"
                try:
                    graph: MasterIntentGraph = app.state.graph
                    if body.is_hitl_continuation:
                        logger.info(
                            "======== LangGraph 호출 | 모드=HITL재진입 | 입력=%s",
                            body.to_hitl_value(),
                        )
                        result = await graph.resume(
                            thread_id=thread_id,
                            value=body.to_hitl_value(),
                            expected_employee_id=employee_id,
                            expected_session_id=session_id,
                            request_context=request_context,
                        )
                    else:
                        logger.info(
                            "======== LangGraph 호출 | 모드=신규질문 | "
                            "프론트에이전트=%s",
                            normalized_agent,
                        )
                        result = await graph.start(
                            thread_id=thread_id,
                            employee_id=employee_id,
                            session_id=session_id,
                            message=body.message,
                            frontend_agent_code=normalized_agent,
                            request_context=request_context,
                        )

                    logger.info(
                        "======== 마스터 분류 결과 | 상태=%s | 분류유형=%s | "
                        "에이전트=%s | 보정질문=%s",
                        result.status,
                        result.classification.classification_type.value,
                        result.classification.agent_code,
                        result.classification.refined_query,
                    )
                    logger.info(
                        "======== 서브에이전트 결과 | 값=%s",
                        result.subagent.model_dump(mode="json")
                        if result.subagent is not None
                        else None,
                    )
                    logger.info(
                        "======== MCP 내부 결과 준비 완료 | 개수=%d | "
                        "결과형식=dict배열 | 프론트MCP결과전송=안함 | "
                        "사용처=최종답변생성 | 정제결과=%s | "
                        "원본결과로그=생략",
                        len(result.mcp_results or []),
                        [
                            {
                                "tool_name": item.tool_name,
                                "request_id": item.request_id,
                                "succeeded": item.succeeded,
                                "result_format": item.result_format,
                                "formatted_result": item.formatted_result,
                                "error": item.error,
                            }
                            for item in (result.mcp_results or [])
                        ],
                    )

                    if result.status == "INPUT_REQUIRED":
                        current_stage = "SSE_ACTION"
                        action = build_action_event(thread_id, result.interrupt)
                        logger.info(
                            "======== 사용자 입력 요청 | 이벤트=action | 값=%s",
                            action,
                        )
                        yield encode_sse("action", action)
                    else:
                        current_stage = "ANSWER_PREPARE"
                        prepared = await answer_service.prepare(result)
                        logger.info(
                            "======== 최종 답변 생성 시작 | 모드=%s | "
                            "출처문서개수=%d",
                            prepared.mode,
                            len(prepared.source_documents),
                        )
                        answer_parts: list[str] = []
                        if prepared.source_documents:
                            logger.info(
                                "======== RAG 출처 문서 단건 전달 | "
                                "이벤트=sourceDocuments | 전송방식=JSON배열1회 | "
                                "MCP원본스트리밍=아님 | 개수=%d | "
                                "문서내용로그=생략",
                                len(prepared.source_documents),
                            )
                            yield encode_sse(
                                "sourceDocuments",
                                prepared.source_documents,
                            )

                        current_stage = "ANSWER_STREAM"
                        token_count = 0
                        async for token in prepared.tokens:
                            if not token:
                                continue
                            token_count += 1
                            answer_parts.append(token)
                            # 프론트에는 생성되는 즉시 token SSE를 전달한다. 다만
                            # chunk별 로그는 전체 답변을 읽기 어렵게 만들기 때문에
                            # 기록하지 않고, 스트림 완료 후 결합된 답변만 한 번 남긴다.
                            yield encode_sse("token", token)

                        full_answer = "".join(answer_parts)
                        logger.info(
                            "======== 최종 답변 전체 로그 | SSE토큰이벤트개수=%d | "
                            "답변길이=%d | 전체답변=%s",
                            token_count,
                            len(full_answer),
                            full_answer,
                        )

                        # token 스트리밍이 끝난 뒤 같은 assistant_message_id로 완성
                        # 메시지를 다시 보낸다. 프론트는 기존 말풍선을 갱신하고,
                        # 본문과 분리된 renderables를 표·차트·카드 등으로 렌더링한다.
                        completed_messages = [
                            initial_messages[0],
                            {
                                "role": "assistant",
                                "id": assistant_message_id,
                                "parentMessageId": user_message_id,
                                "content": full_answer,
                                "metadata": {
                                    "renderables": prepared.renderables,
                                },
                            },
                        ]
                        logger.info(
                            "======== SSE 완성 메시지 전달 | 이벤트=messages | "
                            "답변길이=%d | 확장데이터개수=%d | 메시지ID=%s",
                            len(full_answer),
                            len(prepared.renderables),
                            assistant_message_id,
                        )
                        yield encode_sse("messages", completed_messages)

                        # 예외 응답은 이력에 저장하지 않고 정상 업무 PASS의 답변만
                        # 보정 질문과 같은 사원·세션·에이전트 범위에 저장한다.
                        if (
                            result.status == "PASS"
                            and result.classification.agent_code is not None
                            and full_answer
                        ):
                            current_stage = "REDIS_ASSISTANT_HISTORY_SAVE"
                            saved = await history_store.append_message(
                                employee_id=employee_id,
                                session_id=session_id,
                                agent_code=result.classification.agent_code,
                                role="assistant",
                                content=full_answer,
                                message_id=assistant_message_id,
                                metadata={
                                    "renderables": prepared.renderables,
                                },
                            )
                            logger.info(
                                "======== 최종 답변 Redis 저장 결과 | 저장=%s | "
                                "에이전트=%s | 답변길이=%d",
                                saved,
                                result.classification.agent_code,
                                len(full_answer),
                            )
                        else:
                            logger.info(
                                "======== 최종 답변 Redis 저장 생략 | 상태=%s | "
                                "에이전트=%s | 답변존재=%s",
                                result.status,
                                result.classification.agent_code,
                                bool(full_answer),
                            )

                    duration = round(perf_counter() - started_at, 3)
                    current_stage = "SSE_COMPLETE"
                    logger.info(
                        "======== 처리시간 전달 | 이벤트=duration | %.3f초",
                        duration,
                    )
                    yield encode_sse("duration", {"seconds": duration})
                    logger.info(
                        "======== 정상 종료 전달 | 이벤트=end | 상태=%s",
                        result.status,
                    )
                    yield encode_sse("end", {"status": result.status})
                    logger.info(
                        "======== 요청 처리 전체 완료 | 상태=%s | 소요시간=%.3f초",
                        result.status,
                        duration,
                    )
                except asyncio.CancelledError:
                    logger.info(
                        "======== 사용자 연결 중단 | 현재까지소요시간=%.3f초",
                        perf_counter() - started_at,
                    )
                    raise
                except HitlStateNotFoundError:
                    duration = round(perf_counter() - started_at, 3)
                    logger.warning(
                        "!!!!!!!! 처리 중단 진단 | 실패단계=%s | "
                        "코드위치=app/api.py:event_stream -> "
                        "app/graph.py:MasterIntentGraph.resume | "
                        "오류코드=HITL_STATE_NOT_FOUND | 가능한원인=thread_id 만료, "
                        "이미 완료·삭제된 상태, 사원/session 범위 불일치 | "
                        "확인및수정=Redis HITL 키 TTL과 프론트 thread_id/session_id를 "
                        "확인 | 자동재시도횟수=0 | 소요시간=%.3f초",
                        current_stage,
                        duration,
                    )
                    yield encode_sse("duration", {"seconds": duration})
                    yield encode_sse(
                        "error",
                        {
                            "code": "HITL_STATE_NOT_FOUND",
                            "message": (
                                "이어 갈 입력 상태를 찾을 수 없습니다. 새 질문으로 "
                                "다시 시작해 주세요."
                            ),
                        },
                    )
                except Exception as exc:
                    duration = round(perf_counter() - started_at, 3)
                    log_failure_diagnostic(
                        stage=f"SSE 요청 처리/{current_stage}",
                        code_location="app/api.py:create_app.event_stream",
                        exc=exc,
                        likely_cause=(
                            "현재단계 또는 직전 LangGraph 노드의 외부 호출·상태 "
                            "변환·답변 스트리밍 오류"
                        ),
                        corrective_action=(
                            "같은 request_id의 가장 먼저 출력된 '실패 진단' 로그에서 "
                            "코드위치와 확인및수정 항목을 확인하세요. 이 요청은 "
                            "자동 재시도하지 않았습니다."
                        ),
                        retry_count=0,
                        context={
                            "current_stage": current_stage,
                            "endpoint": normalized_endpoint,
                            "frontend_agent_code": normalized_agent,
                            "is_hitl_continuation": body.is_hitl_continuation,
                            "duration_seconds": duration,
                        },
                    )
                    # URL, 토큰, 스택과 내부 분류 결과는 프론트로 노출하지 않는다.
                    yield encode_sse("duration", {"seconds": duration})
                    yield encode_sse(
                        "error",
                        {
                            "code": "STREAM_PROCESSING_ERROR",
                            "message": (
                                "요청 처리 중 오류가 발생했습니다. request_id를 "
                                "관리자에게 전달해 주세요."
                            ),
                        },
                    )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _require_bearer_token(authorization: str) -> str:
    """Bearer 헤더를 검증하고 실제 토큰 문자열을 반환한다."""

    if not authorization.casefold().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_AUTHORIZATION",
                "message": "Authorization 헤더에 Bearer 토큰이 필요합니다.",
            },
        )
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "INVALID_AUTHORIZATION",
                "message": "Authorization Bearer 토큰 값이 비어 있습니다.",
            },
        )
    return token


def _validate_history_agent_code(
    agent_code: str,
    allowed_codes: tuple[str, ...],
) -> str:
    """이력 API의 에이전트 코드를 정규화하고 등록 여부를 검증한다."""

    normalized = agent_code.strip().upper()
    if normalized not in allowed_codes:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_AGENT_CODE",
                "message": "등록되지 않은 agent_code입니다.",
                "allowed_codes": list(allowed_codes),
            },
        )
    return normalized


def _hitl_store_unavailable() -> HTTPException:
    """HITL Redis 장애에 사용할 공통 503 응답을 만든다."""

    logger.exception(
        "======== HITL 처리 실패 | 일반 Redis 저장소에 연결할 수 없음"
    )
    return HTTPException(
        status_code=503,
        detail={
            "code": "HITL_STATE_STORE_UNAVAILABLE",
            "message": (
                "HITL 상태 저장소에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요."
            ),
            "action": "RETRY",
        },
    )


def _resolve_employee_id(
    user: StreamingUser | None,
    session_id: str,
) -> str:
    """Redis 범위에 사용할 사번 또는 세션별 익명 식별자를 반환한다.

    최종 연계 명세는 user와 user.id의 null을 허용한다. null을 공통 문자열로
    저장하면 서로 다른 사용자의 이력이 섞일 수 있으므로 session_id 해시를 붙인
    내부 값으로 격리한다. 실제 사번이 들어오면 변환 없이 그대로 사용한다.
    """

    if user is not None and user.id is not None:
        return user.id
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    return f"anonymous_{digest}"


def _code_serving_identifier(value: Any, *, prefix: str) -> str:
    """외부 코드서빙 식별자를 내부 키에 사용할 수 있는 값으로 정규화한다."""

    if isinstance(value, str):
        normalized = value.strip()
        if normalized and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", normalized):
            return normalized
        if normalized:
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
            return f"{prefix}_{digest}"
    return f"{prefix}_{uuid4().hex}"


def _derive_recruitment_org_type_code(
    employee_id: str | None,
) -> str | None:
    """사원번호 형식으로 유치조직구분코드를 계산한다.

    - 일반(11): S + 숫자 6자리
    - 제휴(12): K + 숫자 7자리
    - 복합(13): U + 숫자 6자리

    규칙에 맞지 않는 사원번호는 임의 분류하지 않고 None을 반환한다.
    """

    if employee_id is None:
        return None
    normalized = employee_id.strip().upper()
    patterns = (
        (r"S\d{6}", "11"),
        (r"K\d{7}", "12"),
        (r"U\d{6}", "13"),
    )
    for pattern, code in patterns:
        if re.fullmatch(pattern, normalized):
            return code
    return None


def _response(result: MasterResult) -> ChatResponse:
    """내부 그래프 결과를 외부 HTTP 응답 모델로 변환한다."""

    return ChatResponse(
        status=result.status,
        thread_id=result.thread_id,
        classification=result.classification,
        subagent=result.subagent,
        mcp=result.mcp,
        mcp_results=result.mcp_results or [],
        # 필드 이름은 기존 프론트 계약 호환을 위해 interrupt를 유지한다.
        # 실제 구현은 LangGraph interrupt가 아니라 Redis 기반 입력 요청이다.
        interrupt=result.interrupt,
    )
