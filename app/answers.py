"""MCP 결과와 예외 분류를 최종 사용자 답변 스트림으로 변환한다."""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.answerability import (
    AnswerabilityService,
    create_answerability_service,
)
from app.config import Settings
from app.graph import MasterResult
from app.mcp.models import MCP_SAFE_ERROR_MESSAGE, McpExecutionResult
from app.observability import log_failure_diagnostic, logger, timed
from app.rag_policies import (
    filter_documents_by_retrieval_score,
    get_rag_policy,
)
from app.reranking import (
    RerankingService,
    create_reranking_service,
)
from app.subagents.fixed_responses import get_subagent_fixed_response
from app.streaming import split_text


@dataclass(frozen=True)
class AnswerPromptBundle:
    """최종 답변 생성 방식과 RAG 프롬프트 설정."""

    version: str
    rag_system_prompt: str
    temperature: float
    agent_response_modes: dict[str, str]
    default_response_mode: str
    exception_answers: dict[str, str]
    answerability_system_prompt: str = (
        "검색 문서만으로 사용자 질문에 답변 가능한지 판별하고 "
        "JSON Schema 형식으로 반환하세요."
    )


@dataclass(frozen=True)
class PreparedAnswer:
    """SSE 전송 직전에 준비된 본문 스트림과 메시지 확장 데이터."""

    mode: str
    source_documents: list[dict[str, Any]]
    renderables: list[dict[str, Any]]
    tokens: AsyncIterator[str]


class AnswerService(Protocol):
    """API가 답변 생성 방식의 세부 구현에 의존하지 않게 하는 인터페이스."""

    async def prepare(self, result: MasterResult) -> PreparedAnswer: ...

    async def aclose(self) -> None: ...


class AnswerPromptLoader:
    """버전 관리되는 최종 답변 프롬프트와 응답 모드를 로드한다."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = (
            root
            or Path(__file__).resolve().parents[1]
            / "prompts"
            / "answer-generation"
        )

    @timed("최종 답변 프롬프트 로딩")
    def load(self, requested_version: str | None = None) -> AnswerPromptBundle:
        active = yaml.safe_load(
            (self._root / "active.yaml").read_text(encoding="utf-8")
        )
        version = requested_version or str(active["active_version"])
        version_root = self._root / version
        manifest = yaml.safe_load(
            (version_root / "manifest.yaml").read_text(encoding="utf-8")
        )
        prompt_path = version_root / str(manifest["rag_system_prompt"])
        answerability_prompt_path = version_root / str(
            manifest["answerability_system_prompt"]
        )
        modes = {
            str(code).upper(): str(mode)
            for code, mode in manifest.get("agent_response_modes", {}).items()
        }
        allowed_modes = {"fixed_data", "rag"}
        invalid_modes = set(modes.values()) - allowed_modes
        default_mode = str(manifest.get("default_response_mode", "fixed_data"))
        if default_mode not in allowed_modes or invalid_modes:
            raise ValueError(
                "지원하지 않는 최종 답변 모드가 있습니다: "
                f"{sorted(invalid_modes | {default_mode} - allowed_modes)}"
            )
        bundle = AnswerPromptBundle(
            version=version,
            rag_system_prompt=prompt_path.read_text(encoding="utf-8").strip(),
            answerability_system_prompt=answerability_prompt_path.read_text(
                encoding="utf-8"
            ).strip(),
            temperature=float(manifest.get("model", {}).get("temperature", 0)),
            agent_response_modes=modes,
            default_response_mode=default_mode,
            exception_answers={
                str(code): str(answer)
                for code, answer in manifest.get("exception_answers", {}).items()
            },
        )
        logger.info(
            "======== 최종 답변 프롬프트 로딩 완료 | 버전=%s | 모드=%s",
            bundle.version,
            bundle.agent_response_modes,
        )
        return bundle


class DefaultAnswerService:
    """고정 데이터 답변과 문서 기반 RAG 답변을 하나의 인터페이스로 제공한다."""

    def __init__(
        self,
        settings: Settings,
        prompt: AnswerPromptBundle,
        llm: Any | None = None,
        reranker: RerankingService | None = None,
        answerability_service: AnswerabilityService | None = None,
    ) -> None:
        self._prompt = prompt
        self._model = settings.genos_model
        self._endpoint = settings.genos_openai_base_url
        self._max_retries = settings.llm_max_retries
        self._reranker = reranker or create_reranking_service(settings)
        self._answerability = (
            answerability_service
            or create_answerability_service(
                settings,
                prompt.answerability_system_prompt,
            )
        )
        # 테스트에서는 토큰이 없으므로 네트워크를 호출하지 않는 RAG 대체 답변을
        # 사용한다. 운영에서는 마스터와 같은 GenOS OpenAI 호환 엔드포인트다.
        self._llm = llm
        if self._llm is None and settings.genos_bearer_token:
            self._llm = ChatOpenAI(
                base_url=settings.genos_openai_base_url,
                model=settings.genos_model,
                api_key=settings.genos_bearer_token,
                temperature=prompt.temperature,
                max_retries=settings.llm_max_retries,
            )
        logger.info(
            "======== 최종 답변 서비스 준비 | LLM사용=%s | 모델=%s | "
            "자동재시도=%d회 | 코드위치=app/answers.py:"
            "DefaultAnswerService._stream_rag",
            self._llm is not None,
            self._model,
            self._max_retries,
        )

    async def prepare(self, result: MasterResult) -> PreparedAnswer:
        """세부 시나리오별 정책에 따라 고정 데이터와 RAG 답변을 조합한다."""

        if result.status == "EXCEPTION":
            code = result.classification.classification_type.value
            text = self._prompt.exception_answers.get(
                code,
                "요청을 처리할 수 없습니다. 질문 내용을 확인해 주세요.",
            )
            logger.info("======== 고정 예외 답변 선택 | 예외유형=%s", code)
            return PreparedAnswer("exception", [], [], _stream_fixed(text))

        agent_code = (result.classification.agent_code or "").upper()
        mcp_results = result.mcp_results or []
        matches = result.subagent.matches if result.subagent else []

        # 과거 단일 응답이나 서브에이전트가 없는 테스트 결과도 기존 방식으로
        # 처리한다. 실제 등록 시나리오는 아래 세부 시나리오 정책 경로를 탄다.
        if not matches:
            return await self._prepare_legacy_result(result, agent_code)

        answer_parts: list[str | AsyncIterator[str]] = []
        source_documents: list[dict[str, Any]] = []
        renderables: list[dict[str, Any]] = []
        used_modes: set[str] = set()
        fixed_success_count = 0

        mcp_cursor = 0
        for index, match in enumerate(matches):
            detail_code = match.detail_scenario_code.upper()
            fixed_response = get_subagent_fixed_response(
                agent_code,
                detail_code,
            )
            if fixed_response is not None:
                used_modes.add("fixed_data")
                answer_parts.append(fixed_response.message)
                logger.info(
                    "======== 서브에이전트 고정 답변 준비\n"
                    "순번=%d\n에이전트=%s\n세부시나리오=%s\n"
                    "MCP결과사용=아니오\n고정답변=%s",
                    index + 1,
                    agent_code,
                    detail_code,
                    fixed_response.message,
                )
                continue

            if mcp_cursor >= len(mcp_results):
                logger.info(
                    "======== 세부 시나리오 MCP 결과 없음\n"
                    "순번=%d\n에이전트=%s\n세부시나리오=%s",
                    index + 1,
                    agent_code,
                    detail_code,
                )
                used_modes.add("fixed_data")
                answer_parts.append(
                    "조회 결과가 없습니다. 잠시 후 다시 시도해 주세요."
                )
                continue

            mcp = mcp_results[mcp_cursor]
            mcp_cursor += 1
            if mcp.user_message:
                used_modes.add("fixed_data")
                answer_parts.append(mcp.user_message)
                logger.info(
                    "======== MCP 업무결과 고정 답변 선택\n"
                    "순번=%d\n에이전트=%s\n세부시나리오=%s\n"
                    "MCP도구=%s\n업무결과=%s\n업무코드=%s\n사용자답변=%s",
                    index + 1,
                    agent_code,
                    detail_code,
                    mcp.tool_name,
                    mcp.outcome,
                    mcp.business_code,
                    mcp.user_message,
                )
                continue
            policy = get_rag_policy(agent_code, detail_code)
            # 복수 문서 시나리오에서는 전체 질문보다 각 match가 추출한 검색문이
            # 해당 문서 묶음과의 관련성·답변 가능성을 더 정확히 나타낸다.
            scenario_query = str(
                match.parameters.get("search_query")
                or result.classification.refined_query
            ).strip()
            logger.info(
                "======== 세부 시나리오 답변 방식 결정\n"
                "순번=%d\n에이전트=%s\n세부시나리오=%s\n답변방식=%s\n"
                "MCP도구=%s\n시나리오검색문=%s",
                index + 1,
                agent_code,
                detail_code,
                "rag" if policy else "fixed_data",
                mcp.tool_name,
                scenario_query,
            )

            if policy is None:
                used_modes.add("fixed_data")
                if mcp.succeeded:
                    fixed_success_count += 1
                answer_parts.append(
                    _build_fixed_data_answer(result, mcp_results=[mcp])
                )
                renderables.extend(
                    _collect_message_renderables(result, mcp_results=[mcp])
                )
                continue

            used_modes.add("rag")
            documents = _build_source_documents_from_results([mcp])
            documents = filter_documents_by_retrieval_score(
                documents=documents,
                threshold=policy.retrieval_score_threshold,
                agent_code=agent_code,
                detail_scenario_code=detail_code,
            )
            if not documents:
                logger.info(
                    "======== RAG 고정 답변 선택\n"
                    "에이전트=%s\n세부시나리오=%s\n"
                    "사유=검색 임계점수 통과 문서 없음",
                    agent_code,
                    detail_code,
                )
                answer_parts.append(policy.no_documents_answer)
                continue

            if policy.reranking_enabled:
                documents = await self._reranker.rerank(
                    query=scenario_query,
                    documents=documents,
                    top_n=policy.reranking_top_n,
                    score_threshold=policy.reranking_score_threshold,
                )
                if not documents:
                    logger.info(
                        "======== RAG 고정 답변 선택\n"
                        "에이전트=%s\n세부시나리오=%s\n"
                        "사유=Reranking 임계점수 통과 문서 없음",
                        agent_code,
                        detail_code,
                    )
                    answer_parts.append(policy.no_documents_answer)
                    continue
            else:
                logger.info(
                    "======== 시나리오 Reranking 생략\n"
                    "에이전트=%s\n세부시나리오=%s\n정책활성화=false",
                    agent_code,
                    detail_code,
                )

            if policy.answerability_check_enabled:
                answerable = await self._answerability.is_answerable(
                    query=scenario_query,
                    documents=documents,
                )
                if not answerable:
                    logger.info(
                        "======== RAG 고정 답변 선택\n"
                        "에이전트=%s\n세부시나리오=%s\n"
                        "사유=LLM 문서 답변 가능성 판별 false",
                        agent_code,
                        detail_code,
                    )
                    answer_parts.append(policy.not_answerable_answer)
                    continue

            source_documents.extend(documents)
            answer_parts.append(
                self._stream_rag(
                    query=scenario_query,
                    documents=documents,
                )
            )

        if not answer_parts:
            answer_parts.append("조회 결과가 없습니다. 잠시 후 다시 시도해 주세요.")
            used_modes.add("fixed_data")

        mode = "mixed" if len(used_modes) > 1 else next(iter(used_modes))
        logger.info(
            "======== 최종 답변 준비 완료\n"
            "에이전트=%s\n답변모드=%s\n답변파트개수=%d\n"
            "MCP결과개수=%d\n정제성공개수=%d\n출처문서개수=%d\n"
            "확장데이터개수=%d",
            agent_code,
            mode,
            len(answer_parts),
            len(mcp_results),
            fixed_success_count,
            len(source_documents),
            len(renderables),
        )
        return PreparedAnswer(
            mode,
            source_documents,
            renderables,
            _stream_answer_parts(answer_parts),
        )

    async def _prepare_legacy_result(
        self,
        result: MasterResult,
        agent_code: str,
    ) -> PreparedAnswer:
        """세부 매칭이 없는 이전 테스트·호출 계약을 안전하게 유지한다."""

        mode = self._prompt.agent_response_modes.get(
            agent_code,
            self._prompt.default_response_mode,
        )
        if mode == "rag":
            documents = _build_source_documents(result)
            documents = await self._reranker.rerank(
                query=result.classification.refined_query,
                documents=documents,
            )
            return PreparedAnswer(
                mode,
                documents,
                [],
                self._stream_rag(
                    query=result.classification.refined_query,
                    documents=documents,
                ),
            )
        text = _build_fixed_data_answer(result)
        return PreparedAnswer(
            mode,
            [],
            _collect_message_renderables(result),
            _stream_fixed(text),
        )

    async def aclose(self) -> None:
        """애플리케이션 종료 시 reranker HTTP 연결을 닫는다."""

        await self._reranker.aclose()

    async def _stream_rag(
        self,
        *,
        query: str,
        documents: list[dict[str, Any]],
    ) -> AsyncIterator[str]:
        """MCP dict를 문서로 넣어 GenOS 답변 토큰을 스트리밍한다."""

        document_text = json.dumps(documents, ensure_ascii=False, default=str)
        human_prompt = (
            f"사용자 질문:\n{query}\n\n"
            f"참고 문서:\n{document_text}"
        )
        if self._llm is None:
            fallback = (
                "테스트 RAG 답변입니다. 조회된 참고 문서를 기반으로 자격기준을 "
                f"안내합니다. 참고 문서: {document_text}"
            )
            for chunk in split_text(fallback):
                yield chunk
            return

        logger.info(
            "======== RAG LLM 스트리밍 호출 시작 | "
            "코드위치=app/answers.py:DefaultAnswerService._stream_rag | "
            "엔드포인트=%s | 모델=%s | 문서개수=%d | 문서JSON길이=%d | "
            "질문=%s | 자동재시도=%d회",
            self._endpoint,
            self._model,
            len(documents),
            len(document_text),
            query,
            self._max_retries,
        )
        try:
            async for chunk in self._llm.astream(
                [
                    SystemMessage(content=self._prompt.rag_system_prompt),
                    HumanMessage(content=human_prompt),
                ]
            ):
                text = _message_content_text(getattr(chunk, "content", ""))
                if text:
                    yield text
        except Exception as exc:
            # 스트리밍 실패 후 ainvoke로 우회 호출하지 않는다. 최초 오류를 즉시
            # 상위 SSE 처리기로 전달해야 실제 장애 원인을 가리지 않을 수 있다.
            log_failure_diagnostic(
                stage="최종 RAG 답변 LLM 스트리밍",
                code_location="app/answers.py:DefaultAnswerService._stream_rag",
                exc=exc,
                likely_cause=(
                    "GenOS LLM 스트리밍 미지원, 연결·인증·모델 오류, 응답 시간 "
                    "초과 또는 스트림 도중 연결 종료"
                ),
                corrective_action=(
                    "GenOS serving이 stream=true 응답을 지원하는지 확인하고 "
                    ".env의 GENOS_URL/GENOS_SERVING_ID/GENOS_MODEL/"
                    "GENOS_BEARER_TOKEN 및 prompts/answer-generation을 확인하세요."
                ),
                retry_count=self._max_retries,
                context={
                    "endpoint": self._endpoint,
                    "model": self._model,
                    "document_count": len(documents),
                    "document_json_length": len(document_text),
                    "query": query,
                },
            )
            raise
        logger.info(
            "======== RAG LLM 스트리밍 호출 완료 | 문서개수=%d | "
            "자동재시도=%d회",
            len(documents),
            self._max_retries,
        )


async def _stream_fixed(text: str) -> AsyncIterator[str]:
    """고정 문자열을 작은 token 이벤트 단위로 반환한다."""

    for chunk in split_text(text):
        yield chunk


async def _stream_answer_parts(
    parts: list[str | AsyncIterator[str]],
) -> AsyncIterator[str]:
    """복수 세부 시나리오의 고정 문자열과 LLM 스트림을 선택 순서로 합친다."""

    for index, part in enumerate(parts):
        if index:
            yield "\n\n"
        if isinstance(part, str):
            async for chunk in _stream_fixed(part):
                yield chunk
            continue
        async for chunk in part:
            yield chunk


def _build_fixed_data_answer(
    result: MasterResult,
    *,
    mcp_results: list[McpExecutionResult] | None = None,
) -> str:
    """복수 MCP 정제 답변을 서브에이전트 매칭 순서대로 하나로 조합한다."""

    selected_results = (
        mcp_results if mcp_results is not None else (result.mcp_results or [])
    )
    if not selected_results:
        return "조회 결과가 없습니다. 잠시 후 다시 시도해 주세요."
    rows: list[str] = []
    adapter_order: list[str] = []
    succeeded_count = 0
    failed_count = 0
    for index, mcp in enumerate(selected_results, start=1):
        if mcp.user_message:
            rows.append(mcp.user_message)
            # NO_DATA는 MCP 호출 자체는 정상 종료된 업무 결과다. 운영 로그에서
            # 기술 오류(ERROR)와 섞이지 않도록 실패 건수에는 포함하지 않는다.
            if mcp.outcome == "ERROR":
                failed_count += 1
            else:
                succeeded_count += 1
            continue
        if not mcp.succeeded:
            # 내부 오류 원문은 MCP 실행 로그에만 남고 최종 사용자 답변에는
            # 노출하지 않는다.
            rows.append(MCP_SAFE_ERROR_MESSAGE)
            failed_count += 1
            continue

        formatted = mcp.formatted_result
        if mcp.result_format != "query.v1" or not isinstance(formatted, dict):
            raise ValueError(
                "고정 데이터 답변에 query.v1 정제 결과가 없습니다: "
                f"tool={mcp.tool_name}, result_format={mcp.result_format}. "
                "app/mcp/result_adapters.py 설정을 확인하세요."
            )
        answer_text = formatted.get("answer_text")
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError(
                "query.v1 formatted_result.answer_text가 비어 있습니다: "
                f"tool={mcp.tool_name}"
            )
        rows.append(answer_text.strip())
        succeeded_count += 1
        adapter_order.append(str(formatted.get("adapter_code", "")))
        logger.info(
            "======== 시나리오별 정제 답변 조합 대기 | 순번=%d | 도구=%s | "
            "결과형식=%s | adapter=%s | 정제답변=%s",
            index,
            mcp.tool_name,
            mcp.result_format,
            formatted.get("adapter_code"),
            answer_text,
        )
    combined_answer = "\n\n".join(rows)
    logger.info(
        "======== 복수 조회 결과 최종 조합 완료 | MCP결과개수=%d | "
        "정제성공개수=%d | 조회실패개수=%d | adapter순서=%s | "
        "조합답변=%s",
        len(selected_results),
        succeeded_count,
        failed_count,
        adapter_order,
        combined_answer,
    )
    return combined_answer


def _build_source_documents(result: MasterResult) -> list[dict[str, Any]]:
    """MCP structuredContent를 프론트와 RAG가 공유하는 문서 배열로 변환한다."""

    return _build_source_documents_from_results(result.mcp_results or [])


def _build_source_documents_from_results(
    mcp_results: list[McpExecutionResult],
) -> list[dict[str, Any]]:
    """선택된 MCP 결과만 프론트와 RAG가 공유하는 문서 배열로 변환한다."""

    documents: list[dict[str, Any]] = []
    for index, mcp in enumerate(mcp_results, start=1):
        if not mcp.succeeded or mcp.result is None:
            continue
        if mcp.tool_name == "databricks_hybrid_search":
            documents.extend(
                _build_databricks_source_documents(
                    mcp_result=mcp.result,
                    tool_name=mcp.tool_name,
                    request_id=mcp.request_id,
                    arguments=mcp.arguments,
                )
            )
            continue
        documents.append(
            {
                "document_id": f"{mcp.request_id}:document:{index}",
                "title": f"{mcp.tool_name} 조회 문서",
                "source": mcp.tool_name,
                "content": mcp.result,
                "metadata": {
                    "mcp_request_id": mcp.request_id,
                    "arguments": mcp.arguments,
                },
            }
        )
    logger.info(
        "======== RAG 참고 문서 변환 완료 | MCP결과개수=%d | "
        "최종문서개수=%d | 문서ID=%s",
        len(mcp_results),
        len(documents),
        [document.get("document_id") for document in documents],
    )
    return documents


def _build_databricks_source_documents(
    *,
    mcp_result: dict[str, Any],
    tool_name: str,
    request_id: str,
    arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Databricks 하이브리드 검색 행을 RAG 문서 계약으로 변환한다.

    가정한 MCP 계약은 ``structuredContent.data``가 문서 객체 배열인 형태다.
    실제 도구의 컬럼명이 달라지면 MCP 호출부가 아니라 이 함수의 필드 매핑만
    변경하면 sourceDocuments와 LLM 참고 문서가 함께 동일하게 바뀐다.
    """

    rows = mcp_result.get("data")
    if not isinstance(rows, list):
        raise ValueError(
            "Databricks MCP structuredContent.data는 list 형식이어야 합니다."
        )

    documents: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(
                "Databricks MCP structuredContent.data 각 항목은 "
                "object 형식이어야 합니다."
            )
        content = row.get("content")
        if not isinstance(content, str) or not content.strip():
            logger.info(
                "======== Databricks 검색 문서 제외 | 순번=%d | "
                "이유=content없음 | 문서ID=%s",
                index,
                row.get("document_id"),
            )
            continue
        document_id = str(
            row.get("document_id") or f"{request_id}:document:{index}"
        )
        title = str(row.get("title") or "자격기준 조회 문서")
        documents.append(
            {
                "document_id": document_id,
                "title": title,
                "source": tool_name,
                "content": content.strip(),
                "metadata": {
                    "mcp_request_id": request_id,
                    "source_uri": row.get("source_uri"),
                    "updated_at": row.get("updated_at"),
                    "score": row.get("score"),
                    "matched_query": row.get("matched_query"),
                    "search_arguments": arguments,
                },
            }
        )

    logger.info(
        "======== Databricks 하이브리드 검색 문서 매핑 완료 | "
        "도구=%s | 조회행=%d | 유효문서=%d",
        tool_name,
        len(rows),
        len(documents),
    )
    return documents


def _collect_message_renderables(
    result: MasterResult,
    *,
    mcp_results: list[McpExecutionResult] | None = None,
) -> list[dict[str, Any]]:
    """복수 시나리오의 renderables를 선택 순서대로 한 배열로 결합한다."""

    combined: list[dict[str, Any]] = []
    selected_results = (
        mcp_results if mcp_results is not None else (result.mcp_results or [])
    )
    for mcp in selected_results:
        formatted = mcp.formatted_result
        if not isinstance(formatted, dict):
            continue
        items = formatted.get("renderables", [])
        if not isinstance(items, list):
            raise ValueError(
                "formatted_result.renderables는 list 형식이어야 합니다: "
                f"tool={mcp.tool_name}"
            )
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(
                    "renderables 각 항목은 object 형식이어야 합니다: "
                    f"tool={mcp.tool_name}"
                )
            combined.append(item)
    logger.info(
        "======== 메시지 확장 데이터 조합 완료 | 개수=%d | 코드=%s",
        len(combined),
        [item.get("code") for item in combined],
    )
    return combined


def _message_content_text(content: Any) -> str:
    """LangChain 모델별 문자열·콘텐츠 블록 응답을 텍스트로 통일한다."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(content) if content is not None else ""


@timed("최종 답변 서비스 생성")
def create_answer_service(settings: Settings) -> AnswerService:
    """운영용 고정 데이터·RAG 통합 답변 서비스를 생성한다."""

    return DefaultAnswerService(settings, AnswerPromptLoader().load())
