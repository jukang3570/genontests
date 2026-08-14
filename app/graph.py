"""일반 Redis HITL 상태를 사용하는 마스터 에이전트 LangGraph 워크플로."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.classifier import IntentClassifier
from app.csv_trace import EmptyTraceRecorder, TraceRecorder
from app.domain import ClassificationType, IntentClassification
from app.history import ChatHistoryStore
from app.hitl import build_hitl_request, validate_ok_signal
from app.hitl_store import (
    HitlStateNotFoundError,
    HitlStateStore,
    HitlStateStoreUnavailableError,
)
from app.mcp.client import EmptyMcpToolExecutor, McpToolExecutor
from app.mcp.models import MCP_SAFE_ERROR_MESSAGE, McpExecutionResult
from app.mcp.payloads import McpParameterInputRequired
from app.mcp.result_adapters import adapt_mcp_result
from app.observability import log_failure_diagnostic, logger, timed
from app.subagents.models import SubagentResult
from app.subagents.fixed_responses import get_subagent_fixed_response
from app.subagents.router import EmptySubagentRouter, SubagentRouter


class MasterState(TypedDict, total=False):
    """한 번의 stateless 그래프 실행에서 노드 사이에 전달되는 상태."""

    # NEW_CHAT은 최초 진입, HITL_RESUME은 Redis 상태 복원 후 재진입을 뜻한다.
    entry_stage: str
    thread_id: str
    message: str
    message_id: str
    employee_id: str
    session_id: str
    frontend_agent_code: str | None
    # 현재 HTTP 요청에서만 사용하는 MCP 런타임 정보다. AccessToken 등이 포함될
    # 수 있으므로 Redis HITL 상태와 CSV에는 저장하지 않는다. HITL 재진입 때는
    # 재진입 HTTP 요청의 최신 컨텍스트를 다시 주입한다.
    request_context: dict[str, Any]
    history: list[dict[str, str]]
    classification: dict[str, Any]
    subagent: dict[str, Any] | None
    mcp: dict[str, Any] | None
    mcp_results: list[dict[str, Any]]
    # 복수 MCP 중 추가 입력이 필요한 순번이다. 재진입 시 완료된 이전 도구를
    # 다시 호출하지 않고 이 순번부터 이어서 실행한다.
    mcp_start_index: int
    # Redis에 저장된 대기 유형이다. HITL 재진입 시 START 조건부 Edge가 이 값을
    # 사용해 유형별 검증 노드로 직접 분기한다.
    hitl_type: str
    human_input: Any
    interrupt: dict[str, Any] | None
    status: str
    approved: bool


@dataclass(frozen=True)
class MasterResult:
    """FastAPI 응답으로 변환하기 전의 프레임워크 독립적인 그래프 결과."""

    status: Literal["PASS", "INPUT_REQUIRED", "EXCEPTION"]
    thread_id: str
    classification: IntentClassification
    interrupt: dict | None = None
    subagent: SubagentResult | None = None
    mcp: McpExecutionResult | None = None
    mcp_results: list[McpExecutionResult] | None = None


class MasterIntentGraph:
    """1차 의도분류, HITL 승인, 등록된 시나리오 서브에이전트를 실행한다.

    LangGraph Checkpointer와 ``interrupt()``는 사용하지 않는다. 사용자 입력이
    필요하면 필요한 상태만 Redis에 저장하고 현재 실행을 END로 종료한다. 재개
    요청에서는 Redis 상태를 복원해 START의 조건부 Edge가 검증 노드로 바로
    분기하므로 이전 LLM 분류 단계를 다시 실행하지 않는다.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        history_store: ChatHistoryStore,
        hitl_store: HitlStateStore,
        history_limit: int = 10,
        subagent_router: SubagentRouter | None = None,
        mcp_executor: McpToolExecutor | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self._classifier = classifier
        self._history_store = history_store
        self._hitl_store = hitl_store
        self._history_limit = history_limit
        self._subagent_router = subagent_router or EmptySubagentRouter()
        self._mcp_executor = mcp_executor or EmptyMcpToolExecutor()
        self._trace_recorder = trace_recorder or EmptyTraceRecorder()

        builder = StateGraph(MasterState)
        builder.add_node("load_history", self._load_history)
        # 현재는 오류 원인을 즉시 확인하기 위해 외부 호출 노드에 RetryPolicy를
        # 지정하지 않는다(None = 한 번만 실행). 추후 재시도가 필요하면 이 세
        # 노드에 langgraph.types.RetryPolicy를 명시적으로 주입하면 된다.
        builder.add_node(
            "classify_intent",
            self._classify_intent,
            # retry_policy=None,
        )
        builder.add_node("verify_selection", self._verify_selection)
        builder.add_node(
            "validate_agent_code_mismatch",
            self._validate_agent_code_mismatch,
        )
        builder.add_node(
            "validate_mcp_parameter_input",
            self._validate_mcp_parameter_input,
        )
        builder.add_node("save_hitl_state", self._save_hitl_state)
        builder.add_node("persist_user_message", self._persist_user_message)
        builder.add_node(
            "run_subagent",
            self._run_subagent,
            # retry_policy=None,
        )
        builder.add_node("call_mcp", self._call_mcp,)# retry_policy=None)
        builder.add_node("clear_hitl_state", self._clear_hitl_state)
        builder.add_node("finish_exception", self._finish_exception)

        # 매 HTTP 요청은 새 그래프 실행이다. Redis에서 상태를 복원한 요청은
        # load_history와 LLM 의도분류를 건너뛰고 HITL 검증 Edge로 진입한다.
        builder.add_conditional_edges(
            START,
            self._route_entry,
            {
                "load_history": "load_history",
                "validate_agent_code_mismatch": (
                    "validate_agent_code_mismatch"
                ),
                "validate_mcp_parameter_input": (
                    "validate_mcp_parameter_input"
                ),
            },
        )
        builder.add_edge("load_history", "classify_intent")
        builder.add_conditional_edges(
            "classify_intent",
            self._after_classification,
            {
                "verify_selection": "verify_selection",
                "finish_exception": "finish_exception",
            },
        )
        builder.add_conditional_edges(
            "verify_selection",
            self._after_input_decision,
            {
                "save_hitl_state": "save_hitl_state",
                "persist_user_message": "persist_user_message",
            },
        )
        builder.add_conditional_edges(
            "validate_agent_code_mismatch",
            self._after_input_decision,
            {
                "save_hitl_state": "save_hitl_state",
                "persist_user_message": "persist_user_message",
            },
        )
        builder.add_conditional_edges(
            "validate_mcp_parameter_input",
            self._after_mcp_parameter_input,
            {
                "save_hitl_state": "save_hitl_state",
                "call_mcp": "call_mcp",
            },
        )
        builder.add_edge("save_hitl_state", END)
        builder.add_conditional_edges(
            "persist_user_message",
            self._after_message_persist,
            {
                "run_subagent": "run_subagent",
                "clear_hitl_state": "clear_hitl_state",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "run_subagent",
            self._after_subagent,
            {
                "call_mcp": "call_mcp",
                "clear_hitl_state": "clear_hitl_state",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "call_mcp",
            self._after_mcp_call,
            {
                "save_hitl_state": "save_hitl_state",
                "clear_hitl_state": "clear_hitl_state",
                "end": END,
            },
        )
        builder.add_edge("clear_hitl_state", END)
        builder.add_edge("finish_exception", END)

        # Checkpointer를 전달하지 않는다. 상태 지속성은 HitlStateStore가 필요한
        # 필드만 일반 Redis SET/GET/DEL로 직접 관리한다.
        self._graph = builder.compile()
        logger.info(
            "======== LangGraph 컴파일 완료 | Checkpointer=사용안함 | "
            "외부호출노드재시도=0회 | 대상노드=%s | "
            "향후설정위치=app/graph.py:MasterIntentGraph.__init__",
            ["classify_intent", "run_subagent", "call_mcp"],
        )

    @staticmethod
    @timed("그래프 진입 단계 라우팅")
    def _route_entry(
        state: MasterState,
    ) -> Literal[
        "load_history",
        "validate_agent_code_mismatch",
        "validate_mcp_parameter_input",
    ]:
        """신규 요청과 Redis HITL 유형에 맞는 시작 Edge를 선택한다."""

        entry_stage = state.get("entry_stage", "NEW_CHAT")
        if entry_stage == "NEW_CHAT":
            logger.info("======== 진입 라우팅 | 신규 채팅 분류 단계")
            return "load_history"
        if entry_stage == "HITL_RESUME":
            hitl_type = state.get("hitl_type")
            if hitl_type == "AGENT_CODE_MISMATCH":
                logger.info(
                    "======== 진입 라우팅 | HITL유형=%s | "
                    "에이전트 코드 변경 검증 단계",
                    hitl_type,
                )
                return "validate_agent_code_mismatch"

            if hitl_type == "MCP_PARAMETER_REQUIRED":
                logger.info(
                    "======== 진입 라우팅 | HITL유형=%s | "
                    "MCP 필수 파라미터 검증 단계",
                    hitl_type,
                )
                return "validate_mcp_parameter_input"

            raise ValueError(
                f"지원하지 않는 Redis HITL 유형입니다: {hitl_type}"
            )
        raise ValueError(f"지원하지 않는 entry_stage입니다: {entry_stage}")

    @timed("Redis 대화이력 조회")
    async def _load_history(self, state: MasterState) -> MasterState:
        """선택 에이전트 또는 session의 최근 에이전트 이력을 조회한다."""

        frontend_code = state.get("frontend_agent_code")
        if frontend_code is None:
            # 프론트에서 에이전트를 선택하지 않았더라도 같은 session의
            # 멀티턴 문맥은 유지해야 한다. 가장 최근에 사용한 에이전트 하나를
            # 선택하여 그 범위의 이력만 읽으므로 서로 다른 에이전트 이력은
            # 섞이지 않는다.
            recent_agent_code, history = (
                await self._history_store.get_recent_for_session(
                    state["employee_id"],
                    state["session_id"],
                    self._history_limit,
                )
            )
            logger.info(
                "======== Redis 이력 조회 완료 | 프론트 에이전트 미선택 | "
                "최근에이전트=%s | 조회개수=%d",
                recent_agent_code or "없음",
                len(history),
            )
            update: MasterState = {"history": history}
            self._trace_recorder.record("대화이력조회완료", {**state, **update})
            return update

        logger.info(
            "======== Redis 이력 조회 요청 | 사원번호=%s | "
            "session_id=%s | 에이전트=%s | 최대개수=%d",
            state["employee_id"],
            state["session_id"],
            frontend_code,
            self._history_limit,
        )
        history = await self._history_store.get_recent(
            state["employee_id"],
            state["session_id"],
            frontend_code,
            self._history_limit,
        )
        logger.info(
            "======== Redis 이력 조회 완료 | 사원번호=%s | "
            "session_id=%s | 에이전트=%s | 조회개수=%d",
            state["employee_id"],
            state["session_id"],
            frontend_code,
            len(history),
        )
        update = {"history": history}
        self._trace_recorder.record("대화이력조회완료", {**state, **update})
        return update

    @timed("마스터 에이전트 1차 의도분류")
    async def _classify_intent(self, state: MasterState) -> MasterState:
        """현재 질문과 같은 범위의 멀티턴 이력으로 GenOS를 한 번 호출한다."""

        logger.info(
            "======== 1차 의도 분류 시작 | 이전대화개수=%d",
            len(state["history"]),
        )
        started_at = perf_counter()
        try:
            result = await self._classifier.classify(
                state["message"],
                state["history"],
                state.get("frontend_agent_code"),
            )
        except Exception as exc:
            log_failure_diagnostic(
                stage="LangGraph 마스터 의도분류 노드",
                code_location="app/graph.py:MasterIntentGraph._classify_intent",
                exc=exc,
                likely_cause=(
                    "마스터 분류기 내부 LLM 호출 또는 구조화 결과 변환 실패"
                ),
                corrective_action=(
                    "바로 앞의 '마스터 1차 의도분류 LLM 호출' 실패 진단과 "
                    "app/classifier.py를 확인하세요."
                ),
                retry_count=0,
                context={
                    "graph_node": "classify_intent",
                    "frontend_agent_code": state.get("frontend_agent_code"),
                    "history_count": len(state.get("history", [])),
                    "message": state.get("message"),
                },
            )
            self._trace_recorder.record(
                "마스터의도분류오류",
                state,
                elapsed_seconds=perf_counter() - started_at,
                error=exc,
            )
            raise
        logger.info(
            "======== 1차 의도 분류 완료 | 분류유형=%s | 에이전트=%s | "
            "원본질문=%s | 보정질문=%s",
            result.classification_type.value,
            result.agent_code,
            state["message"],
            result.refined_query,
        )
        update = {"classification": result.model_dump(mode="json")}
        self._trace_recorder.record(
            "마스터의도분류완료",
            {**state, **update},
            elapsed_seconds=perf_counter() - started_at,
        )
        return update

    @staticmethod
    @timed("의도분류 결과 라우팅")
    def _after_classification(
        state: MasterState,
    ) -> Literal["verify_selection", "finish_exception"]:
        """정상 업무는 코드 비교로, 예외 유형은 예외 종료로 보낸다."""

        classification = IntentClassification.model_validate(
            state["classification"]
        )
        if classification.classification_type == ClassificationType.AGENT:
            logger.info("======== 라우팅 결정 | 에이전트 코드 비교 단계")
            return "verify_selection"

        logger.info(
            "======== 라우팅 결정 | 예외 종료 단계 | 유형=%s",
            classification.classification_type.value,
        )
        return "finish_exception"

    @staticmethod
    @timed("예외 분류 종료")
    def _finish_exception(state: MasterState) -> MasterState:
        """모든 마스터 예외 유형을 이력 저장 없이 종료한다."""

        logger.info(
            "======== 예외 분류 종료 | 유형=%s | Redis저장=안함",
            state["classification"]["classification_type"],
        )
        return {
            "status": "EXCEPTION",
            "approved": False,
            "interrupt": None,
        }

    @timed("프론트 선택 코드 비교")
    def _verify_selection(self, state: MasterState) -> MasterState:
        """프론트 코드가 있으면 비교하고, 미선택이면 분류 결과를 바로 통과시킨다."""

        classification = IntentClassification.model_validate(
            state["classification"]
        )
        frontend_code = state.get("frontend_agent_code")
        if frontend_code is None:
            logger.info(
                "======== 에이전트 코드 비교 생략 | 프론트 미선택 | "
                "LLM분류=%s | 결과=PASS",
                classification.agent_code,
            )
            update: MasterState = {
                "status": "PASS",
                "approved": True,
                "interrupt": None,
            }
            self._trace_recorder.record("에이전트비교생략", {**state, **update})
            return update

        frontend_code = frontend_code.upper()
        logger.info(
            "======== 에이전트 코드 비교 | 프론트=%s | LLM분류=%s",
            frontend_code,
            classification.agent_code,
        )

        if classification.agent_code == frontend_code:
            logger.info(
                "======== 에이전트 코드 일치 | 결과=PASS | 에이전트=%s",
                frontend_code,
            )
            update = {
                "status": "PASS",
                "approved": True,
                "interrupt": None,
            }
            self._trace_recorder.record("에이전트코드일치", {**state, **update})
            return update

        interrupt = build_hitl_request(
            hitl_type="AGENT_CODE_MISMATCH",
            message=(
                "선택한 에이전트와 질문 의도가 다릅니다. "
                "분류된 에이전트로 변경하시겠습니까?"
            ),
            fields=[
                {
                    "name": "signal",
                    "label": "변경 승인",
                    "type": "hidden",
                    "required": True,
                    "expected_value": "OK",
                }
            ],
            context={
                "frontend_agent_code": frontend_code,
                "classified_agent_code": classification.agent_code,
            },
        )
        logger.info(
            "======== 에이전트 코드 불일치 | Redis HITL 저장 단계 | "
            "프론트=%s | 변경대상=%s",
            frontend_code,
            classification.agent_code,
        )
        update = {
            "status": "INPUT_REQUIRED",
            "approved": False,
            "interrupt": interrupt,
        }
        self._trace_recorder.record("HITL입력요청", {**state, **update})
        return update

    @staticmethod
    @timed("Redis HITL 사용자 입력 검증")
    def _validate_agent_code_mismatch(state: MasterState) -> MasterState:
        """에이전트 코드 불일치 팝업에서 받은 OK 입력을 검증한다."""

        interrupt = state["interrupt"]
        if not isinstance(interrupt, dict):
            raise ValueError("복원된 HITL 입력 요청이 없습니다.")

        hitl_type = state.get("hitl_type")
        if hitl_type != "AGENT_CODE_MISMATCH":
            raise ValueError(
                f"에이전트 코드 검증 노드에 잘못 진입했습니다: {hitl_type}"
            )

        errors = validate_ok_signal(state.get("human_input"))
        if errors:
            logger.info(
                "======== HITL 입력 검증 실패 | 유형=%s | 오류=%s",
                hitl_type,
                errors,
            )
            refreshed = build_hitl_request(
                hitl_type=hitl_type,
                message=str(interrupt["message"]),
                fields=list(interrupt["fields"]),
                context=dict(interrupt["context"]),
                errors=errors,
            )
            return {
                "status": "INPUT_REQUIRED",
                "approved": False,
                "interrupt": refreshed,
            }

        logger.info("======== HITL 입력 검증 완료 | 유형=%s", hitl_type)
        return {
            "status": "PASS",
            "approved": True,
            "interrupt": None,
        }

    @staticmethod
    @timed("MCP 필수 파라미터 사용자 입력 검증")
    def _validate_mcp_parameter_input(state: MasterState) -> MasterState:
        """action으로 받은 값을 대기 중인 MCP match 파라미터에 반영한다."""

        interrupt = state.get("interrupt")
        if not isinstance(interrupt, dict):
            raise ValueError("복원된 MCP 파라미터 입력 요청이 없습니다.")
        if state.get("hitl_type") != "MCP_PARAMETER_REQUIRED":
            raise ValueError(
                "MCP 파라미터 검증 노드에 잘못 진입했습니다: "
                f"{state.get('hitl_type')}"
            )

        context = dict(interrupt.get("context", {}))
        input_code = str(context.get("input_code", ""))
        parameter_name = str(context.get("parameter_name", ""))
        match_index = int(context.get("match_index", -1))
        human_input = state.get("human_input")
        value = (
            human_input.get(input_code)
            if isinstance(human_input, dict)
            else None
        )
        normalized_value = str(value).strip() if value is not None else ""

        if not input_code or not parameter_name or not normalized_value:
            errors = {
                input_code or "input": "필수 입력값을 입력해 주세요."
            }
            logger.info(
                "======== MCP 파라미터 입력 검증 실패 | 입력코드=%s | "
                "파라미터=%s | 오류=%s",
                input_code,
                parameter_name,
                errors,
            )
            refreshed = build_hitl_request(
                hitl_type="MCP_PARAMETER_REQUIRED",
                message=str(interrupt.get("message", "필수값을 입력해 주세요.")),
                fields=list(interrupt.get("fields", [])),
                context=context,
                errors=errors,
            )
            return {
                "status": "INPUT_REQUIRED",
                "approved": False,
                "interrupt": refreshed,
            }

        subagent = SubagentResult.model_validate(state["subagent"])
        if match_index < 0 or match_index >= len(subagent.matches):
            raise ValueError(
                f"MCP 재개 match_index가 올바르지 않습니다: {match_index}"
            )

        match = subagent.matches[match_index]
        updated_parameters = {
            **match.parameters,
            parameter_name: normalized_value,
        }
        match.parameters = updated_parameters
        if match_index == 0:
            subagent.parameters = dict(updated_parameters)

        logger.info(
            "======== MCP 파라미터 입력 검증 완료 | 입력코드=%s | "
            "파라미터=%s | 값=%s | 재개순번=%d",
            input_code,
            parameter_name,
            normalized_value,
            match_index,
        )
        return {
            "status": "PASS",
            "approved": True,
            "interrupt": None,
            "subagent": subagent.model_dump(mode="json"),
            "mcp_start_index": match_index,
        }

    @staticmethod
    @timed("MCP 파라미터 입력 결과 라우팅")
    def _after_mcp_parameter_input(
        state: MasterState,
    ) -> Literal["save_hitl_state", "call_mcp"]:
        """입력이 유효하면 LLM 재호출 없이 대기 중 MCP부터 재개한다."""

        if state["status"] == "INPUT_REQUIRED":
            return "save_hitl_state"
        return "call_mcp"

    @staticmethod
    @timed("HITL 입력 결과 라우팅")
    def _after_input_decision(
        state: MasterState,
    ) -> Literal["save_hitl_state", "persist_user_message"]:
        """입력이 필요하면 Redis 저장, 통과하면 다음 업무 단계로 보낸다."""

        if state["status"] == "INPUT_REQUIRED":
            return "save_hitl_state"
        return "persist_user_message"

    @timed(
        "일반 Redis HITL 상태 저장",
        expected_exceptions=(HitlStateStoreUnavailableError,),
    )
    async def _save_hitl_state(self, state: MasterState) -> MasterState:
        """다음 HTTP 요청에 필요한 그래프 상태만 일반 Redis에 저장한다."""

        interrupt = state["interrupt"]
        if not isinstance(interrupt, dict):
            raise ValueError("Redis에 저장할 HITL 입력 요청이 없습니다.")

        # Redis에는 재개에 꼭 필요한 값만 명시적으로 선택해 저장한다. 상태 전체를
        # 그대로 저장하면 LangGraph 노드가 늘어날 때 임시 값이나 불필요한 사용자
        # 정보까지 자동으로 영속화될 수 있으므로 허용 목록 방식을 사용한다.
        stored_state = {
            "thread_id": state["thread_id"],
            "message_id": state["message_id"],
            "employee_id": state["employee_id"],
            "session_id": state["session_id"],
            "frontend_agent_code": state.get("frontend_agent_code"),
            "classification": state["classification"],
        }
        if interrupt["type"] == "MCP_PARAMETER_REQUIRED":
            # MCP 파라미터 재입력은 마스터/서브 LLM을 다시 호출하지 않는다.
            # 선택된 시나리오, 이미 완료된 MCP 결과와 재개 순번만 추가 저장한다.
            stored_state.update(
                {
                    "subagent": state["subagent"],
                    "mcp_results": state.get("mcp_results", []),
                    "mcp": state.get("mcp"),
                    "mcp_start_index": state.get("mcp_start_index", 0),
                    "status": "INPUT_REQUIRED",
                    "approved": False,
                }
            )
        await self._hitl_store.save(
            thread_id=state["thread_id"],
            hitl_type=str(interrupt["type"]),
            graph_state=stored_state,
            interrupt=interrupt,
        )
        logger.info(
            "======== HITL 대기 상태 저장 완료 | thread_id=%s | 유형=%s",
            state["thread_id"],
            interrupt["type"],
        )
        # LangGraph 0.2는 노드가 선언된 상태 키를 하나도 쓰지 않으면
        # InvalidUpdateError를 발생시킨다. 저장소 반영이 주목적인 노드라도
        # 현재 HITL 상태를 명시적으로 반환해 유효한 상태 갱신으로 처리한다.
        return {"status": "INPUT_REQUIRED", "approved": False}

    @timed("사용자 질문 Redis 저장")
    async def _persist_user_message(
        self,
        state: MasterState,
    ) -> MasterState:
        """보정된 질문을 최종 분류된 에이전트의 대화 이력에 저장한다."""

        classification = IntentClassification.model_validate(
            state["classification"]
        )
        assert classification.agent_code is not None

        logger.info(
            "======== Redis 대화 저장 요청 | 사원번호=%s | "
            "session_id=%s | 에이전트=%s | 역할=user | 보정질문=%s",
            state["employee_id"],
            state["session_id"],
            classification.agent_code,
            classification.refined_query,
        )
        await self._history_store.append_message(
            employee_id=state["employee_id"],
            session_id=state["session_id"],
            agent_code=classification.agent_code,
            role="user",
            content=classification.refined_query,
            message_id=f"{state['message_id']}:user",
        )
        logger.info("======== Redis 대화 저장 단계 완료")
        # Redis 저장은 부수효과지만 LangGraph 상태 노드는 최소 한 개의 상태
        # 키를 반환해야 한다. 기존 message_id를 유지하는 쓰기를 명시한다.
        return {"message_id": state["message_id"]}

    @timed("대화 저장 이후 라우팅")
    def _after_message_persist(
        self,
        state: MasterState,
    ) -> Literal["run_subagent", "clear_hitl_state", "end"]:
        """등록된 에이전트는 서브에이전트로, 나머지는 종료로 보낸다."""

        classification = IntentClassification.model_validate(
            state["classification"]
        )
        if (
            classification.agent_code is not None
            and self._subagent_router.supports(classification.agent_code)
        ):
            logger.info(
                "======== 서브에이전트 라우팅 | 에이전트=%s | 실행=예",
                classification.agent_code,
            )
            return "run_subagent"

        if state.get("entry_stage") == "HITL_RESUME":
            return "clear_hitl_state"
        return "end"

    @timed("시나리오 서브에이전트 실행")
    async def _run_subagent(self, state: MasterState) -> MasterState:
        """마스터가 선택한 에이전트의 시나리오와 파라미터를 분류한다."""

        classification = IntentClassification.model_validate(
            state["classification"]
        )
        assert classification.agent_code is not None
        logger.info(
            "======== 서브에이전트 실행 시작 | 에이전트=%s | 질문=%s",
            classification.agent_code,
            classification.refined_query,
        )
        started_at = perf_counter()
        try:
            result = await self._subagent_router.classify(
                agent_code=classification.agent_code,
                query=classification.refined_query,
            )
        except Exception as exc:
            log_failure_diagnostic(
                stage="LangGraph 서브에이전트 시나리오 분류 노드",
                code_location="app/graph.py:MasterIntentGraph._run_subagent",
                exc=exc,
                likely_cause=(
                    "선택된 서브에이전트의 LLM 호출, 시나리오 매칭 또는 "
                    "파라미터 구조화 처리 실패"
                ),
                corrective_action=(
                    "바로 앞의 서브에이전트 LLM 실패 진단, 해당 "
                    "prompts/subagents 폴더와 app/subagents/router.py를 확인하세요."
                ),
                retry_count=0,
                context={
                    "graph_node": "run_subagent",
                    "agent_code": classification.agent_code,
                    "refined_query": classification.refined_query,
                },
            )
            self._trace_recorder.record(
                "서브에이전트의도분류오류",
                state,
                elapsed_seconds=perf_counter() - started_at,
                error=exc,
            )
            # 오류를 정상 마스터 결과로 숨기지 않고 SSE error까지 즉시 전달한다.
            raise
        if result is None:
            logger.info(
                "======== 서브에이전트 실행 결과 없음 | 에이전트=%s",
                classification.agent_code,
            )
            return {"subagent": None}
        update = {"subagent": result.model_dump(mode="json")}
        logger.info(
            "======== 서브에이전트 실행 완료 | 에이전트=%s | "
            "매칭개수=%d | 결과=%s",
            result.agent_code,
            len(result.matches),
            result.model_dump(mode="json"),
        )
        traced_state = {**state, **update}
        elapsed = perf_counter() - started_at
        self._trace_recorder.record(
            "서브에이전트의도분류완료",
            traced_state,
            elapsed_seconds=elapsed,
        )
        self._trace_recorder.record("시나리오의도분류완료", traced_state)
        return update

    @staticmethod
    @timed("서브에이전트 이후 라우팅")
    def _after_subagent(
        state: MasterState,
    ) -> Literal["call_mcp", "clear_hitl_state", "end"]:
        """조회 시나리오만 MCP로 보내고 고정 답변만 있으면 바로 종료한다."""

        subagent_data = state.get("subagent")
        if subagent_data is not None:
            subagent = SubagentResult.model_validate(subagent_data)
            fixed_details = [
                match.detail_scenario_code
                for match in subagent.matches
                if get_subagent_fixed_response(
                    subagent.agent_code,
                    match.detail_scenario_code,
                )
                is not None
            ]
            requires_mcp = len(fixed_details) < len(subagent.matches)
            logger.info(
                "======== 서브에이전트 이후 라우팅\n"
                "에이전트=%s\n전체매칭개수=%d\n고정답변세부시나리오=%s\n"
                "MCP필요=%s",
                subagent.agent_code,
                len(subagent.matches),
                fixed_details,
                requires_mcp,
            )
            if requires_mcp:
                return "call_mcp"
            return MasterIntentGraph._completion_route(state)
        return MasterIntentGraph._completion_route(state)

    @timed("MCP 도구 실행")
    async def _call_mcp(self, state: MasterState) -> MasterState:
        """세부 시나리오별 Python payload로 MCP 도구를 호출한다."""

        subagent_data = state.get("subagent")
        if subagent_data is None:
            return {"mcp": None}
        subagent = SubagentResult.model_validate(subagent_data)
        started_at = perf_counter()
        results = [
            McpExecutionResult.model_validate(item)
            for item in state.get("mcp_results", [])
        ]
        start_index = int(state.get("mcp_start_index", 0))
        if start_index < 0 or start_index > len(subagent.matches):
            raise ValueError(f"MCP 시작 순번이 올바르지 않습니다: {start_index}")
        logger.info(
            "======== MCP 실행 범위 | 전체매칭=%d | 시작순번=%d | "
            "기완료결과=%d",
            len(subagent.matches),
            start_index,
            len(results),
        )
        try:
            for match_index in range(start_index, len(subagent.matches)):
                match = subagent.matches[match_index]
                fixed_response = get_subagent_fixed_response(
                    subagent.agent_code,
                    match.detail_scenario_code,
                )
                if fixed_response is not None:
                    logger.info(
                        "======== 서브에이전트 고정 답변 선택\n"
                        "순번=%d\n에이전트=%s\n세부시나리오=%s\n"
                        "MCP호출=생략\n고정답변=%s",
                        match_index,
                        subagent.agent_code,
                        match.detail_scenario_code,
                        fixed_response.message,
                    )
                    continue
                # 각 매칭을 독립된 단일 결과로 변환해 코드에 등록된 MCP
                # payload를 차례대로 실행한다.
                single = SubagentResult(
                    agent_code=subagent.agent_code,
                    prompt_version=subagent.prompt_version,
                    scenario_code=match.scenario_code,
                    scenario_name=match.scenario_name,
                    detail_scenario_code=match.detail_scenario_code,
                    detail_scenario_name=match.detail_scenario_name,
                    parameters=match.parameters,
                    matches=[match],
                )
                logger.info(
                    "======== MCP 개별 실행 시작 | 순번=%d | 시나리오=%s | "
                    "세부시나리오=%s | 파라미터=%s",
                    match_index,
                    match.scenario_code,
                    match.detail_scenario_code,
                    match.parameters,
                )
                try:
                    result = await self._mcp_executor.execute(
                        subagent=single,
                        employee_id=state["employee_id"],
                        session_id=state["session_id"],
                        thread_id=state["thread_id"],
                        request_context=state.get("request_context", {}),
                    )
                except McpParameterInputRequired as required:
                    interrupt = build_hitl_request(
                        hitl_type="MCP_PARAMETER_REQUIRED",
                        message=required.message,
                        fields=[
                            {
                                "name": required.input_code,
                                "label": required.label,
                                "type": required.input_type,
                                "required": True,
                            }
                        ],
                        context={
                            "agent_code": subagent.agent_code,
                            "scenario_code": match.scenario_code,
                            "detail_scenario_code": (
                                match.detail_scenario_code
                            ),
                            "match_index": match_index,
                            "input_code": required.input_code,
                            "parameter_name": required.parameter_name,
                        },
                    )
                    serialized_results = [
                        item.model_dump(mode="json") for item in results
                    ]
                    update: MasterState = {
                        "status": "INPUT_REQUIRED",
                        "approved": False,
                        "interrupt": interrupt,
                        "subagent": subagent.model_dump(mode="json"),
                        "mcp_start_index": match_index,
                        "mcp": (
                            serialized_results[0]
                            if serialized_results
                            else None
                        ),
                        "mcp_results": serialized_results,
                    }
                    logger.info(
                        "======== MCP 필수 파라미터 누락 | 순번=%d | "
                        "세부시나리오=%s | 입력코드=%s | 파라미터=%s | "
                        "message=%s",
                        match_index,
                        match.detail_scenario_code,
                        required.input_code,
                        required.parameter_name,
                        required.message,
                    )
                    self._trace_recorder.record(
                        "MCP파라미터입력요청",
                        {**state, **update},
                    )
                    return update
                if result is not None:
                    # 원본 structuredContent는 result에 유지하고 조회형 시나리오는
                    # 별도의 query.v1 정제 결과와 고정 답변을 생성한다. RAG는
                    # raw.rag로 표시하고 원본 문서 구조를 그대로 다음 단계에 넘긴다.
                    try:
                        result = adapt_mcp_result(
                            execution=result,
                            subagent=single,
                            employee_id=state["employee_id"],
                            session_id=state["session_id"],
                            thread_id=state["thread_id"],
                            request_context=state.get("request_context", {}),
                        )
                    except Exception as exc:
                        # MCP 호출은 성공했더라도 결과 컬럼·data 형식 또는 포맷
                        # 함수에서 오류가 날 수 있다. 원인은 로그에 상세히 남기고
                        # 해당 시나리오는 안전한 고정 답변으로 계속 진행한다.
                        log_failure_diagnostic(
                            stage="MCP 조회 결과 정제",
                            code_location=(
                                "app/graph.py:MasterIntentGraph._call_mcp -> "
                                "app/mcp/result_adapters.py:adapt_mcp_result"
                            ),
                            exc=exc,
                            likely_cause=(
                                "MCP structuredContent.data 구조, objId 컬럼 또는 "
                                "시나리오 답변 포맷 함수 불일치"
                            ),
                            corrective_action=(
                                "app/mcp/result_adapters.py의 해당 "
                                "SCENARIO_QUERY_CONFIGS와 formatter를 확인하세요."
                            ),
                            retry_count=0,
                            context={
                                "agent_code": single.agent_code,
                                "detail_scenario_code": (
                                    single.detail_scenario_code
                                ),
                                "tool_name": result.tool_name,
                                "mcp_request_id": result.request_id,
                            },
                        )
                        result = result.model_copy(
                            update={
                                "succeeded": False,
                                "outcome": "ERROR",
                                "user_message": MCP_SAFE_ERROR_MESSAGE,
                                "formatted_result": None,
                                "error": str(exc),
                            }
                        )
                    results.append(result)
                    logger.info(
                        "======== MCP 개별 실행 완료 | 도구=%s | 성공=%s | "
                        "추적ID=%s | 결과형식=%s | 정제결과=%s | 오류=%s | "
                        "원본결과로그=생략",
                        result.tool_name,
                        result.succeeded,
                        result.request_id,
                        result.result_format,
                        result.formatted_result,
                        result.error,
                    )
        except Exception as exc:
            log_failure_diagnostic(
                stage="LangGraph MCP 다중 도구 실행 노드",
                code_location="app/graph.py:MasterIntentGraph._call_mcp",
                exc=exc,
                likely_cause=(
                    "MCP payload 생성, GenOS HTTP 호출, 원본 응답 변환 또는 "
                    "조회형 결과 어댑터 처리 실패"
                ),
                corrective_action=(
                    "바로 앞의 MCP 실패 진단과 app/mcp/payloads.py의 해당 "
                    "세부 시나리오, app/mcp/client.py와 "
                    "app/mcp/result_adapters.py를 확인하세요."
                ),
                retry_count=0,
                context={
                    "graph_node": "call_mcp",
                    "agent_code": subagent.agent_code,
                    "detail_scenario_codes": [
                        match.detail_scenario_code
                        for match in subagent.matches
                    ],
                    "start_index": start_index,
                    "completed_result_count": len(results),
                },
            )
            self._trace_recorder.record(
                "MCP도구호출오류",
                state,
                elapsed_seconds=perf_counter() - started_at,
                error=exc,
            )
            raise
        if not results:
            logger.info(
                "======== MCP 실행 결과 없음 | 에이전트=%s | 매칭개수=%d",
                subagent.agent_code,
                len(subagent.matches),
            )
            return {"mcp": None, "mcp_results": []}
        logger.info(
            "======== MCP 다중 실행 완료 | 실행개수=%d | 도구=%s | 성공=%s",
            len(results),
            [result.tool_name for result in results],
            [result.succeeded for result in results],
        )
        serialized_results = [
            result.model_dump(mode="json") for result in results
        ]
        update = {
            "mcp": serialized_results[0],
            "mcp_results": serialized_results,
            "mcp_start_index": len(subagent.matches),
            "status": "PASS",
            "approved": True,
            "interrupt": None,
        }
        # 코드 기반 executor가 실제 tool_name을 결정하므로 실행 결과가 만들어진
        # 직후 선택 행을 남겨 정확한 도구명이 CSV에 포함되게 한다.
        self._trace_recorder.record("MCP도구선택완료", {**state, **update})
        self._trace_recorder.record(
            "MCP도구호출완료",
            {**state, **update},
            elapsed_seconds=perf_counter() - started_at,
        )
        return update

    @staticmethod
    @timed("MCP 이후 라우팅")
    def _after_mcp_call(
        state: MasterState,
    ) -> Literal["save_hitl_state", "clear_hitl_state", "end"]:
        """추가 입력이면 저장하고, MCP 완료면 실행을 끝낸다."""

        if state.get("status") == "INPUT_REQUIRED":
            return "save_hitl_state"
        return MasterIntentGraph._completion_route(state)

    @staticmethod
    def _completion_route(
        state: MasterState,
    ) -> Literal["clear_hitl_state", "end"]:
        """HITL 재진입이면 저장 상태를 정리하고 신규 요청이면 종료한다."""

        if state.get("entry_stage") == "HITL_RESUME":
            return "clear_hitl_state"
        return "end"

    @timed(
        "일반 Redis HITL 상태 정리",
        expected_exceptions=(HitlStateStoreUnavailableError,),
    )
    async def _clear_hitl_state(self, state: MasterState) -> MasterState:
        """승인 완료 후 더 이상 필요 없는 HITL 상태를 삭제한다."""

        await self._hitl_store.delete(state["thread_id"])
        # 삭제 자체는 부수효과이므로 현재 thread_id를 유지하는 상태 쓰기를
        # 반환해 LangGraph의 빈 업데이트 오류를 방지한다.
        return {"thread_id": state["thread_id"]}

    @timed(
        "LangGraph 신규 실행",
        expected_exceptions=(HitlStateStoreUnavailableError,),
    )
    async def start(
        self,
        *,
        thread_id: str,
        employee_id: str,
        session_id: str,
        message: str,
        frontend_agent_code: str | None,
        request_context: dict[str, Any] | None = None,
    ) -> MasterResult:
        """새 질문에 대한 stateless 그래프 실행을 시작한다."""

        logger.info(
            "======== LangGraph 신규 시작 | thread_id=%s | "
            "session_id=%s",
            thread_id,
            session_id,
        )
        input_state: MasterState = {
                "entry_stage": "NEW_CHAT",
                "thread_id": thread_id,
                "message": message,
                "message_id": thread_id,
                "employee_id": employee_id,
                "session_id": session_id,
                "frontend_agent_code": (
                    frontend_agent_code.upper()
                    if frontend_agent_code is not None
                    else None
                ),
                "request_context": dict(request_context or {}),
            }
        self._trace_recorder.record("요청도착", input_state)
        state = await self._graph.ainvoke(input_state)
        self._trace_recorder.record("요청처리완료", state)
        return self._to_result(thread_id, state)

    @timed(
        "LangGraph Redis HITL 재진입",
        expected_exceptions=(
            HitlStateNotFoundError,
            HitlStateStoreUnavailableError,
        ),
    )
    async def resume(
        self,
        *,
        thread_id: str,
        value: Any,
        expected_employee_id: str | None = None,
        expected_session_id: str | None = None,
        request_context: dict[str, Any] | None = None,
    ) -> MasterResult:
        """일반 Redis 상태를 복원해 HITL 검증 Edge부터 새로 실행한다.

        스트리밍 API는 요청의 사원번호와 session_id를 함께 전달한다. Redis에
        저장된 원래 요청 범위와 다르면 상태 존재 여부를 노출하지 않고 찾을 수
        없는 상태와 동일하게 처리한다. 기존 JSON API는 호환을 위해 두 검증값을
        생략할 수 있다.
        """

        logger.info(
            "======== Redis HITL 재개 시작 | thread_id=%s | 입력=%s",
            thread_id,
            value,
        )
        entry = await self._hitl_store.get(thread_id)
        if entry is None:
            raise HitlStateNotFoundError(thread_id)

        stored_employee_id = entry.graph_state.get("employee_id")
        stored_session_id = entry.graph_state.get("session_id")
        if (
            expected_employee_id is not None
            and stored_employee_id != expected_employee_id
        ) or (
            expected_session_id is not None
            and stored_session_id != expected_session_id
        ):
            logger.info(
                "======== Redis HITL 범위 불일치 | thread_id=%s | "
                "요청사원=%s | 저장사원=%s | 요청session=%s | 저장session=%s",
                thread_id,
                expected_employee_id,
                stored_employee_id,
                expected_session_id,
                stored_session_id,
            )
            raise HitlStateNotFoundError(thread_id)

        restored_state: MasterState = dict(entry.graph_state)
        restored_state.update(
            {
                "entry_stage": "HITL_RESUME",
                "hitl_type": entry.hitl_type,
                "thread_id": thread_id,
                "human_input": value,
                "interrupt": entry.interrupt,
                # 최초 요청의 토큰을 Redis에서 복원하지 않는다. 현재 HITL 요청에
                # 포함된 새 토큰과 사용자 정보만 MCP 단계까지 전달한다.
                "request_context": dict(request_context or {}),
            }
        )
        self._trace_recorder.record("HITL재진입", restored_state)
        logger.info(
            "======== Redis HITL 그래프 상태 복원 | thread_id=%s | "
            "유형=%s | 진입단계=%s | 복원상태=%s | 요청컨텍스트키=%s",
            thread_id,
            entry.hitl_type,
            restored_state["entry_stage"],
            {
                key: value
                for key, value in restored_state.items()
                if key != "request_context"
            },
            sorted(restored_state.get("request_context", {})),
        )
        state = await self._graph.ainvoke(restored_state)
        self._trace_recorder.record("요청처리완료", state)
        return self._to_result(thread_id, state)

    @staticmethod
    def _to_result(thread_id: str, state: MasterState) -> MasterResult:
        """일반 Redis HITL 요청을 포함한 그래프 상태를 API 결과로 변환한다."""

        classification = IntentClassification.model_validate(
            state["classification"]
        )
        status = state["status"]
        interrupt = state.get("interrupt")
        subagent_data = state.get("subagent")
        mcp_data = state.get("mcp")
        mcp_results_data = state.get("mcp_results", [])
        logger.info(
            "======== 그래프 결과 변환 | 상태=%s | thread_id=%s",
            status,
            thread_id,
        )
        return MasterResult(
            status=status,
            thread_id=thread_id,
            classification=classification,
            interrupt=interrupt if status == "INPUT_REQUIRED" else None,
            subagent=(
                SubagentResult.model_validate(subagent_data)
                if subagent_data is not None
                else None
            ),
            mcp=(
                McpExecutionResult.model_validate(mcp_data)
                if mcp_data is not None
                else None
            ),
            mcp_results=[
                McpExecutionResult.model_validate(item)
                for item in mcp_results_data
            ],
        )
