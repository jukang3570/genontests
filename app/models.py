"""FastAPI 요청과 응답에 사용하는 Pydantic 모델."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from app.domain import IntentClassification
from app.mcp.models import McpExecutionResult
from app.subagents.models import SubagentResult


class ChatRequest(BaseModel):
    """신규 질문과 HITL 입력을 하나의 채팅 API로 받는 요청.

    ``thread_id``가 없으면 신규 질문이며 message와 employee_id가 필요하다.
    frontend_agent_code는 사용자가 프론트에서 에이전트를 직접 선택했을 때만
    전달한다. ``thread_id``가 있으면 Redis에 대기 중인 HITL 요청에 대한
    응답이며 hitl_input만 필요하다. 두 형태를 섞어 보내면 잘못된 상태가
    만들어질 수 있으므로 모델 검증 단계에서 거절한다.
    """

    model_config = ConfigDict(extra="forbid")

    thread_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="HITL 상태를 이어 갈 때 이전 응답에서 받은 thread_id",
    )
    message: str | None = Field(
        default=None,
        min_length=1,
        max_length=10_000,
    )
    employee_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="사원번호 또는 사원을 고유하게 식별하는 값",
    )
    frontend_agent_code: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "프론트에서 선택한 에이전트 코드. 미선택이면 생략, null 또는 빈 문자열"
        ),
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    hitl_input: JsonValue | None = Field(
        default=None,
        description=(
            "HITL 승인 또는 향후 MCP 파라미터 입력에 사용하는 JSON 값"
        ),
    )

    @field_validator("frontend_agent_code", mode="before")
    @classmethod
    def normalize_optional_frontend_agent_code(cls, value):
        """미선택을 나타내는 null·빈 문자열·공백 문자열을 모두 None으로 통일한다."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_request_mode(self) -> "ChatRequest":
        """thread_id 존재 여부에 따라 신규·재진입 요청 필드를 검증한다."""

        if self.thread_id is not None:
            if self.hitl_input is None:
                raise ValueError(
                    "thread_id가 있는 요청에는 hitl_input이 필요합니다."
                )
            supplied_new_fields = {
                "message": self.message,
                "employee_id": self.employee_id,
                "frontend_agent_code": self.frontend_agent_code,
                "session_id": self.session_id,
            }
            mixed_fields = [
                name
                for name, value in supplied_new_fields.items()
                if value is not None
            ]
            if mixed_fields:
                raise ValueError(
                    "HITL 재진입 요청에는 신규 질문 필드를 함께 보낼 수 없습니다: "
                    + ", ".join(mixed_fields)
                )
            return self

        missing = [
            name
            for name, value in {
                "message": self.message,
                "employee_id": self.employee_id,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "신규 채팅 요청에 필수 필드가 없습니다: "
                + ", ".join(missing)
            )
        if self.hitl_input is not None:
            raise ValueError(
                "신규 채팅 요청에는 hitl_input을 사용할 수 없습니다."
            )
        return self

    @property
    def is_hitl_continuation(self) -> bool:
        """Redis HITL 상태를 이어 가는 요청인지 반환한다."""

        return self.thread_id is not None


class HumanInputItem(BaseModel):
    """프론트 입력창 한 개에서 전달되는 코드와 사용자 입력값."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=200)
    input: JsonValue

    @field_validator("input")
    @classmethod
    def reject_null_input(cls, value: JsonValue) -> JsonValue:
        """최종 연계 규격에 따라 입력 항목의 null 값은 거절한다."""

        if value is None:
            raise ValueError("humanInput[].input은 null일 수 없습니다.")
        return value


class StreamingUser(BaseModel):
    """WAS가 인증 사용자 정보를 채워 전달하는 사용자 객체.

    연계 규격상 객체와 각 키는 항상 전달하지만 값은 null일 수 있다. 실제 사번이
    없을 때는 API 계층이 session_id 기반의 익명 내부 식별자를 생성하므로 서로
    다른 세션의 Redis 이력이 한 사용자로 섞이지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    deptcode: str | None = Field(..., min_length=1, max_length=100)
    deptname: str | None = Field(..., min_length=1, max_length=200)


class StreamingChatRequest(BaseModel):
    """프론트 → WAS → 에이전트 최종 규격을 사용하는 SSE 채팅 요청.

    모든 최상위 키는 요청 JSON에 포함한다. ``message``는 null을 허용하지 않고,
    agent_code는 프론트 미선택을 표현하기 위해 null을 허용한다. thread_id는 일반
    질문에도 올 수 있으므로 HITL 여부는 thread_id가 아니라 ``humanInput``에
    실제 입력 항목이 있는지로 판단한다.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    message: str = Field(..., min_length=1, max_length=10_000)
    session_id: str | None = Field(..., min_length=1, max_length=200)
    thread_id: str | None = Field(..., min_length=1, max_length=200)
    endpoint: str | None = Field(..., min_length=1, max_length=100)
    agent_code: str | None = Field(..., min_length=1, max_length=100)
    human_input: list[HumanInputItem] | None = Field(
        ...,
        alias="humanInput",
    )
    user: StreamingUser | None = Field(...)

    @field_validator("agent_code", mode="before")
    @classmethod
    def normalize_agent_code(cls, value):
        """미선택은 None으로, 선택된 코드는 대문자로 통일한다."""

        if isinstance(value, str):
            normalized = value.strip().upper()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_stream_request_mode(self) -> "StreamingChatRequest":
        """HITL 입력의 식별자와 입력 코드 중복을 검증한다."""

        inputs = self.human_input or []
        if inputs and self.thread_id is None:
            raise ValueError(
                "humanInput이 있으면 이전 action에서 받은 thread_id가 필요합니다."
            )
        codes = [item.code for item in inputs]
        if len(codes) != len(set(codes)):
            raise ValueError("humanInput의 code는 요청 안에서 중복될 수 없습니다.")
        return self

    @property
    def is_hitl_continuation(self) -> bool:
        """사용자 입력 목록이 있으면 Redis HITL 재진입 요청으로 판단한다."""

        return bool(self.human_input)

    def to_hitl_value(self) -> dict[str, JsonValue]:
        """humanInput 배열을 기존 HITL 검증기가 사용하는 코드-값 객체로 바꾼다."""

        return {
            item.code: item.input
            for item in (self.human_input or [])
        }


class ChatResponse(BaseModel):
    """마스터 분류와 등록된 시나리오 서브에이전트의 결과."""

    status: Literal["PASS", "INPUT_REQUIRED", "EXCEPTION"]
    thread_id: str
    classification: IntentClassification
    # 마스터 agent_code에 구현된 서브에이전트가 없거나 HITL 대기 중이면 null이다.
    subagent: SubagentResult | None = None
    # 세부 시나리오 Python 코드에 등록된 MCP 도구와 추적 ID, 조회 결과이다.
    mcp: McpExecutionResult | None = None
    # 다중 시나리오 각각의 MCP 결과. 기존 mcp는 첫 번째 결과를 유지한다.
    mcp_results: list[McpExecutionResult] = Field(default_factory=list)
    # 기존 프론트 계약과의 호환성을 위해 필드명은 interrupt를 유지한다.
    # 실제 구현은 LangGraph interrupt()가 아니며, 에이전트 승인·MCP 파라미터
    # 입력·잘못된 값 재입력 등을 표현하는 Redis 기반 공통 입력 요청이다.
    interrupt: dict[str, JsonValue] | None = None
