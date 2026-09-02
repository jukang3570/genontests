"""FastAPI 요청과 응답에 사용하는 Pydantic 모델."""

import ast
import json
from collections.abc import Collection, Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


def _decode_input_envelope(value: Any) -> Any:
    """외부 ``input`` envelope의 dict/bytes/문자열을 JSON 값으로 변환한다."""

    if isinstance(value, Mapping):
        return dict(value)

    raw_text: str
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            raw_text = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("input bytes는 UTF-8이어야 합니다.") from exc
    elif isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith(("b'", 'b"', "B'", 'B"')):
            try:
                literal_value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    "input의 bytes 문자열 표현이 올바르지 않습니다."
                ) from exc
            if not isinstance(literal_value, bytes):
                raise ValueError("input의 bytes 문자열 표현이 올바르지 않습니다.")
            try:
                raw_text = literal_value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("input bytes는 UTF-8이어야 합니다.") from exc
        else:
            raw_text = candidate
    else:
        raise ValueError(
            "input은 JSON object, UTF-8 JSON bytes 또는 JSON 문자열이어야 합니다."
        )

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("input 내부 값은 유효한 JSON이어야 합니다.") from exc


def normalize_json_request_body(
    value: Any,
    *,
    direct_fields: Collection[str],
) -> dict[str, Any]:
    """일반 객체와 루트/input 래퍼의 직렬화된 JSON을 공통 객체로 만든다."""

    if isinstance(value, Mapping):
        if set(direct_fields).intersection(value):
            return dict(value)
        if "input" not in value:
            return dict(value)
        wrapped_value = value["input"]
    else:
        wrapped_value = value

    decoded = _decode_input_envelope(wrapped_value)
    if not isinstance(decoded, Mapping):
        raise ValueError("요청 본문은 JSON object여야 합니다.")
    return dict(decoded)


class HumanInputItem(BaseModel):
    """프론트 입력창 한 개에서 전달되는 코드와 사용자 입력값."""

    # 개발 단계에서는 프론트가 새 필드를 먼저 보내도 요청 전체를 거절하지 않는다.
    # 사용하지 않는 필드는 버려 Redis/MCP 문맥으로 의도치 않게 전파하지 않는다.
    model_config = ConfigDict(extra="ignore")

    code: str = ""
    input: JsonValue = None

    @field_validator("code", mode="before")
    @classmethod
    def normalize_code(cls, value: Any) -> str:
        """누락·null code도 요청 검증 오류 대신 빈 값으로 후속 검증에 넘긴다."""

        return str(value or "").strip()


class StreamingUser(BaseModel):
    """WAS가 인증 사용자 정보를 채워 전달하는 사용자 객체.

    연계 규격상 객체와 각 키는 항상 전달하지만 값은 null일 수 있다. 실제 사번이
    없을 때는 API 계층이 session_id 기반의 익명 내부 식별자를 생성하므로 서로
    다른 세션의 Redis 이력이 한 사용자로 섞이지 않는다.
    """

    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    deptcode: str | None = None
    deptname: str | None = None

    @field_validator("id", "deptcode", "deptname", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any) -> str | None:
        """사용자 선택 정보의 null·빈 문자열·숫자 입력을 유연하게 정규화한다."""

        if value is None:
            return None
        return str(value).strip() or None


class StreamingChatRequest(BaseModel):
    """프론트 → WAS → 에이전트 최종 규격을 사용하는 SSE 채팅 요청.

    개발 단계에서는 ``message``만 필수다. session/thread/endpoint 등 선택 필드는
    누락·null·빈 문자열을 같은 미지정 상태로 정규화하며 API가 UUID 또는 기본
    project code를 채운다. HITL 여부는 thread_id가 아니라 ``humanInput``에 실제
    입력 항목이 있는지로 판단한다.
    """

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    message: str = Field(..., min_length=1, max_length=10_000)
    session_id: str | None = None
    thread_id: str | None = None
    endpoint: str | None = None
    agent_code: str | None = None
    recommendation_id: str | None = None
    human_input: list[HumanInputItem] | None = Field(
        default=None,
        alias="humanInput",
    )
    user: StreamingUser | None = None

    @model_validator(mode="before")
    @classmethod
    def unwrap_input_envelope(cls, value: Any) -> Any:
        """문자열/bytes 본문 또는 외부 ``input`` envelope에서 요청을 복원한다.

        일반 HTTP JSON 객체는 그대로 검증한다. 본문 루트 자체 또는 정상 요청
        필드 없이 존재하는 바깥쪽 ``input`` 값이 UTF-8 JSON bytes, JSON 문자열,
        Python bytes 문자열 표현(``b'{...}'``)이면 내부 JSON 객체로 변환한다.
        """

        return normalize_json_request_body(
            value,
            direct_fields={
                "message",
                "session_id",
                "thread_id",
                "endpoint",
                "agent_code",
                "recommendation_id",
                "humanInput",
                "human_input",
                "user",
            },
        )

    @field_validator(
        "session_id",
        "thread_id",
        "endpoint",
        mode="before",
    )
    @classmethod
    def normalize_optional_identifiers(cls, value: Any) -> str | None:
        """null·빈 문자열·공백은 서버 기본값/UUID 생성 대상으로 통일한다."""

        if value is None:
            return None
        return str(value).strip() or None

    @field_validator("agent_code", mode="before")
    @classmethod
    def normalize_agent_code(cls, value: Any) -> str | None:
        """미선택은 None으로, 선택된 코드는 대문자로 통일한다."""

        if value is None:
            return None
        return str(value).strip().upper() or None

    @field_validator("recommendation_id", mode="before")
    @classmethod
    def normalize_recommendation_id(cls, value: Any) -> str | None:
        """버튼 선택 ID의 양끝 공백을 제거하고 빈 문자열은 미선택으로 본다."""

        if value is None:
            return None
        return str(value).strip() or None

    @property
    def is_hitl_continuation(self) -> bool:
        """사용자 입력 목록이 있으면 Redis HITL 재진입 요청으로 판단한다."""

        return bool(self.human_input)

    def to_hitl_value(self) -> dict[str, JsonValue]:
        """humanInput 배열을 기존 HITL 검증기가 사용하는 코드-값 객체로 바꾼다."""

        return {item.code: item.input for item in (self.human_input or [])}
