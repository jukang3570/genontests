"""프론트 SSE 계약의 이벤트 직렬화와 공통 변환 도구."""

import json
from typing import Any, Iterator


def encode_sse(event: str, data: Any) -> str:
    """한 이벤트를 한 줄 JSON과 SSE 종료 개행으로 직렬화한다."""

    payload = json.dumps(
        {"event": event, "data": data},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return f"data: {payload}\n\n"


def split_text(text: str, chunk_size: int = 16) -> Iterator[str]:
    """고정 답변도 실제 token 이벤트 흐름을 시험할 수 있도록 나눈다."""

    if chunk_size < 1:
        raise ValueError("chunk_size는 1 이상이어야 합니다.")
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


def build_action_event(
    thread_id: str,
    interrupt: dict[str, Any] | None,
) -> dict[str, Any]:
    """내부 interrupt를 간결한 프론트 action/humanInput 계약으로 변환한다.

    분류 결과와 내부 Redis 상태는 로그에 남기고 외부에는 action 처리에 필요한
    코드, 안내 문구, 입력 정의와 검증 오류만 노출한다.
    """

    source = interrupt or {}
    inputs = []
    for field in source.get("fields", []):
        if not isinstance(field, dict):
            continue
        item = {
            "code": field.get("name"),
            "label": field.get("label"),
            "type": field.get("type", "text"),
            "required": bool(field.get("required", False)),
        }
        optional_fields = {
            "expectedValue": field.get("expected_value"),
            "pattern": field.get("pattern"),
            "minLength": field.get("min_length"),
            "maxLength": field.get("max_length"),
            "allowedValues": field.get("allowed_values"),
        }
        item.update(
            {
                key: value
                for key, value in optional_fields.items()
                if value is not None
            }
        )
        if field.get("sensitive"):
            item["sensitive"] = True
        inputs.append(item)
    event = {
        "code": source.get("action_code") or source.get("type"),
        "thread_id": thread_id,
        "message": source.get("message"),
        "inputs": inputs,
    }
    errors = source.get("errors")
    if isinstance(errors, dict) and errors:
        event["errors"] = errors
    # 새 프론트 필드가 실제로 필요해질 때만 이 whitelist에 명시적으로 추가한다.
    # interrupt.context, MCP 원본, handler 내부 상태는 운영 action에 노출하지 않는다.
    return event
