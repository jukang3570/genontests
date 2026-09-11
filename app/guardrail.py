"""Bastion Guardian AI 가드레일 호출과 문장 단위 출력 정규화."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from app.config import Settings
from app.observability import logger, timed


GuardrailAction = Literal["PASS", "MASK", "BLOCK"]
GuardrailProcessType = Literal["INPUT", "OUTPUT"]
GuardrailRole = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class GuardrailContext:
    """고객사 API의 option/additionalData 추적 문맥."""

    trace_id: str
    session_id: str
    user_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    additional_data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardrailDecision:
    """가드레일 판정과 실제로 사용해야 하는 처리 결과."""

    action: GuardrailAction
    processed_content: str | None
    applied: bool
    reason: str | None = None

    @property
    def allowed(self) -> bool:
        return self.action in {"PASS", "MASK"} and self.processed_content is not None


class GuardrailClient(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def process_text(
        self,
        content: str,
        *,
        role: GuardrailRole,
        process_type: GuardrailProcessType,
        context: GuardrailContext,
    ) -> GuardrailDecision: ...

    async def aclose(self) -> None: ...


class BastionGuardianClient:
    """API 키 유무에 따라 실제 검사 또는 명시적 PASS-through를 수행한다."""

    def __init__(
        self,
        *,
        base_url: str,
        endpoint_path: str,
        api_key: str | None,
        timeout_seconds: float,
        fail_open: bool,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key.strip() if api_key else None
        self._url = f"{base_url.rstrip('/')}/{endpoint_path.strip('/')}"
        self._fail_open = fail_open
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def enabled(self) -> bool:
        return self._api_key is not None

    @timed("Bastion Guardian 텍스트 검사")
    async def process_text(
        self,
        content: str,
        *,
        role: GuardrailRole,
        process_type: GuardrailProcessType,
        context: GuardrailContext,
    ) -> GuardrailDecision:
        normalized_role = str(role).strip().casefold()
        normalized_process = str(process_type).strip().upper()
        if normalized_role not in {"user", "assistant", "system"}:
            raise ValueError("가드레일 role은 user, assistant, system이어야 합니다.")
        if normalized_process not in {"INPUT", "OUTPUT"}:
            raise ValueError("가드레일 processType은 INPUT 또는 OUTPUT이어야 합니다.")

        # 고객사 규칙상 system은 검사에서 제외한다. 키가 아직 없으면 개발 단계의
        # 명시적 PASS 처리로 원문을 그대로 후속 단계에 넘긴다.
        if normalized_role == "system" or not self.enabled:
            return GuardrailDecision(
                action="PASS",
                processed_content=content,
                applied=False,
                reason="SYSTEM_EXCLUDED" if normalized_role == "system" else "NO_API_KEY",
            )

        payload = {
            "messages": [
                {
                    "role": normalized_role,
                    "content": [{"type": "text", "text": content}],
                }
            ],
            "processType": normalized_process,
            "additionalData": dict(context.additional_data),
            "option": {
                "trace_id": context.trace_id,
                "session_id": context.session_id,
                "user_id": context.user_id,
                "metadata": dict(context.metadata),
                "tags": list(context.tags),
            },
        }
        try:
            response = await self._http_client.post(
                self._url,
                headers={
                    "X-Starfort-Guard-Api-Key": self._api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            decision = parse_guardrail_response(response.json(), original=content)
            logger.info(
                "======== AI 가드레일 완료 | processType=%s | role=%s | "
                "action=%s | 원문길이=%d | 처리길이=%d",
                normalized_process,
                normalized_role,
                decision.action,
                len(content),
                len(decision.processed_content or ""),
            )
            return decision
        except Exception as exc:
            logger.warning(
                "!!!!!!!! AI 가드레일 실패 | processType=%s | role=%s | "
                "정책=%s | 오류유형=%s",
                normalized_process,
                normalized_role,
                "fail-open" if self._fail_open else "fail-closed",
                type(exc).__name__,
            )
            if self._fail_open:
                return GuardrailDecision(
                    action="PASS",
                    processed_content=content,
                    applied=False,
                    reason="ERROR_FAIL_OPEN",
                )
            return GuardrailDecision(
                action="BLOCK",
                processed_content=None,
                applied=False,
                reason="ERROR_FAIL_CLOSED",
            )

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()


def parse_guardrail_response(payload: Any, *, original: str) -> GuardrailDecision:
    """PASS/MASK/BLOCK 및 공통 오류 봉투를 엄격하게 해석한다."""

    if not isinstance(payload, Mapping):
        raise ValueError("가드레일 응답이 JSON object가 아닙니다.")
    if payload.get("ok") is False or isinstance(payload.get("error"), Mapping):
        error = payload.get("error")
        code = error.get("code") if isinstance(error, Mapping) else "UNKNOWN"
        raise ValueError(f"가드레일 오류 응답: {code}")

    items = _result_items(payload)
    raw_action = payload.get("action")
    if raw_action is None and items:
        raw_action = items[0].get("action")
    action = str(raw_action or "").strip().upper()
    if action == "PASS":
        return GuardrailDecision("PASS", original, True)
    if action == "BLOCK":
        return GuardrailDecision("BLOCK", None, True)
    if action != "MASK":
        raise ValueError("가드레일 action은 PASS, MASK, BLOCK이어야 합니다.")

    for item in items:
        processed = item.get("processed_content")
        if isinstance(processed, str):
            return GuardrailDecision("MASK", processed, True)
    raise ValueError("MASK 응답에 processed_content가 없습니다.")


def split_period_sentences(buffer: str) -> tuple[list[str], str]:
    """온점(.)까지 완성된 문장과 아직 미완성인 꼬리를 분리한다.

    연속 온점은 한 경계로 묶으며 공백과 줄바꿈은 다음 문장에 남겨, 모든 조각을
    다시 합치면 원문과 동일하게 유지한다.
    """

    sentences: list[str] = []
    start = 0
    index = 0
    while index < len(buffer):
        if buffer[index] != ".":
            index += 1
            continue
        end = index + 1
        while end < len(buffer) and buffer[end] == ".":
            end += 1
        sentence = buffer[start:end]
        if sentence:
            sentences.append(sentence)
        start = end
        index = end
    return sentences, buffer[start:]


def _result_items(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("input_results", "output_results", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def create_guardrail_client(settings: Settings) -> GuardrailClient:
    client = BastionGuardianClient(
        base_url=settings.guardrail_base_url,
        endpoint_path=settings.guardrail_endpoint_path,
        api_key=settings.guardrail_api_key,
        timeout_seconds=settings.guardrail_timeout_seconds,
        fail_open=settings.guardrail_fail_open,
    )
    logger.info(
        "======== AI 가드레일 준비 | 상태=%s | endpoint=%s | 장애정책=%s",
        "활성" if client.enabled else "API키 없음/PASS-through",
        settings.guardrail_endpoint_path,
        "fail-open" if settings.guardrail_fail_open else "fail-closed",
    )
    return client
