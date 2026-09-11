"""세부 시나리오별 수동 추천질문을 프론트 출력 형식으로 변환한다."""

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from app.observability import logger
from app.subagents.models import SubagentResult
from app.subagents.prompt_loader import ScenarioPromptBundle, SubagentPromptLoader


# 후속 업무를 실행하는 표현이므로 부분 문자열이나 LLM 추측을 사용하지 않는다.
# 공백·일반 문장부호만 제거한 전체 문장이 아래 값과 정확히 같을 때만 동의로 본다.
_AFFIRMATIVE_RESPONSES = frozenset(
    {
        "네",
        "예",
        "응",
        "그래",
        "좋아",
        "좋아요",
        "동의해",
        "동의합니다",
        "그렇게해줘",
        "그렇게해주세요",
        "진행해줘",
        "진행해주세요",
        "해주세요",
        "해줘",
        "ok",
        "okay",
        "yes",
    }
)


@dataclass(frozen=True)
class RecommendedQuestionDefinition:
    """manifest에 고정 등록된 추천질문 한 건과 추적용 시나리오 정보."""

    agent_code: str
    prompt_version: str
    scenario_code: str
    detail_scenario_code: str
    question: str
    order: int
    affirmative_followup_message: str | None = None
    affirmative_detail_scenario_code: str | None = None

    def to_event_item(self) -> dict[str, Any]:
        """프론트에 전달할 camelCase 추천질문 객체를 반환한다."""

        item: dict[str, Any] = {
            "id": (f"{self.agent_code}:{self.detail_scenario_code}:{self.order}"),
            "question": self.question,
            "interactionType": (
                "confirmation"
                if self.affirmative_followup_message is not None
                else "prompt"
            ),
            "agentCode": self.agent_code,
            "promptVersion": self.prompt_version,
            "scenarioCode": self.scenario_code,
            "detailScenarioCode": self.detail_scenario_code,
        }
        if (
            self.affirmative_followup_message is not None
            and self.affirmative_detail_scenario_code is not None
        ):
            item["affirmativeFollowup"] = {
                "message": self.affirmative_followup_message,
                "agentCode": self.agent_code,
                "detailScenarioCode": self.affirmative_detail_scenario_code,
            }
        return item


class RecommendedQuestionRegistry:
    """활성 manifest의 모든 세부 시나리오와 추천질문을 시작 시 검증한다.

    추천질문은 선택사항이며 개수의 상한도 두지 않는다. 필드가 없거나 빈
    배열이면 해당 detail은 추천질문을 보내지 않는다. 유효한 질문만 정규화하며
    YAML 배열 순서가 프론트 표시 순서다.
    """

    def __init__(
        self,
        definitions: Mapping[
            tuple[str, str],
            tuple[RecommendedQuestionDefinition, ...],
        ],
    ) -> None:
        self._definitions = MappingProxyType(dict(definitions))

    @classmethod
    def from_bundles(
        cls,
        bundles: Mapping[str, ScenarioPromptBundle],
    ) -> "RecommendedQuestionRegistry":
        definitions: dict[
            tuple[str, str],
            tuple[RecommendedQuestionDefinition, ...],
        ] = {}

        for bundle in bundles.values():
            agent_code = bundle.agent_code.upper()
            bundle_detail_codes = {
                str(detail["code"])
                for scenario in bundle.manifest["scenarios"]
                for detail in scenario["details"]
            }
            for scenario in bundle.manifest["scenarios"]:
                scenario_code = str(scenario["code"])
                for detail in scenario["details"]:
                    detail_code = str(detail["code"])
                    raw_questions = detail.get("recommended_questions")
                    location = f"{agent_code}.{scenario_code}.{detail_code}"
                    key = (agent_code, detail_code)
                    if key in definitions:
                        raise ValueError(
                            "같은 에이전트 안에 중복 세부 시나리오 코드가 있습니다: "
                            f"{location}"
                        )

                    if raw_questions is None:
                        raw_questions = []
                    elif isinstance(raw_questions, str):
                        # 운영 중 실수로 단일 문자열을 넣어도 서버 시작을 막지
                        # 않고 질문 한 개짜리 배열로 정규화한다.
                        raw_questions = [raw_questions]
                    elif not isinstance(raw_questions, list):
                        logger.warning(
                            "!!!!!!!! 추천질문 설정 무시 | 위치=%s | 타입=%s | "
                            "이유=문자열 또는 배열 아님",
                            location,
                            type(raw_questions).__name__,
                        )
                        raw_questions = []

                    questions: list[tuple[str, str | None, str | None]] = []
                    for index, raw_question in enumerate(raw_questions, start=1):
                        raw_followup: Any = None
                        if isinstance(raw_question, Mapping):
                            raw_followup = raw_question.get("affirmative_followup")
                            raw_question = raw_question.get("question")
                        if (
                            not isinstance(raw_question, str)
                            or not raw_question.strip()
                        ):
                            logger.warning(
                                "!!!!!!!! 추천질문 항목 무시 | 위치=%s[%d] | "
                                "타입=%s | 이유=비어 있지 않은 문자열 아님",
                                location,
                                index,
                                type(raw_question).__name__,
                            )
                            continue
                        normalized_question = raw_question.strip()
                        if any(
                            question == normalized_question
                            for question, _, _ in questions
                        ):
                            logger.warning(
                                "!!!!!!!! 중복 추천질문 무시 | 위치=%s[%d]",
                                location,
                                index,
                            )
                            continue
                        followup_message: str | None = None
                        followup_detail_code: str | None = None
                        if isinstance(raw_followup, Mapping):
                            configured_message = raw_followup.get("message")
                            configured_detail = raw_followup.get("detail_scenario_code")
                            if (
                                isinstance(configured_message, str)
                                and configured_message.strip()
                                and isinstance(configured_detail, str)
                                and configured_detail in bundle_detail_codes
                            ):
                                followup_message = configured_message.strip()
                                followup_detail_code = configured_detail
                            else:
                                logger.warning(
                                    "!!!!!!!! 추천질문 긍정 후속 설정 무시 | "
                                    "위치=%s[%d] | 이유=message 또는 대상 detail 오류",
                                    location,
                                    index,
                                )
                        questions.append(
                            (
                                normalized_question,
                                followup_message,
                                followup_detail_code,
                            )
                        )

                    definitions[key] = tuple(
                        RecommendedQuestionDefinition(
                            agent_code=agent_code,
                            prompt_version=bundle.version,
                            scenario_code=scenario_code,
                            detail_scenario_code=detail_code,
                            question=question,
                            order=index,
                            affirmative_followup_message=followup_message,
                            affirmative_detail_scenario_code=followup_detail_code,
                        )
                        for index, (
                            question,
                            followup_message,
                            followup_detail_code,
                        ) in enumerate(questions, start=1)
                    )

        return cls(definitions)

    def for_subagent(
        self,
        subagent: SubagentResult | None,
    ) -> list[dict[str, Any]]:
        """선택된 세부 시나리오 순서대로 모든 수동 추천질문을 반환한다."""

        if subagent is None:
            return []

        items: list[dict[str, Any]] = []
        agent_code = subagent.agent_code.upper()
        for match in subagent.matches:
            key = (agent_code, match.detail_scenario_code)
            for definition in self._definitions.get(key, ()):
                items.append(definition.to_event_item())
        return items


def create_recommended_question_registry() -> RecommendedQuestionRegistry:
    """활성 서브에이전트 manifest에서 추천질문 registry를 생성한다."""

    return RecommendedQuestionRegistry.from_bundles(SubagentPromptLoader().load_all())


def resolve_affirmative_recommendation(
    message: str,
    history: list[dict[str, Any]],
    recommendation_id: str | None = None,
) -> dict[str, str] | None:
    """버튼 선택 또는 직전 assistant 제안의 명시적 동의를 후속 업무로 해석한다.

    오래된 추천질문을 실행하지 않도록 이력의 마지막 항목이 assistant일 때만
    확인한다. 추천질문 ID가 있으면 그 항목만 선택한다. ID가 없는 ``네``는 한
    답변에 긍정 후속 업무가 정확히 하나일 때만 실행한다.
    """

    normalized_message = re.sub(r"[\s.,!?~。！？]+", "", message).casefold()
    candidates = _latest_affirmative_candidates(history)
    if recommendation_id is not None:
        selected = [
            candidate
            for candidate in candidates
            if candidate["recommendation_id"] == recommendation_id
        ]
        if len(selected) == 1:
            return _followup_result(selected[0])
        logger.info(
            "======== 추천질문 ID 선택 무시 | ID=%s | 일치개수=%d",
            recommendation_id,
            len(selected),
        )
        return None

    # 구형 프론트가 버튼 클릭 시 followup.message만 보내는 경우도 정확히
    # 일치하는 문장 하나에 한해 계속 지원한다.
    if normalized_message not in _AFFIRMATIVE_RESPONSES:
        exact = [
            candidate
            for candidate in candidates
            if candidate["message"].casefold() == message.strip().casefold()
        ]
        return _followup_result(exact[0]) if len(exact) == 1 else None

    if len(candidates) != 1:
        if candidates:
            logger.info(
                "======== 추천질문 긍정 후속 자동연결 생략 | "
                "이유=대상여러개 | 후보개수=%d",
                len(candidates),
            )
        return None
    return _followup_result(candidates[0])


def ambiguous_affirmative_recommendations(
    message: str,
    history: list[dict[str, Any]],
    recommendation_id: str | None = None,
) -> list[dict[str, Any]]:
    """대상 없는 ``네``가 둘 이상의 실행 제안을 가리키면 선택지를 반환한다."""

    normalized_message = re.sub(r"[\s.,!?~。！？]+", "", message).casefold()
    candidates = _latest_affirmative_candidates(history)
    # 존재하지 않거나 중복된 ID를 받은 경우에도 자연어 LLM에 선택을 추측시키지
    # 않고, 현재 assistant의 실행 가능한 버튼을 다시 보여 준다.
    if recommendation_id is not None:
        return [dict(candidate["event_item"]) for candidate in candidates]
    if normalized_message not in _AFFIRMATIVE_RESPONSES:
        return []
    if len(candidates) <= 1:
        return []
    return [dict(candidate["event_item"]) for candidate in candidates]


def _latest_affirmative_candidates(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """직전 assistant metadata에서 실행 가능한 추천질문만 안전하게 추출한다."""

    if not history:
        return []
    latest = history[-1]
    if str(latest.get("role", "")).casefold() != "assistant":
        return []
    metadata = latest.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    raw_questions = metadata.get("recommendedQuestions")
    if not isinstance(raw_questions, list):
        return []

    candidates: list[dict[str, Any]] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, Mapping):
            continue
        recommendation_id = raw_question.get("id")
        raw_followup = raw_question.get("affirmativeFollowup")
        if not isinstance(raw_followup, Mapping):
            continue
        followup_message = raw_followup.get("message")
        agent_code = raw_followup.get("agentCode")
        detail_code = raw_followup.get("detailScenarioCode")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                recommendation_id,
                followup_message,
                agent_code,
                detail_code,
            )
        ):
            continue
        candidates.append(
            {
                "recommendation_id": str(recommendation_id).strip(),
                "message": str(followup_message).strip(),
                "agent_code": str(agent_code).strip().upper(),
                "detail_scenario_code": str(detail_code).strip(),
                "event_item": dict(raw_question),
            }
        )
    return candidates


def _followup_result(candidate: Mapping[str, Any]) -> dict[str, str]:
    """내부 후보에서 그래프에 필요한 값만 반환한다."""

    return {
        "message": str(candidate["message"]),
        "agent_code": str(candidate["agent_code"]),
        "detail_scenario_code": str(candidate["detail_scenario_code"]),
    }
