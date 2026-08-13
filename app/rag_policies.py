"""세부 시나리오별 RAG 품질 정책을 한 곳에서 관리한다.

운영자가 문서 검색 품질을 조정할 때 가장 먼저 수정할 파일이다. MCP payload나
답변 생성 코드를 찾지 않아도 각 세부 시나리오의 검색 임계점수, reranking 여부,
reranking 임계점수, 최종 문서 수와 고정 답변을 독립적으로 변경할 수 있다.
"""

from dataclasses import dataclass
from typing import Any

from app.observability import logger


@dataclass(frozen=True)
class RagScenarioPolicy:
    """하나의 문서 조회 세부 시나리오에 적용할 품질 정책."""

    retrieval_score_threshold: float
    reranking_enabled: bool
    reranking_score_threshold: float
    reranking_top_n: int
    answerability_check_enabled: bool
    no_documents_answer: str
    not_answerable_answer: str

    def __post_init__(self) -> None:
        """잘못된 임계값을 서버 시작 단계에서 즉시 발견한다."""

        for name, value in (
            ("retrieval_score_threshold", self.retrieval_score_threshold),
            ("reranking_score_threshold", self.reranking_score_threshold),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name}는 0 이상 1 이하이어야 합니다: {value}")
        if self.reranking_top_n < 1:
            raise ValueError("reranking_top_n은 1 이상이어야 합니다.")


_COMMON_NO_DOCUMENTS_ANSWER = (
    "질문과 관련된 업무 문서를 찾지 못했습니다. 질문을 조금 더 구체적으로 "
    "입력해 주세요."
)
_COMMON_NOT_ANSWERABLE_ANSWER = (
    "조회된 업무 문서만으로는 질문에 정확히 답변할 수 없습니다. 질문을 조금 "
    "더 구체적으로 입력하거나 관련 업무 담당 채널을 통해 확인해 주세요."
)


def _policy(
    *,
    retrieval: float,
    reranking: float,
    top_n: int = 5,
    reranking_enabled: bool = True,
    answerability_check_enabled: bool = True,
    no_documents_answer: str = _COMMON_NO_DOCUMENTS_ANSWER,
    not_answerable_answer: str = _COMMON_NOT_ANSWERABLE_ANSWER,
) -> RagScenarioPolicy:
    """반복되는 기본 문구를 공유하면서 시나리오별 값을 명시한다."""

    return RagScenarioPolicy(
        retrieval_score_threshold=retrieval,
        reranking_enabled=reranking_enabled,
        reranking_score_threshold=reranking,
        reranking_top_n=top_n,
        answerability_check_enabled=answerability_check_enabled,
        no_documents_answer=no_documents_answer,
        not_answerable_answer=not_answerable_answer,
    )


# ---------------------------------------------------------------------------
# 운영 커스터마이징 지점
# 키는 (서브에이전트 코드, 세부 시나리오 코드)다.
# 이 표에 등록된 세부 시나리오만 RAG 방식으로 처리한다.
# ---------------------------------------------------------------------------
RAG_SCENARIO_POLICIES: dict[tuple[str, str], RagScenarioPolicy] = {
    ("QUALIFICATION", "NEW_MEMBER_QUALIFICATION"): _policy(
        retrieval=0.60, reranking=0.50
    ),
    ("QUALIFICATION", "FOREIGNER_QUALIFICATION"): _policy(
        retrieval=0.62, reranking=0.52
    ),
    ("QUALIFICATION", "MINOR_QUALIFICATION"): _policy(
        retrieval=0.62, reranking=0.52
    ),
    ("QUALIFICATION", "FAMILY_CARD_ISSUANCE_QUALIFICATION"): _policy(
        retrieval=0.60, reranking=0.50
    ),
    ("QUALIFICATION", "INCOME_PROOF_ACCEPTANCE_CRITERIA"): _policy(
        retrieval=0.65, reranking=0.55
    ),
    # RP는 혼합형 에이전트다. 이 세부 시나리오만 Databricks RAG이며,
    # APARTMENT_RP_LIST와 복합환산 조회는 기존 고정 데이터 답변을 사용한다.
    ("RP", "RP_DOCUMENT_SEARCH"): _policy(
        retrieval=0.58,
        reranking=0.48,
        no_documents_answer=(
            "질문과 관련된 RP 업무 문서를 찾지 못했습니다. RP 업무명과 확인할 "
            "기준을 구체적으로 입력해 주세요."
        ),
        not_answerable_answer=(
            "조회된 RP 업무 문서만으로는 질문에 정확히 답변할 수 없습니다. "
            "질문을 구체적으로 입력하거나 RP 업무 담당 채널을 통해 확인해 주세요."
        ),
    ),
}


def get_rag_policy(
    agent_code: str,
    detail_scenario_code: str,
) -> RagScenarioPolicy | None:
    """세부 시나리오의 RAG 정책을 반환하며 미등록 조회는 고정 데이터로 본다."""

    return RAG_SCENARIO_POLICIES.get(
        (agent_code.strip().upper(), detail_scenario_code.strip().upper())
    )


def filter_documents_by_retrieval_score(
    *,
    documents: list[dict[str, Any]],
    threshold: float,
    agent_code: str,
    detail_scenario_code: str,
) -> list[dict[str, Any]]:
    """Databricks 검색 점수가 임계값 이상인 문서만 반환한다.

    점수가 없거나 숫자가 아닌 문서는 품질을 판단할 수 없으므로 제외한다.
    """

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for document in documents:
        metadata = document.get("metadata")
        score = metadata.get("score") if isinstance(metadata, dict) else None
        if (
            isinstance(score, (int, float))
            and not isinstance(score, bool)
            and float(score) >= threshold
        ):
            selected.append(document)
            continue
        rejected.append(
            {
                "document_id": document.get("document_id"),
                "score": score,
            }
        )

    logger.info(
        "======== 문서 검색 임계값 적용 완료\n"
        "에이전트=%s\n세부시나리오=%s\n임계점수=%.3f\n"
        "입력문서개수=%d\n통과문서개수=%d\n제외문서=%s",
        agent_code,
        detail_scenario_code,
        threshold,
        len(documents),
        len(selected),
        rejected,
    )
    return selected
