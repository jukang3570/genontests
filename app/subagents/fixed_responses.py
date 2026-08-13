"""서브에이전트 단계에서 종료하는 고정 답변 시나리오를 관리한다.

마스터 EXCEPTION과는 별개다. 마스터는 정상 AGENT로 분류하고 서브에이전트가
특정 세부 시나리오를 선택한 뒤, 이 표에 등록된 응답을 MCP 호출 없이 반환한다.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SubagentFixedResponse:
    """하나의 세부 시나리오가 반환할 고정 답변 설정."""

    message: str


# 운영 커스터마이징 지점: 키는 (에이전트 코드, 세부 시나리오 코드)다.
SUBAGENT_FIXED_RESPONSES: dict[
    tuple[str, str], SubagentFixedResponse
] = {
    (
        "PERFORMANCE_FEE",
        "PERFORMANCE_FEE_INFORMATION_INQUIRY",
    ): SubagentFixedResponse(
        message="자세한 내용은 지점을 통해 확인해 주세요."
    ),
}


def get_subagent_fixed_response(
    agent_code: str,
    detail_scenario_code: str,
) -> SubagentFixedResponse | None:
    """등록된 서브 시나리오 고정 답변을 반환한다."""

    return SUBAGENT_FIXED_RESPONSES.get(
        (agent_code.strip().upper(), detail_scenario_code.strip().upper())
    )
