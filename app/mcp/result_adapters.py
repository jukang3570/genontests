"""조회형 MCP 결과에서 필요한 컬럼만 뽑아 시나리오별 답변을 만든다.

운영자가 수정할 핵심은 두 곳뿐이다.

1. ``SCENARIO_QUERY_CONFIGS``에서 세부 시나리오별 사용할 ``columns``를 정한다.
2. 연결된 ``answer_formatter(data, parameters, request_context)`` 함수에서
   원하는 문장, 표, 합계와 조건문을 자유롭게 작성한다.

HTTP 호출과 원본 응답 파싱은 ``client.py``, MCP 요청 payload는 ``payloads.py``가
담당한다. 이 파일은 조회 결과 선택과 최종 답변 형식만 담당한다.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.mcp.models import McpExecutionResult
from app.observability import logger, timed
from app.rag_policies import RAG_SCENARIO_POLICIES
from app.renderables import (
    ScenarioAnswer,
    create_table_renderable,
    extract_data_items,
    extract_value,
    normalize_scenario_answer,
)
from app.subagents.models import SubagentResult


QUERY_RESULT_FORMAT = "query.v1"
RAG_RAW_RESULT_FORMAT = "raw.rag"


class McpResultFormatError(ValueError):
    """MCP 결과가 약속한 data 배열 형식이 아닐 때 발생한다."""


# 포맷 함수가 받는 인자는 의도적으로 단순하게 유지한다.
# data: columns에 지정한 objId만 남긴 MCP data 목록
# parameters: 서브에이전트가 추출한 파라미터 전체
# request_context: AccessToken을 제외한 사용자·세션·시나리오 정보
QueryAnswerFormatter = Callable[
    [list[dict[str, Any]], dict[str, Any], dict[str, Any]],
    Any,
]


@dataclass(frozen=True)
class ScenarioQueryConfig:
    """세부 시나리오 하나의 조회 컬럼과 최종 답변 함수."""

    columns: tuple[str, ...]
    answer_formatter: QueryAnswerFormatter


def filter_mcp_data(
    structured_content: Any,
    columns: Sequence[str],
) -> list[dict[str, Any]]:
    """structuredContent.data에서 지정한 objId 항목만 원래 순서로 반환한다.

    같은 objId가 여러 번 등장해도 목록형 조회 결과일 수 있으므로 모두 유지한다.
    반환값은 포맷 함수가 자유롭게 사용할 수 있도록 MCP 항목 dict를 보존한다.
    """

    try:
        raw_data = extract_data_items(structured_content)
    except ValueError as exc:
        raise McpResultFormatError(str(exc)) from exc

    selected_columns = {
        str(column).strip() for column in columns if str(column).strip()
    }
    include_all_columns = "*" in selected_columns
    selected_data: list[dict[str, Any]] = []
    for index, item in enumerate(raw_data):
        if not isinstance(item, Mapping):
            raise McpResultFormatError(
                f"structuredContent.data[{index}]는 object 형식이어야 합니다."
            )
        obj_id = item.get("objId")
        if not isinstance(obj_id, str) or not obj_id.strip():
            raise McpResultFormatError(
                f"structuredContent.data[{index}].objId가 올바르지 않습니다."
            )
        if include_all_columns or obj_id.strip() in selected_columns:
            selected_data.append(dict(item))
    return selected_data


def _value_text(value: Any) -> str:
    """조회값을 한글이 보존되는 화면 문자열로 바꾼다."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _two_column_answer(
    data: list[dict[str, Any]],
    *,
    title: str,
    first_label: str,
    second_label: str,
) -> ScenarioAnswer:
    """현재 test_tool 결과로 본문과 Markdown table renderable을 만든다."""

    first_value = extract_value(data, "column1")
    second_value = extract_value(data, "column2")
    text = (
        f"[{title}]\n"
        f"- {first_label}: {_value_text(first_value)}\n"
        f"- {second_label}: {_value_text(second_value)}"
    )
    table = create_table_renderable(
        code="result-table",
        title=title,
        format="markdown",
        columns=("항목", "값"),
        rows=(
            (first_label, first_value),
            (second_label, second_value),
        ),
    )
    return ScenarioAnswer(text=text, renderables=[table])


# -------------------------------------------------------------------------
# 세부 시나리오별 최종 답변 포맷 함수
# -------------------------------------------------------------------------
# 각 함수의 본문은 운영 업무에 맞게 자유롭게 수정한다.
# data는 이미 columns 기준으로 필터링된 list[dict]이며, parameters와
# request_context를 함께 사용해 조건문, 반복문, 합계, 표 형식을 만들 수 있다.


def format_performance_summary_total(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """실적 종합 조회 최종 답변."""

    rows = [
        (item.get("objId", ""), item.get("objVal", ""))
        for item in data
    ]
    text = "[실적 종합 조회 결과]"
    if rows:
        text += "\n" + "\n".join(
            f"- {_value_text(name)}: {_value_text(value)}"
            for name, value in rows
        )
    else:
        text += "\n- 조회된 실적 항목이 없습니다."
    return ScenarioAnswer(
        text=text,
        renderables=[
            create_table_renderable(
                code="result-table",
                title="실적 종합 조회 결과",
                format="markdown",
                columns=("항목", "값"),
                rows=rows,
            )
        ],
    )


def format_performance_fee_composite_score(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """PERFORMANCE_FEE 복합환산 점수 및 실적 답변."""

    return _two_column_answer(
        data,
        title="복합환산 점수 및 실적 조회 결과",
        first_label="복합환산점수",
        second_label="실적건수",
    )


def format_performance_fee_composite_excluded(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """PERFORMANCE_FEE 환산 미반영 내역 답변."""

    return _two_column_answer(
        data,
        title="환산 미반영 내역 조회 결과",
        first_label="미반영내역",
        second_label="미반영사유",
    )


def format_unregistered_member_summary(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """미등록 회원 현황 답변."""

    return _two_column_answer(
        data,
        title="미등록 회원 현황 조회 결과",
        first_label="유형",
        second_label="건수",
    )


def format_unregistered_member_handoff_detail(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """타인수령 및 업체인계 상세 답변."""

    return _two_column_answer(
        data,
        title="타인수령 및 업체인계 상세 조회 결과",
        first_label="대상",
        second_label="상세내역",
    )


def format_disposal_fee_summary(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """폐기 환산 및 회입 수수료 답변."""

    return _two_column_answer(
        data,
        title="폐기 환산 및 회입 수수료 조회 결과",
        first_label="폐기환산정보",
        second_label="회입수수료",
    )


def format_disposal_fee_customer_detail(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """고객별 폐기 환산 상세 답변."""

    return _two_column_answer(
        data,
        title="고객별 폐기 환산 상세 조회 결과",
        first_label="폐기사유",
        second_label="폐기상세",
    )


def format_fee_item_details(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """항목별 수수료 답변."""

    return _two_column_answer(
        data,
        title="항목별 수수료 조회 결과",
        first_label="수수료항목",
        second_label="수수료금액",
    )


def format_fee_tax_net_payment(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """세금 및 실지급액 답변."""

    return _two_column_answer(
        data,
        title="세금 및 실지급액 조회 결과",
        first_label="세금·공제내역",
        second_label="실지급액",
    )


def format_fee_12_month_trend(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """12개월 수수료 추이 답변."""

    return _two_column_answer(
        data,
        title="12개월 수수료 추이 조회 결과",
        first_label="기준월",
        second_label="수수료금액",
    )


def format_withholding_tax(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """원천징수 내역 답변."""

    return _two_column_answer(
        data,
        title="원천징수 내역 조회 결과",
        first_label="원천징수항목",
        second_label="원천징수금액",
    )


def format_rp_apartment_list(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """아파트관리비 RP 연결 가능 단지 답변."""

    return _two_column_answer(
        data,
        title="아파트관리비 RP 연결 가능 단지 조회 결과",
        first_label="아파트명",
        second_label="주소",
    )


def format_rp_composite_score(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """RP 복합환산 점수 및 실적 답변."""

    return _two_column_answer(
        data,
        title="복합환산 점수 및 실적 조회 결과",
        first_label="복합환산점수",
        second_label="실적건수",
    )


def format_rp_composite_excluded(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """RP 환산 미반영 내역 답변."""

    return _two_column_answer(
        data,
        title="환산 미반영 내역 조회 결과",
        first_label="미반영내역",
        second_label="미반영사유",
    )


# -------------------------------------------------------------------------
# 세부 시나리오별 컬럼과 답변 함수 연결
# -------------------------------------------------------------------------
# 실제 MCP 컬럼이 바뀌면 columns만 수정한다.
# 답변 모양이 바뀌면 연결된 format_* 함수 본문만 수정한다.
SCENARIO_QUERY_CONFIGS: dict[tuple[str, str], ScenarioQueryConfig] = {
    ("PERFORMANCE_FEE", "PERFORMANCE_SUMMARY_TOTAL"): ScenarioQueryConfig(
        # 실제 EAI 응답의 objId 전체를 보존한다. 운영 명칭이 확정되면 필요한
        # objId만 명시하고 format_performance_summary_total에서 한글 라벨을 매핑한다.
        columns=("*",),
        answer_formatter=format_performance_summary_total,
    ),
    ("PERFORMANCE_FEE", "COMPOSITE_CONVERSION_SCORE"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_performance_fee_composite_score,
    ),
    (
        "PERFORMANCE_FEE",
        "COMPOSITE_CONVERSION_EXCLUDED",
    ): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_performance_fee_composite_excluded,
    ),
    ("PERFORMANCE_FEE", "UNREGISTERED_MEMBER_SUMMARY"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_unregistered_member_summary,
    ),
    (
        "PERFORMANCE_FEE",
        "UNREGISTERED_MEMBER_HANDOFF_DETAIL",
    ): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_unregistered_member_handoff_detail,
    ),
    ("PERFORMANCE_FEE", "DISPOSAL_FEE_SUMMARY"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_disposal_fee_summary,
    ),
    (
        "PERFORMANCE_FEE",
        "DISPOSAL_FEE_CUSTOMER_DETAIL",
    ): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_disposal_fee_customer_detail,
    ),
    ("PERFORMANCE_FEE", "FEE_ITEM_DETAILS"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_fee_item_details,
    ),
    ("PERFORMANCE_FEE", "FEE_TAX_NET_PAYMENT"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_fee_tax_net_payment,
    ),
    ("PERFORMANCE_FEE", "FEE_12_MONTH_TREND"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_fee_12_month_trend,
    ),
    ("PERFORMANCE_FEE", "WITHHOLDING_TAX"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_withholding_tax,
    ),
    ("RP", "APARTMENT_RP_LIST"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_rp_apartment_list,
    ),
    ("RP", "COMPOSITE_CONVERSION_SCORE"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_rp_composite_score,
    ),
    ("RP", "COMPOSITE_CONVERSION_EXCLUDED"): ScenarioQueryConfig(
        columns=("column1", "column2"),
        answer_formatter=format_rp_composite_excluded,
    ),
}


# RAG 결과는 조회형 objId 형식과 다른 문서 목록이므로 세부 시나리오 단위로
# 원본을 유지한다. RP처럼 RAG와 조회형 업무가 섞인 에이전트도 지원한다.
RAG_PASSTHROUGH_SCENARIOS = frozenset(RAG_SCENARIO_POLICIES)
# 이전 외부 import 호환용이다. 실제 분기는 위 세부 시나리오 표를 사용한다.
RAG_PASSTHROUGH_AGENTS = frozenset({"QUALIFICATION"})


@timed("MCP 조회 결과 정제")
def adapt_mcp_result(
    *,
    execution: McpExecutionResult,
    subagent: SubagentResult,
    employee_id: str,
    session_id: str,
    thread_id: str,
    request_context: Mapping[str, Any],
) -> McpExecutionResult:
    """컬럼 필터링 후 시나리오 전용 포맷 함수로 답변을 만든다."""

    if (
        execution.outcome != "SUCCESS"
        or not execution.succeeded
        or execution.result is None
    ):
        return execution

    agent_code = subagent.agent_code.upper()
    detail_code = subagent.detail_scenario_code
    if (agent_code, detail_code) in RAG_PASSTHROUGH_SCENARIOS:
        logger.info(
            "======== MCP 결과 정제 생략 | 에이전트=%s | "
            "세부시나리오=%s | 결과형식=%s | 이유=RAG원본유지",
            agent_code,
            detail_code,
            RAG_RAW_RESULT_FORMAT,
        )
        return execution.model_copy(
            update={"result_format": RAG_RAW_RESULT_FORMAT}
        )

    key = (agent_code, detail_code)
    config = SCENARIO_QUERY_CONFIGS.get(key)
    if config is None:
        raise McpResultFormatError(
            "조회형 세부 시나리오 설정이 없습니다: "
            f"agent_code={agent_code}, detail_scenario_code={detail_code}. "
            "app/mcp/result_adapters.py의 SCENARIO_QUERY_CONFIGS를 확인하세요."
        )

    selected_data = filter_mcp_data(execution.result, config.columns)
    parameters = dict(subagent.parameters)
    safe_request_context = _build_formatter_request_context(
        request_context=request_context,
        employee_id=employee_id,
        session_id=session_id,
        thread_id=thread_id,
        subagent=subagent,
    )
    scenario_answer = normalize_scenario_answer(
        config.answer_formatter(
            selected_data,
            parameters,
            safe_request_context,
        ),
        default_renderable_code=f"{agent_code}:{detail_code}:table",
    )
    answer_text = scenario_answer.text
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise McpResultFormatError(
            "시나리오 답변 포맷 함수가 비어 있는 문자열을 반환했습니다: "
            f"agent_code={agent_code}, detail_scenario_code={detail_code}"
        )

    renderables = []
    for renderable in scenario_answer.renderables:
        renderable_metadata = {
            **renderable.metadata,
            "agent_code": agent_code,
            "scenario_code": subagent.scenario_code,
            "detail_scenario_code": detail_code,
        }
        renderable_code = renderable.code
        if ":" not in renderable_code:
            renderable_code = f"{agent_code}:{detail_code}:{renderable_code}"
        renderables.append(
            renderable.model_copy(
                update={
                    "code": renderable_code,
                    "metadata": renderable_metadata,
                }
            ).model_dump(mode="json")
        )

    formatted_result = {
        "format": QUERY_RESULT_FORMAT,
        "adapter_code": f"{agent_code}:{detail_code}",
        "columns": list(config.columns),
        "data": selected_data,
        "parameters": parameters,
        "request_context": safe_request_context,
        "answer_text": answer_text,
        "renderables": renderables,
    }
    logger.info(
        "======== MCP 조회 결과 정제 완료 | 에이전트=%s | "
        "세부시나리오=%s | 결과형식=%s | 선택컬럼=%s | "
        "정제데이터=%s | 답변=%s | 확장데이터개수=%d",
        agent_code,
        detail_code,
        QUERY_RESULT_FORMAT,
        list(config.columns),
        selected_data,
        answer_text,
        len(renderables),
    )
    return execution.model_copy(
        update={
            "result_format": QUERY_RESULT_FORMAT,
            "formatted_result": formatted_result,
        }
    )


def _build_formatter_request_context(
    *,
    request_context: Mapping[str, Any],
    employee_id: str,
    session_id: str,
    thread_id: str,
    subagent: SubagentResult,
) -> dict[str, Any]:
    """포맷 함수에 전달할 요청 context를 만들되 AccessToken은 제외한다."""

    user = request_context.get("user")
    if not isinstance(user, Mapping):
        user = {}
    return {
        "employee_id": employee_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "agent_code": subagent.agent_code,
        "scenario_code": subagent.scenario_code,
        "detail_scenario_code": subagent.detail_scenario_code,
        "endpoint": request_context.get("endpoint", ""),
        "recruitment_org_type_code": request_context.get(
            "recruitment_org_type_code",
            "",
        ),
        "user": {
            "id": user.get("id"),
            "deptcode": user.get("deptcode"),
            "deptname": user.get("deptname"),
        },
    }
