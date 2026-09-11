"""RP 세부 시나리오별 MCP 입력 구성·실행·출력 전처리 함수."""

from datetime import date
from typing import Any

from app.mcp.models import McpExecutionResult
from app.mcp.scenario_runtime import ScenarioMcpHandlerContext
from app.mcp.scenarios.contracts import (
    ScenarioMcpOutput,
    ScenarioMcpOutputContext,
)
from app.mcp.scenarios.helpers import (
    keyword_list,
    month_and_reference_date,
    month_value,
    text,
)
from app.renderables import ScenarioAnswer, create_table_renderable
from app.scenario_actions import (
    ScenarioActionDefinition,
    ScenarioActionInput,
    register_scenario_action,
)


def _apartment_rp_address_action() -> ScenarioActionDefinition:
    """아파트 조회 action을 코드에서 직접 커스터마이징하는 함수."""

    return ScenarioActionDefinition(
        agent_code="RP",
        detail_scenario_code="APARTMENT_RP_LIST",
        action_code="APARTMENT_RP_ADDRESS_REQUIRED",
        message="조회할 아파트명 또는 읍면동을 입력해 주세요.",
        inputs=(
            ScenarioActionInput(
                parameter_name="address",
                input_code="param1",
                label="아파트명 또는 읍면동",
                min_length=1,
                # 주소/아파트명은 MCP 검색키이므로 마스킹하지 않는다.
                guardrail_enabled=False,
            ),
        ),
    )


APARTMENT_RP_ADDRESS_ACTION = register_scenario_action(
    _apartment_rp_address_action()
)


async def rp_document_search(
    context: ScenarioMcpHandlerContext,
) -> McpExecutionResult:
    query = text(context.subagent.parameters.get("rag_query")) or context.refined_query
    keywords = keyword_list(
        context.subagent.parameters.get("keywords"),
        fallback_query=query,
    )
    return await context.call(
        step_code="RP_DOCUMENT_SEARCH",
        tool_name="databricks_hybrid_search",
        arguments={
            "query": query,
            "keywords": keywords,
            "query_type": "HYBRID",
            "index_name": "rp_documents",
            "columns": [
                "document_id",
                "title",
                "content",
                "source_uri",
                "updated_at",
            ],
            "num_results": 5,
            "filters": {
                "document_category": "RP",
                "scenario_code": context.subagent.scenario_code,
                "detail_scenario_code": context.subagent.detail_scenario_code,
            },
        },
    )


async def apartment_rp_list(
    context: ScenarioMcpHandlerContext,
) -> McpExecutionResult:
    action_values = APARTMENT_RP_ADDRESS_ACTION.require(
        context.subagent.parameters
    )
    return await context.call(
        step_code="APARTMENT_RP_LIST",
        tool_name="test_tool",
        arguments={
            "param1": action_values["address"],
            "param2": "",
        },
        enabled=True,
        unavailable_message="아파트관리비 RP 조회 도구가 아직 연결되지 않았습니다.",
    )


async def composite_conversion_score(
    context: ScenarioMcpHandlerContext,
) -> McpExecutionResult:
    closing_month, reference_date = month_and_reference_date(
        context.subagent.parameters,
        today=date.today(),
        default_month="CURRENT",
    )
    return await context.call(
        step_code="RP_COMPOSITE_SCORE",
        tool_name="test_tool",
        arguments={"param1": closing_month, "param2": reference_date},
    )


async def composite_conversion_excluded(
    context: ScenarioMcpHandlerContext,
) -> McpExecutionResult:
    return await context.call(
        step_code="RP_COMPOSITE_EXCLUDED",
        tool_name="test_tool",
        arguments={
            "param1": month_value(
                context.subagent.parameters,
                today=date.today(),
                default_month="CURRENT",
            ),
            "param2": "",
        },
    )


# RP 결과도 중앙 columns 표가 아닌 detail별 Python 함수에서 직접 전처리한다.


def _display(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _two_column_output(
    context: ScenarioMcpOutputContext,
    *,
    title: str,
    first_label: str,
    second_label: str,
) -> ScenarioMcpOutput:
    items = context.data_items()
    values = {
        str(item.get("objId", "")).strip(): item.get("objVal")
        for item in items
        if str(item.get("objId", "")).strip()
    }
    first_value = values.get("column1", "")
    second_value = values.get("column2", "")
    return ScenarioMcpOutput(
        data=items,
        answer=ScenarioAnswer(
            text=(
                f"[{title}]\n"
                f"- {first_label}: {_display(first_value)}\n"
                f"- {second_label}: {_display(second_value)}"
            ),
            renderables=[
                create_table_renderable(
                    code="result-table",
                    title=title,
                    format="markdown",
                    columns=("항목", "값"),
                    rows=((first_label, first_value), (second_label, second_value)),
                )
            ],
        ),
    )


def apartment_rp_list_output(context: ScenarioMcpOutputContext) -> ScenarioMcpOutput:
    return _two_column_output(
        context,
        title="아파트관리비 RP 연결 가능 단지 조회 결과",
        first_label="아파트명",
        second_label="주소",
    )


def composite_conversion_score_output(context: ScenarioMcpOutputContext) -> ScenarioMcpOutput:
    return _two_column_output(
        context,
        title="복합환산 점수 및 실적 조회 결과",
        first_label="복합환산점수",
        second_label="실적건수",
    )


def composite_conversion_excluded_output(context: ScenarioMcpOutputContext) -> ScenarioMcpOutput:
    return _two_column_output(
        context,
        title="환산 미반영 내역 조회 결과",
        first_label="미반영내역",
        second_label="미반영사유",
    )
