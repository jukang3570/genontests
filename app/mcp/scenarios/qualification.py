"""QUALIFICATION 문서 검색 MCP 실행 함수.

RAG 검색어는 프론트 Action으로 받지 않는다. 마스터 보정 질문과 서브에이전트가
세부 시나리오별로 추출한 keywords를 Databricks 하이브리드 검색에 전달한다.
"""

from app.mcp.models import McpExecutionResult
from app.mcp.scenario_runtime import ScenarioMcpHandlerContext
from app.mcp.scenarios.helpers import keyword_list, text


async def qualification_document_search(
    context: ScenarioMcpHandlerContext,
) -> McpExecutionResult:
    """현재 detail을 filter로 유지한 Databricks 하이브리드 검색 1회."""

    query = text(context.subagent.parameters.get("rag_query")) or context.refined_query
    keywords = keyword_list(
        context.subagent.parameters.get("keywords"),
        fallback_query=query,
    )
    return await context.call(
        step_code="QUALIFICATION_DOCUMENT_SEARCH",
        tool_name="databricks_hybrid_search",
        arguments={
            "query": query,
            "keywords": keywords,
            "query_type": "HYBRID",
            "index_name": "qualification_documents",
            "columns": [
                "document_id",
                "title",
                "content",
                "source_uri",
                "updated_at",
            ],
            "num_results": 5,
            "filters": {
                "document_category": "QUALIFICATION",
                "scenario_code": context.subagent.scenario_code,
                "detail_scenario_code": context.subagent.detail_scenario_code,
            },
        },
    )
