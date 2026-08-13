"""세부 시나리오별 MCP 도구명, 기본값, 필수값과 payload를 직접 관리한다.

프롬프트 manifest는 LLM이 추출할 필드와 시나리오만 정의한다. 날짜 기본값,
필수값 검증, 필드 조합과 실제 MCP arguments는 운영자가 이 파일에서 확인하고
수정한다.
"""

from datetime import date, timedelta
from typing import Any

from app.subagents.models import SubagentResult


class McpParameterInputRequired(ValueError):
    """MCP 호출 전 사용자에게 추가 입력을 받아야 함을 나타낸다."""

    def __init__(
        self,
        *,
        input_code: str,
        parameter_name: str,
        label: str,
        message: str,
        input_type: str = "text",
    ) -> None:
        self.input_code = input_code
        self.parameter_name = parameter_name
        self.label = label
        self.message = message
        self.input_type = input_type
        super().__init__(message)


def build_mcp_payload(
    *,
    request_id: str,
    subagent: SubagentResult,
    employee_id: str,
    session_id: str,
    thread_id: str,
    request_context: dict[str, Any],
    today: date | None = None,
) -> dict[str, Any] | None:
    """선택된 세부 시나리오의 최종 JSON-RPC payload를 만든다.

    새 MCP 도구를 연결할 때 아래 ``match``에서 에이전트 코드와 세부 시나리오
    코드를 찾아 도구명, 기본값, 필수값과 arguments를 직접 수정한다.
    """

    agent_code = subagent.agent_code.upper()
    detail_code = subagent.detail_scenario_code
    parameters = subagent.parameters
    reference_today = today or date.today()

    # ------------------------------------------------------------------
    # PERFORMANCE_FEE
    # ------------------------------------------------------------------
    match (agent_code, detail_code):
        case ("PERFORMANCE_FEE", "PERFORMANCE_SUMMARY_TOTAL"):
            closing_month, reference_date = _month_and_reference_date(
                parameters,
                today=reference_today,
                default_month="CURRENT",
            )
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": closing_month,
                    "param2": reference_date,
                },
            )

        case ("PERFORMANCE_FEE", "COMPOSITE_CONVERSION_SCORE"):
            closing_month, reference_date = _month_and_reference_date(
                parameters,
                today=reference_today,
                default_month="CURRENT",
            )
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": closing_month,
                    "param2": reference_date,
                },
            )

        case ("PERFORMANCE_FEE", "COMPOSITE_CONVERSION_EXCLUDED"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="CURRENT",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "UNREGISTERED_MEMBER_SUMMARY"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="CURRENT",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "UNREGISTERED_MEMBER_HANDOFF_DETAIL"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="CURRENT",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "DISPOSAL_FEE_SUMMARY"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="PREVIOUS",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "DISPOSAL_FEE_CUSTOMER_DETAIL"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="PREVIOUS",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "FEE_ITEM_DETAILS"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="PREVIOUS",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "FEE_TAX_NET_PAYMENT"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="PREVIOUS",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "FEE_12_MONTH_TREND"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="CURRENT",
                    ),
                    "param2": "",
                },
            )

        case ("PERFORMANCE_FEE", "WITHHOLDING_TAX"):
            reference_year = _text(parameters.get("reference_year"))
            if not reference_year:
                reference_year = str(reference_today.year - 1)
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={"param1": reference_year, "param2": ""},
            )

        # --------------------------------------------------------------
        # QUALIFICATION
        # 문서 조회 대상인 3개 상위 시나리오의 모든 세부 업무는 동일한 Databricks 하이브리드
        # 검색 MCP를 호출한다. 시나리오별 차이는 검색 query와 filter로 전달한다.
        # --------------------------------------------------------------
        case (
            "QUALIFICATION",
            (
                "NEW_MEMBER_QUALIFICATION"
                | "FOREIGNER_QUALIFICATION"
                | "MINOR_QUALIFICATION"
                | "FAMILY_CARD_ISSUANCE_QUALIFICATION"
                | "INCOME_PROOF_ACCEPTANCE_CRITERIA"
            ),
        ):
            # 운영자가 가장 자주 변경할 부분이다. 실제 GenOS MCP에 등록한 도구
            # 이름과 Databricks 검색 도구의 input schema에 맞춰 이 분기의
            # tool_name/arguments만 수정하면 된다.
            search_query = _require_text(
                parameters.get("search_query"),
                input_code="search_query",
                parameter_name="search_query",
                label="자격기준 문서 검색어",
                message="검색할 자격기준 규정이나 업무 내용을 입력해 주세요.",
            )
            return _make_payload(
                request_id,
                tool_name="databricks_hybrid_search",
                arguments={
                    "query": search_query,
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
                        "scenario_code": subagent.scenario_code,
                        "detail_scenario_code": detail_code,
                    },
                },
            )

        # --------------------------------------------------------------
        # RP
        # --------------------------------------------------------------
        case ("RP", "RP_DOCUMENT_SEARCH"):
            # RP는 조회형과 RAG가 함께 있는 혼합형 에이전트다. 문서 조회
            # 세부 시나리오만 Databricks 하이브리드 검색 도구를 호출한다.
            search_query = _require_text(
                parameters.get("search_query"),
                input_code="search_query",
                parameter_name="search_query",
                label="RP 업무 문서 검색어",
                message="검색할 RP 업무 기준이나 정책을 입력해 주세요.",
            )
            return _make_payload(
                request_id,
                tool_name="databricks_hybrid_search",
                arguments={
                    "query": search_query,
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
                        "scenario_code": subagent.scenario_code,
                        "detail_scenario_code": detail_code,
                    },
                },
            )

        case ("RP", "APARTMENT_RP_LIST"):
            # 필수 파라미터 입력 action 테스트 대상이다. LLM이 주소를 추출하지
            # 못하면 test_tool을 호출하지 않고 param1 입력을 프론트에 요청한다.
            address = _require_text(
                parameters.get("address"),
                input_code="param1",
                parameter_name="address",
                label="아파트명 또는 읍면동",
                message="조회할 아파트명 또는 읍면동을 입력해 주세요.",
            )
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={"param1": address, "param2": ""},
            )

        case ("RP", "COMPOSITE_CONVERSION_SCORE"):
            closing_month, reference_date = _month_and_reference_date(
                parameters,
                today=reference_today,
                default_month="CURRENT",
            )
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": closing_month,
                    "param2": reference_date,
                },
            )

        case ("RP", "COMPOSITE_CONVERSION_EXCLUDED"):
            return _make_payload(
                request_id,
                tool_name="test_tool",
                arguments={
                    "param1": _month_value(
                        parameters,
                        today=reference_today,
                        default_month="CURRENT",
                    ),
                    "param2": "",
                },
            )

        case _:
            return None


def _make_payload(
    request_id: str,
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """모든 도구가 공통으로 사용하는 GenOS MCP JSON-RPC 봉투를 만든다."""

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def _text(value: Any) -> str:
    """누락값은 None이 아니라 빈 문자열로 통일한다."""

    if value is None:
        return ""
    return str(value).strip()


def _month_value(
    parameters: dict[str, str],
    *,
    today: date,
    default_month: str,
) -> str:
    """마감작업년월을 추출값 또는 코드에 명시한 월 기본값으로 만든다."""

    extracted = _text(parameters.get("closing_year_month"))
    if extracted:
        return extracted
    if default_month == "CURRENT":
        return today.strftime("%Y%m")
    if default_month == "PREVIOUS":
        previous_month = today.replace(day=1) - timedelta(days=1)
        return previous_month.strftime("%Y%m")
    raise ValueError(f"지원하지 않는 월 기본값 코드입니다: {default_month}")


def _month_and_reference_date(
    parameters: dict[str, str],
    *,
    today: date,
    default_month: str,
) -> tuple[str, str]:
    """기준일자가 있으면 마감작업년월을 빈 문자열로 만드는 업무 규칙."""

    reference_date = _text(parameters.get("reference_date"))
    if reference_date:
        return "", reference_date
    return (
        _month_value(
            parameters,
            today=today,
            default_month=default_month,
        ),
        "",
    )


def _require_text(
    value: Any,
    *,
    input_code: str,
    parameter_name: str,
    label: str,
    message: str,
) -> str:
    """필수 문자열이 없으면 공통 HITL action 생성을 위한 예외를 발생시킨다."""

    normalized = _text(value)
    if normalized:
        return normalized
    raise McpParameterInputRequired(
        input_code=input_code,
        parameter_name=parameter_name,
        label=label,
        message=message,
    )
