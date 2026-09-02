"""함수형 MCP 결과 전처리·프런트 출력 어댑터.

활성 세부 시나리오는 중앙 columns 설정을 사용하지 않는다. 각 detail은
``app/mcp/scenarios/<agent>.py``의 output 함수에서 원본 결과를 자유롭게 파싱하고
전처리 데이터, 답변 본문, table/card/file renderable을 직접 만든다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.mcp.models import McpExecutionResult
from app.mcp.scenarios.contracts import ScenarioMcpOutputContext
from app.mcp.scenarios.registry import get_scenario_handler_spec
from app.mcp.workflow_handlers import get_workflow_result_formatter
from app.observability import logger, timed
from app.rag_policies import RAG_SCENARIO_POLICIES
from app.renderables import normalize_scenario_answer
from app.scenario_actions import redact_scenario_action_parameters
from app.subagents.models import SubagentResult


QUERY_RESULT_FORMAT = "query.v1"
RAG_RAW_RESULT_FORMAT = "raw.rag"
RAG_PASSTHROUGH_SCENARIOS = frozenset(RAG_SCENARIO_POLICIES)


class McpResultFormatError(ValueError):
    """세부 시나리오 결과 전처리 함수 계약이 맞지 않을 때 발생한다."""


@timed("MCP 조회 결과 정제")
def adapt_mcp_result(
    *,
    execution: McpExecutionResult,
    subagent: SubagentResult,
    employee_id: str,
    session_id: str,
    thread_id: str,
    request_context: Mapping[str, Any],
    workflow_results: Sequence[McpExecutionResult] = (),
) -> McpExecutionResult:
    """세부 시나리오의 함수형 output handler를 실행해 프런트 결과를 만든다."""

    if (
        execution.outcome != "SUCCESS"
        or not execution.succeeded
        or execution.result is None
    ):
        return execution

    agent_code = subagent.agent_code.upper()
    detail_code = subagent.detail_scenario_code
    handler_spec = get_scenario_handler_spec(agent_code, detail_code)

    if handler_spec is not None and handler_spec.output_handler is not None:
        safe_request_context = _build_output_request_context(
            request_context=request_context,
            employee_id=employee_id,
            session_id=session_id,
            thread_id=thread_id,
            subagent=subagent,
            execution=execution,
            workflow_results=workflow_results,
            output_handler_code=handler_spec.output_handler_code,
        )
        output_context = ScenarioMcpOutputContext(
            execution=execution,
            subagent=subagent,
            employee_id=employee_id,
            session_id=session_id,
            thread_id=thread_id,
            request_context=safe_request_context,
            workflow_results=tuple(workflow_results),
            workflow=safe_request_context.get("mcp_workflow", {}),
        )
        output = handler_spec.output_handler(output_context)
        answer = normalize_scenario_answer(
            output.answer,
            default_renderable_code=f"{agent_code}:{detail_code}:renderable",
        )
        if not answer.text.strip():
            raise McpResultFormatError(
                "시나리오 output handler가 비어 있는 답변을 반환했습니다: "
                f"agent_code={agent_code}, detail_scenario_code={detail_code}"
            )
        renderables = _serialize_renderables(
            answer.renderables,
            agent_code=agent_code,
            scenario_code=subagent.scenario_code,
            detail_code=detail_code,
        )
        output_code = (
            handler_spec.output_handler_code
            or f"{handler_spec.output_handler.__module__}:"
            f"{handler_spec.output_handler.__name__}"
        )
        formatted_result = {
            "format": output.result_format or QUERY_RESULT_FORMAT,
            "adapter_code": f"{agent_code}:{detail_code}:function",
            "result_formatter_code": output_code,
            "output_handler_code": output_code,
            "data": output.data,
            "parameters": redact_scenario_action_parameters(
                agent_code,
                detail_code,
                subagent.parameters,
            ),
            "request_context": safe_request_context,
            "answer_text": answer.text,
            "renderables": renderables,
            "metadata": dict(output.metadata),
        }
        logger.info(
            "======== MCP 함수형 결과 전처리 완료 | 에이전트=%s | "
            "세부시나리오=%s | outputHandler=%s | 전처리결과=%s | "
            "답변=%s | 확장데이터개수=%d",
            agent_code,
            detail_code,
            output_code,
            output.data,
            answer.text,
            len(renderables),
        )
        return execution.model_copy(
            update={
                "result_format": output.result_format or QUERY_RESULT_FORMAT,
                "formatted_result": formatted_result,
            }
        )

    # output handler가 없는 interaction detail의 legacy fallback이다. 새 detail은
    # registry에 output_handler를 연결하면 이 분기보다 우선한다.
    if (
        subagent.interaction is not None
        and subagent.interaction.tool.result_mode == "success_message"
    ):
        sensitive_parameters = {
            step.parameter_name for step in subagent.interaction.steps if step.sensitive
        }
        safe_parameters = {
            name: "<민감값 마스킹>" if name in sensitive_parameters else value
            for name, value in subagent.parameters.items()
        }
        safe_parameters = redact_scenario_action_parameters(
            agent_code,
            detail_code,
            safe_parameters,
        )
        return execution.model_copy(
            update={
                "result_format": QUERY_RESULT_FORMAT,
                "formatted_result": {
                    "format": QUERY_RESULT_FORMAT,
                    "adapter_code": f"{agent_code}:{detail_code}:interaction-fallback",
                    "data": {"raw_result": execution.result},
                    "parameters": safe_parameters,
                    "request_context": {},
                    "answer_text": subagent.interaction.tool.success_message,
                    "renderables": [],
                    "metadata": {},
                },
            }
        )

    if (agent_code, detail_code) in RAG_PASSTHROUGH_SCENARIOS:
        logger.info(
            "======== MCP 결과 전처리 생략 | 에이전트=%s | "
            "세부시나리오=%s | 결과형식=%s | 이유=RAG원본유지",
            agent_code,
            detail_code,
            RAG_RAW_RESULT_FORMAT,
        )
        return execution.model_copy(update={"result_format": RAG_RAW_RESULT_FORMAT})

    # explicit legacy manifest workflow만 기존 formatter를 유지한다. 활성
    # 함수형 detail은 registry output_handler를 사용하며 columns 표가 없다.
    if subagent.mcp_workflow is not None and subagent.mcp_workflow.result_formatter:
        formatter = get_workflow_result_formatter(subagent.mcp_workflow.result_formatter)
        safe_request_context = _build_output_request_context(
            request_context=request_context,
            employee_id=employee_id,
            session_id=session_id,
            thread_id=thread_id,
            subagent=subagent,
            execution=execution,
            workflow_results=workflow_results,
            output_handler_code=subagent.mcp_workflow.result_formatter,
        )
        answer = normalize_scenario_answer(
            formatter([], dict(subagent.parameters), safe_request_context),
            default_renderable_code=f"{agent_code}:{detail_code}:legacy",
        )
        return execution.model_copy(
            update={
                "result_format": QUERY_RESULT_FORMAT,
                "formatted_result": {
                    "format": QUERY_RESULT_FORMAT,
                    "adapter_code": f"{agent_code}:{detail_code}:legacy-workflow",
                    "result_formatter_code": subagent.mcp_workflow.result_formatter,
                    "data": execution.result,
                    "parameters": redact_scenario_action_parameters(
                        agent_code,
                        detail_code,
                        subagent.parameters,
                    ),
                    "request_context": safe_request_context,
                    "answer_text": answer.text,
                    "renderables": _serialize_renderables(
                        answer.renderables,
                        agent_code=agent_code,
                        scenario_code=subagent.scenario_code,
                        detail_code=detail_code,
                    ),
                    "metadata": {},
                },
            }
        )

    raise McpResultFormatError(
        "MCP 결과 output handler가 등록되지 않았습니다: "
        f"agent_code={agent_code}, detail_scenario_code={detail_code}. "
        "app/mcp/scenarios/registry.py에 output_handler를 연결하세요."
    )


def _serialize_renderables(
    renderables: Sequence[Any],
    *,
    agent_code: str,
    scenario_code: str,
    detail_code: str,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for renderable in renderables:
        renderable_metadata = {
            **renderable.metadata,
            "agent_code": agent_code,
            "scenario_code": scenario_code,
            "detail_scenario_code": detail_code,
        }
        renderable_code = renderable.code
        if ":" not in renderable_code:
            renderable_code = f"{agent_code}:{detail_code}:{renderable_code}"
        serialized.append(
            renderable.model_copy(
                update={
                    "code": renderable_code,
                    "metadata": renderable_metadata,
                }
            ).model_dump(mode="json")
        )
    return serialized


def _build_output_request_context(
    *,
    request_context: Mapping[str, Any],
    employee_id: str,
    session_id: str,
    thread_id: str,
    subagent: SubagentResult,
    execution: McpExecutionResult,
    workflow_results: Sequence[McpExecutionResult] = (),
    output_handler_code: str | None = None,
) -> dict[str, Any]:
    """출력 함수에 access token을 제외한 요청 정보와 전체 MCP 원장을 전달한다."""

    user = request_context.get("user")
    if not isinstance(user, Mapping):
        user = {}
    output_context: dict[str, Any] = {
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
    if subagent.mcp_workflow is not None or workflow_results:
        serialized_results = [
            _workflow_result_for_output(item) for item in workflow_results
        ]
        by_step: dict[str, list[dict[str, Any]]] = {}
        for item in serialized_results:
            by_step.setdefault(str(item["step_code"]), []).append(item)
        terminal_result = execution.result if isinstance(execution.result, Mapping) else {}
        output_context["mcp_workflow"] = {
            "output_handler": output_handler_code,
            "handler_code": execution.workflow_handler_code,
            "fanout": terminal_result.get("fanout"),
            "execution": terminal_result.get("execution"),
            "batches": terminal_result.get("batches", []),
            "results": serialized_results,
            "by_step": by_step,
        }
    return output_context


def _workflow_result_for_output(
    execution: McpExecutionResult,
) -> dict[str, Any]:
    """출력 함수가 다단계 원장을 안전하게 읽을 수 있도록 직렬화한다."""

    return {
        "step_code": execution.workflow_step_code,
        "step_index": execution.workflow_step_index,
        "step_count": execution.workflow_step_count,
        "tool_name": execution.tool_name,
        "request_id": execution.request_id,
        "arguments": execution.arguments,
        "succeeded": execution.succeeded,
        "outcome": execution.outcome,
        "business_code": execution.business_code,
        "result": execution.result,
        "error": execution.error,
        "execution_mode": execution.workflow_execution_mode,
        "item_index": execution.workflow_item_index,
        "item_count": execution.workflow_item_count,
        "source_step_code": execution.workflow_source_step_code,
        "is_aggregate": execution.workflow_is_aggregate,
        "input_mapper": execution.workflow_input_mapper_code,
        "handler_code": execution.workflow_handler_code,
    }
