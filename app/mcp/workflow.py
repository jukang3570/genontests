"""MCP for_each fan-out 결과를 formatter용 terminal 결과로 합친다."""

from collections.abc import Mapping, Sequence
from typing import Any

from app.mcp.models import (
    MCP_NO_DATA_MESSAGE,
    MCP_SAFE_ERROR_MESSAGE,
    McpExecutionResult,
)
from app.renderables import extract_data_items
from app.subagents.models import McpWorkflowStep


def aggregate_fanout_results(
    *,
    step: McpWorkflowStep,
    workflow_step_index: int,
    workflow_step_count: int,
    item_count: int,
    results: Sequence[McpExecutionResult],
    aggregate_request_id: str,
) -> McpExecutionResult:
    """개별 호출 원장과 data를 순서대로 합쳐 formatter용 결과를 만든다."""

    batches: list[dict[str, Any]] = []
    flattened_data: list[dict[str, Any]] = []
    success_count = 0
    no_data_count = 0
    error_count = 0
    disabled_count = 0

    for fallback_index, execution in enumerate(results):
        item_index = (
            execution.workflow_item_index
            if execution.workflow_item_index is not None
            else fallback_index
        )
        if (
            execution.succeeded
            and execution.outcome == "SUCCESS"
            and execution.backend != "disabled"
        ):
            success_count += 1
        elif execution.outcome == "NO_DATA":
            no_data_count += 1
        elif execution.backend == "disabled":
            disabled_count += 1
        else:
            error_count += 1

        batch = {
            "index": item_index,
            "toolName": execution.tool_name,
            "requestId": execution.request_id,
            "arguments": execution.arguments,
            "succeeded": execution.succeeded,
            "outcome": execution.outcome,
            "businessCode": execution.business_code,
            "result": execution.result,
            "error": execution.error,
        }
        batches.append(batch)

        if not (
            execution.succeeded
            and execution.outcome == "SUCCESS"
            and isinstance(execution.result, Mapping)
        ):
            continue
        try:
            data_items = extract_data_items(execution.result)
        except ValueError:
            # child raw 결과는 batches에 보존한다. 표준 data가 없으면 formatter의
            # 평탄화 data에 억지로 넣지 않는다.
            continue
        for item in data_items:
            flattened_data.append(
                {
                    **dict(item),
                    "_fanout": {
                        "index": item_index,
                        "requestId": execution.request_id,
                        "arguments": execution.arguments,
                    },
                }
            )

    summary = {
        "mode": step.execution.mode,
        "sourceStep": step.execution.source_step,
        "stepCode": step.code,
        "itemCount": item_count,
        "callCount": len(results),
        "successCount": success_count,
        "noDataCount": no_data_count,
        "errorCount": error_count,
        "disabledCount": disabled_count,
        "errorPolicy": step.execution.error_policy,
    }
    aggregate_result = {
        "data": flattened_data,
        "fanout": summary,
        "batches": batches,
    }

    common = {
        "backend": "fanout",
        "tool_name": step.tool.name,
        "request_id": aggregate_request_id,
        "arguments": {
            "source_step": step.execution.source_step,
            "item_count": item_count,
        },
        "business_code": None,
        "result": aggregate_result,
        "workflow_step_code": step.code,
        "workflow_step_index": workflow_step_index,
        "workflow_step_count": workflow_step_count,
        "workflow_is_final": workflow_step_index == workflow_step_count - 1,
        "workflow_execution_mode": step.execution.mode,
        "workflow_item_index": None,
        "workflow_item_count": item_count,
        "workflow_source_step_code": step.execution.source_step,
        "workflow_is_aggregate": True,
        "workflow_input_mapper_code": step.input_mapper,
    }

    if success_count:
        partial_error = None
        # 일부 1001(NO_DATA)은 기술 오류가 아니다. 성공 batch가 있으면 해당
        # 데이터만 포맷하고, 실제 ERROR/disabled가 있을 때만 error를 설정한다.
        if error_count or disabled_count:
            partial_error = (
                "fan-out 일부 호출 미완료: "
                f"success={success_count}, no_data={no_data_count}, "
                f"error={error_count}, disabled={disabled_count}"
            )
        return McpExecutionResult(
            **common,
            succeeded=True,
            outcome="SUCCESS",
            error=partial_error,
        )

    # fan-out의 모든 개별 호출이 1001(NO_DATA)이면 terminal 집계 결과도
    # business_code=1001로 승격한다. 이 값은 output 계층의 고정답변 분기에
    # 사용되고, 빈 테이블이 만들어지는 것을 막는다.
    if results and no_data_count == len(results):
        return McpExecutionResult(
            **{**common, "business_code": "1001"},
            succeeded=True,
            outcome="NO_DATA",
            user_message=MCP_NO_DATA_MESSAGE,
        )
    if item_count == 0:
        return McpExecutionResult(
            **common,
            succeeded=True,
            outcome="NO_DATA",
            user_message=MCP_NO_DATA_MESSAGE,
        )

    if disabled_count and not error_count:
        unavailable_message = next(
            (
                result.user_message
                for result in results
                if result.user_message
            ),
            step.tool.unavailable_message,
        )
        return McpExecutionResult(
            **{**common, "backend": "disabled"},
            succeeded=True,
            outcome="SUCCESS",
            user_message=unavailable_message,
        )

    return McpExecutionResult(
        **common,
        succeeded=False,
        outcome="ERROR",
        user_message=MCP_SAFE_ERROR_MESSAGE,
        error=(
            f"fan-out 호출이 모두 실패했습니다: error={error_count}, "
            f"item_count={item_count}"
        ),
    )
