"""legacy YAML input mapper와 전체 결과 formatter registry.

신규 MCP 호출 흐름은 app/mcp/scenarios의 async handler와 scenario_runtime을
사용한다. 이 파일의 input mapper는 이전 선언형 workflow 호환용이며, 복합 결과를
답변/표로 바꾸는 RESULT_FORMATTER_REGISTRY는 함수형 handler에서도 재사용한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from app.mcp.models import McpExecutionResult
from app.renderables import ScenarioAnswer, create_table_renderable, extract_data_items
from app.subagents.models import McpWorkflowStep, SubagentResult


class WorkflowHandlerError(ValueError):
    """등록되지 않았거나 잘못된 workflow handler 결과를 나타낸다."""


@dataclass(frozen=True)
class WorkflowInputMapperContext:
    """input mapper가 다음 MCP arguments를 만들 때 사용하는 전체 문맥."""

    subagent: SubagentResult
    step: McpWorkflowStep
    step_index: int
    step_count: int
    previous_results: tuple[McpExecutionResult, ...]
    employee_id: str
    session_id: str
    thread_id: str
    request_context: Mapping[str, Any]

    def results_for(self, step_code: str) -> list[McpExecutionResult]:
        """같은 step의 개별 호출과 합성 결과를 실행 순서대로 반환한다."""

        normalized = str(step_code).strip()
        return [
            result
            for result in self.previous_results
            if result.workflow_step_code == normalized
        ]

    def terminal_result_for(self, step_code: str) -> McpExecutionResult:
        """앞 step의 fan-in 합성 결과 또는 마지막 단일 결과를 반환한다."""

        candidates = self.results_for(step_code)
        if not candidates:
            raise WorkflowHandlerError(
                f"input mapper가 참조한 이전 step 결과가 없습니다: {step_code}"
            )
        for result in reversed(candidates):
            if result.workflow_is_aggregate:
                return result
        return candidates[-1]


WorkflowInputMapper = Callable[
    [WorkflowInputMapperContext],
    Sequence[Mapping[str, Any]],
]
WorkflowResultFormatter = Callable[
    [list[dict[str, Any]], dict[str, Any], dict[str, Any]],
    Any,
]


def _composite_excluded_seed_v1(
    context: WorkflowInputMapperContext,
) -> Sequence[Mapping[str, Any]]:
    """예제 1단계: 사용자 월과 runtime 사원번호로 test_tool 1회를 만든다."""

    closing_year_month = str(
        context.subagent.parameters.get("closing_year_month") or ""
    ).strip()
    return [
        {
            "param1": closing_year_month or date.today().strftime("%Y%m"),
            "param2": context.employee_id,
        }
    ]


def _composite_excluded_details_v1(
    context: WorkflowInputMapperContext,
) -> Sequence[Mapping[str, Any]]:
    """예제 2단계: 첫 결과의 모든 objId/objVal을 각각 다음 호출로 변환한다."""

    source = context.terminal_result_for("EXCLUDED_TARGETS")
    arguments: list[dict[str, Any]] = []
    for item in extract_data_items(source.result or {}):
        obj_id = str(item.get("objId", "")).strip()
        # mock test_tool에는 adapter 필터 검증용 unusedColumn도 포함되므로 이
        # 시나리오가 실제로 사용하는 두 업무 필드만 다음 단계 호출로 보낸다.
        if obj_id not in {"column1", "column2"}:
            continue
        arguments.append(
            {
                "param1": item.get("objVal"),
                "param2": f"{obj_id}:상세조회",
            }
        )
    return arguments


def _composite_excluded_final_v1(
    context: WorkflowInputMapperContext,
) -> Sequence[Mapping[str, Any]]:
    """예제 3단계: 2단계 호출별 원본 결과를 조합해 최종 호출들을 만든다."""

    source = context.terminal_result_for("EXCLUDED_DETAILS")
    source_result = source.result if isinstance(source.result, Mapping) else {}
    batches = source_result.get("batches", [])
    if not isinstance(batches, list):
        raise WorkflowHandlerError("EXCLUDED_DETAILS fan-in batches가 배열이 아닙니다.")

    arguments: list[dict[str, Any]] = []
    for batch in batches:
        if not isinstance(batch, Mapping) or not batch.get("succeeded"):
            continue
        raw_result = batch.get("result")
        values = {
            str(item.get("objId", "")): item.get("objVal")
            for item in extract_data_items(raw_result)
            if str(item.get("objId", "")).strip()
        }
        arguments.append(
            {
                "param1": values.get("column1", ""),
                "param2": f"{values.get('column2', '')}:최종가공",
            }
        )
    return arguments


def _performance_fee_composite_excluded_answer_v1(
    data: list[dict[str, Any]],
    parameters: dict[str, Any],
    request_context: dict[str, Any],
) -> ScenarioAnswer:
    """예제 최종 formatter: 세 단계 원장과 terminal batches를 함께 사용한다."""

    workflow = request_context.get("mcp_workflow", {})
    by_step = workflow.get("by_step", {}) if isinstance(workflow, Mapping) else {}
    final_results = (
        by_step.get("EXCLUDED_FINAL", []) if isinstance(by_step, Mapping) else []
    )
    aggregate = next(
        (
            item
            for item in reversed(final_results)
            if isinstance(item, Mapping) and item.get("is_aggregate")
        ),
        {},
    )
    aggregate_result = (
        aggregate.get("result", {}) if isinstance(aggregate, Mapping) else {}
    )
    batches = (
        aggregate_result.get("batches", [])
        if isinstance(aggregate_result, Mapping)
        else []
    )

    rows: list[tuple[Any, ...]] = []
    for index, batch in enumerate(batches if isinstance(batches, list) else []):
        if not isinstance(batch, Mapping) or not batch.get("succeeded"):
            continue
        values = {
            str(item.get("objId", "")): item.get("objVal")
            for item in extract_data_items(batch.get("result"))
            if str(item.get("objId", "")).strip()
        }
        rows.append(
            (
                index + 1,
                values.get("column1", ""),
                values.get("column2", ""),
            )
        )

    # 실제 도구 연결 전에도 formatter가 받은 terminal data를 확인할 수 있도록
    # batches가 없을 때는 평탄화 data의 모양을 진단용 한 행으로 보존한다.
    if not rows and data:
        rows.append((1, "terminal data", f"{len(data)}개 항목"))

    step_counts = {
        code: len(
            [
                item
                for item in items
                if isinstance(item, Mapping) and not item.get("is_aggregate")
            ]
        )
        for code, items in (by_step.items() if isinstance(by_step, Mapping) else [])
        if isinstance(items, list)
    }
    month = parameters.get("closing_year_month") or "현재월"
    text = (
        "[환산 미반영 내역 3단계 조회 예제]\n"
        f"- 기준월: {month}\n"
        f"- 1단계 호출: {step_counts.get('EXCLUDED_TARGETS', 0)}회\n"
        f"- 2단계 호출: {step_counts.get('EXCLUDED_DETAILS', 0)}회\n"
        f"- 3단계 호출: {step_counts.get('EXCLUDED_FINAL', 0)}회\n"
        f"- 최종 결과: {len(rows)}건"
    )
    return ScenarioAnswer(
        text=text,
        renderables=[
            create_table_renderable(
                code="result-table",
                title="환산 미반영 내역 3단계 조회 예제",
                format="markdown",
                columns=("순번", "미반영 대상 값", "처리 결과"),
                rows=rows,
                metadata={
                    "example": "test_tool -> test_tool -> test_tool",
                    "handler": (
                        "performance_fee.composite_excluded.answer_v1"
                    ),
                },
            )
        ],
    )


INPUT_MAPPER_REGISTRY: dict[str, WorkflowInputMapper] = {
    "performance_fee.composite_excluded.seed_v1": _composite_excluded_seed_v1,
    "performance_fee.composite_excluded.details_v1": (
        _composite_excluded_details_v1
    ),
    "performance_fee.composite_excluded.final_v1": _composite_excluded_final_v1,
}

RESULT_FORMATTER_REGISTRY: dict[str, WorkflowResultFormatter] = {
    "performance_fee.composite_excluded.answer_v1": (
        _performance_fee_composite_excluded_answer_v1
    ),
}


def build_mapped_step_arguments(
    *,
    subagent: SubagentResult,
    step: McpWorkflowStep,
    step_index: int,
    step_count: int,
    previous_results: Sequence[McpExecutionResult],
    employee_id: str,
    session_id: str,
    thread_id: str,
    request_context: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """mapped step이면 등록 mapper를 실행하고 검증된 arguments 배열을 반환한다."""

    if step.execution.mode != "mapped":
        return None
    mapper_code = str(step.input_mapper or "").strip()
    mapper = INPUT_MAPPER_REGISTRY.get(mapper_code)
    if mapper is None:
        raise WorkflowHandlerError(
            f"등록되지 않은 MCP input_mapper입니다: {mapper_code}"
        )
    raw_invocations = mapper(
        WorkflowInputMapperContext(
            subagent=subagent,
            step=step,
            step_index=step_index,
            step_count=step_count,
            previous_results=tuple(previous_results),
            employee_id=employee_id,
            session_id=session_id,
            thread_id=thread_id,
            request_context=request_context,
        )
    )
    if isinstance(raw_invocations, (str, bytes)) or not isinstance(
        raw_invocations, Sequence
    ):
        raise WorkflowHandlerError(
            f"input_mapper는 arguments object 배열을 반환해야 합니다: {mapper_code}"
        )
    if len(raw_invocations) > step.execution.max_items:
        raise WorkflowHandlerError(
            "input_mapper 호출 수가 max_items를 초과했습니다: "
            f"mapper={mapper_code}, count={len(raw_invocations)}, "
            f"max_items={step.execution.max_items}"
        )

    invocations: list[dict[str, Any]] = []
    for index, arguments in enumerate(raw_invocations):
        if not isinstance(arguments, Mapping):
            raise WorkflowHandlerError(
                "input_mapper 각 항목은 MCP arguments object여야 합니다: "
                f"mapper={mapper_code}, index={index}"
            )
        normalized = {str(key): value for key, value in arguments.items()}
        if any(not key.strip() for key in normalized):
            raise WorkflowHandlerError(
                "input_mapper arguments에는 빈 key를 사용할 수 없습니다: "
                f"mapper={mapper_code}, index={index}"
            )
        invocations.append(normalized)
    return invocations


def get_workflow_result_formatter(code: str) -> WorkflowResultFormatter:
    """manifest result_formatter code를 등록된 함수로 해석한다."""

    normalized = str(code).strip()
    formatter = RESULT_FORMATTER_REGISTRY.get(normalized)
    if formatter is None:
        raise WorkflowHandlerError(
            f"등록되지 않은 MCP result_formatter입니다: {normalized}"
        )
    return formatter
