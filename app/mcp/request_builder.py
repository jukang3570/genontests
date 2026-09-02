"""업무 코드와 분리된 MCP JSON-RPC 요청 조립 도구.

현재 활성 업무의 도구명, arguments, 호출 순서와 next-key 규칙은 모두
``app/mcp/scenarios/*.py``의 함수형 handler가 결정한다. 이 모듈은 함수가 넘긴
값을 공통 ``tools/call`` 요청으로 감싸고, 이전 선언형 interaction/workflow의
일반적인 argument source 변환만 호환한다. 에이전트·detail별 분기는 두지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from app.hitl import register_hitl_input_guardrail_policy, validate_input_value
from app.mcp.exceptions import McpParameterInputRequired
from app.mcp.models import McpExecutionResult
from app.subagents.models import McpWorkflowStep, SubagentResult


def get_mcp_workflow_step_count(subagent: SubagentResult) -> int:
    """detail의 실행 step 수를 반환한다.

    함수형 handler와 legacy interaction은 공통 executor 관점에서 한 번의 실행
    진입이므로 명시적인 workflow가 없을 때 1을 반환한다.
    """

    if subagent.mcp_workflow is None:
        return 1
    return len(subagent.mcp_workflow.steps)


def get_mcp_workflow_step(
    subagent: SubagentResult,
    step_index: int,
) -> McpWorkflowStep | None:
    """선언형 workflow step을 반환하고 미선언 detail은 ``None``을 반환한다."""

    if subagent.mcp_workflow is None:
        if step_index != 0:
            raise IndexError(
                f"기본 MCP workflow step index가 올바르지 않습니다: {step_index}"
            )
        return None
    try:
        return subagent.mcp_workflow.steps[step_index]
    except IndexError as exc:
        raise IndexError(
            "MCP workflow step index가 올바르지 않습니다: "
            f"detail={subagent.detail_scenario_code}, index={step_index}"
        ) from exc


def get_mcp_workflow_step_code(
    subagent: SubagentResult,
    step_index: int,
) -> str:
    """추적 ID와 tester에서 사용할 안정적인 workflow step code를 반환한다."""

    step = get_mcp_workflow_step(subagent, step_index)
    return step.code if step is not None else "FINAL"


def get_mcp_workflow_iteration_items(
    *,
    subagent: SubagentResult,
    step_index: int,
    previous_results: Sequence[McpExecutionResult],
) -> list[Any] | None:
    """legacy ``for_each`` step의 source collection을 반환한다."""

    step = get_mcp_workflow_step(subagent, step_index)
    if step is None or step.execution.mode == "single":
        return None

    source_code = str(step.execution.source_step)
    candidates = [
        result
        for result in previous_results
        if result.workflow_step_code == source_code
    ]
    if not candidates:
        raise ValueError(
            "for_each source MCP workflow 결과가 없습니다: "
            f"detail={subagent.detail_scenario_code}, step={step.code}, "
            f"source_step={source_code}"
        )
    source = next(
        (result for result in reversed(candidates) if result.workflow_is_aggregate),
        candidates[-1],
    )
    if not source.succeeded or source.outcome != "SUCCESS" or source.result is None:
        raise ValueError(
            "성공하지 않은 MCP workflow 결과는 for_each source로 사용할 수 없습니다: "
            f"step={source.workflow_step_code}, outcome={source.outcome}"
        )
    collection = _extract_mapping_path(
        source.result,
        step.execution.collection_path,
    )
    if not isinstance(collection, list):
        raise ValueError(
            "for_each collection_path 결과는 배열이어야 합니다: "
            f"step={step.code}, path={step.execution.collection_path}, "
            f"type={type(collection).__name__}"
        )
    if len(collection) > step.execution.max_items:
        raise ValueError(
            "for_each 항목 수가 max_items를 초과했습니다: "
            f"step={step.code}, count={len(collection)}, "
            f"max_items={step.execution.max_items}"
        )
    return list(collection)


def build_mcp_workflow_step_request(
    *,
    request_id: str,
    subagent: SubagentResult,
    employee_id: str,
    session_id: str,
    thread_id: str,
    request_context: Mapping[str, Any],
    step_index: int,
    previous_results: Sequence[McpExecutionResult] = (),
    current_item: Any = None,
    argument_overrides: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """함수형 또는 legacy workflow의 현재 step을 JSON-RPC 요청으로 만든다.

    함수형 handler는 ``scenario_runtime``이 생성한 메모리 내 mapped step과
    ``argument_overrides``를 전달한다. 명시적 legacy workflow는 각 argument의
    source를 공통 규칙으로 해석한다. workflow가 없는 legacy interaction은 기존
    입력 검증과 도구 호출을 계속 지원한다. 그 외 detail의 업무별 fallback은
    의도적으로 제공하지 않으며 반드시 scenario registry handler를 사용해야 한다.
    """

    step = get_mcp_workflow_step(subagent, step_index)
    reference_today = today or date.today()
    if step is None:
        if subagent.interaction is not None:
            return _build_interaction_request(
                request_id=request_id,
                subagent=subagent,
                employee_id=employee_id,
                session_id=session_id,
                thread_id=thread_id,
                request_context=request_context,
                today=reference_today,
            )
        raise ValueError(
            "MCP 함수형 handler 또는 명시적 workflow가 필요합니다: "
            f"agent={subagent.agent_code}, detail={subagent.detail_scenario_code}. "
            "app/mcp/scenarios/registry.py 등록을 확인하세요."
        )

    if subagent.interaction is not None:
        _validate_interaction_parameters(subagent)

    if argument_overrides is not None:
        if step.execution.mode != "mapped":
            raise ValueError("argument_overrides는 mapped step에서만 사용할 수 있습니다.")
        return build_mcp_tool_call_request(
            request_id=request_id,
            tool_name=step.tool.name,
            arguments={str(key): value for key, value in argument_overrides.items()},
        )

    runtime = {
        "employee_id": employee_id,
        "session_id": session_id,
        "thread_id": thread_id,
    }
    date_values = {
        "current_date": reference_today.strftime("%Y%m%d"),
        "current_year": str(reference_today.year),
        "current_year_month": reference_today.strftime("%Y%m"),
        "previous_year": str(reference_today.year - 1),
    }
    result_by_step = {
        result.workflow_step_code: result for result in previous_results
    }
    arguments: dict[str, Any] = {}
    for name, definition in step.tool.arguments.items():
        if definition.source == "parameter":
            value = subagent.parameters.get(str(definition.key))
        elif definition.source == "runtime":
            value = runtime.get(str(definition.key))
        elif definition.source == "request_context":
            value = request_context.get(str(definition.key))
        elif definition.source == "date":
            value = date_values.get(str(definition.key))
        elif definition.source == "step_result":
            previous = result_by_step.get(str(definition.step_code))
            value = _extract_workflow_result_value(
                previous,
                obj_id=str(definition.key),
            )
        elif definition.source == "current_item":
            value = _extract_current_item_value(
                current_item,
                key=str(definition.key),
            )
        else:
            value = definition.value

        value = _transform_argument(value, definition.transform)
        if definition.required and (value is None or value == ""):
            raise ValueError(
                "필수 MCP workflow argument 값이 없습니다: "
                f"detail={subagent.detail_scenario_code}, step={step.code}, "
                f"argument={name}"
            )
        arguments[name] = value

    return build_mcp_tool_call_request(
        request_id=request_id,
        tool_name=step.tool.name,
        arguments=arguments,
    )


def build_mcp_tool_call_request(
    *,
    request_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """모든 MCP 도구가 공유하는 JSON-RPC ``tools/call`` 요청을 만든다."""

    normalized_tool = str(tool_name).strip()
    if not normalized_tool:
        raise ValueError("MCP tool_name은 비어 있을 수 없습니다.")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": normalized_tool,
            "arguments": {str(key): value for key, value in arguments.items()},
        },
    }


def _extract_mapping_path(source: Any, path: str) -> Any:
    """dot path로 object/list 값을 읽고 배열 index도 지원한다."""

    current = source
    for segment in (part.strip() for part in path.split(".")):
        if not segment:
            raise ValueError(f"비어 있는 collection path segment입니다: {path}")
        if isinstance(current, Mapping):
            if segment not in current:
                raise ValueError(f"collection path를 찾지 못했습니다: {path}")
            current = current[segment]
            continue
        if isinstance(current, list) and segment.isdigit():
            index = int(segment)
            try:
                current = current[index]
            except IndexError as exc:
                raise ValueError(f"collection path index가 없습니다: {path}") from exc
            continue
        raise ValueError(f"collection path를 따라갈 수 없습니다: {path}")
    return current


def _extract_current_item_value(item: Any, *, key: str) -> Any:
    """fan-out 현재 항목에서 하나의 argument 값을 추출한다."""

    normalized_key = key.strip()
    if normalized_key in {"$", "."}:
        return item
    if not normalized_key:
        raise ValueError("current_item source에는 key가 필요합니다.")

    if isinstance(item, Mapping):
        try:
            return _extract_mapping_path(item, normalized_key)
        except ValueError:
            pass
        if str(item.get("objId", "")).strip() == normalized_key:
            return item.get("objVal")
        for container_name in ("fields", "values", "data", "items"):
            found, value = _find_obj_value(item.get(container_name), normalized_key)
            if found:
                return value
    elif isinstance(item, list):
        found, value = _find_obj_value(item, normalized_key)
        if found:
            return value

    raise ValueError(
        "현재 fan-out 항목에서 필요한 값을 찾지 못했습니다: "
        f"key={normalized_key}, item_type={type(item).__name__}"
    )


def _find_obj_value(source: Any, obj_id: str) -> tuple[bool, Any]:
    if not isinstance(source, list):
        return False, None
    for item in source:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("objId", "")).strip() == obj_id:
            return True, item.get("objVal")
    return False, None


def _extract_workflow_result_value(
    execution: McpExecutionResult | None,
    *,
    obj_id: str,
) -> Any:
    """앞선 step의 normalized structuredContent.data에서 objVal을 추출한다."""

    if execution is None:
        raise ValueError(f"참조할 이전 MCP workflow 결과가 없습니다: objId={obj_id}")
    if not execution.succeeded or execution.outcome != "SUCCESS":
        raise ValueError(
            "실패 또는 NO_DATA인 MCP workflow 결과는 참조할 수 없습니다: "
            f"step={execution.workflow_step_code}, outcome={execution.outcome}"
        )
    result = execution.result
    data = result.get("data") if isinstance(result, Mapping) else None
    if not isinstance(data, list):
        raise ValueError(
            "이전 MCP workflow 결과에 data 배열이 없습니다: "
            f"step={execution.workflow_step_code}"
        )
    for item in data:
        if not isinstance(item, Mapping) or str(item.get("objId", "")) != obj_id:
            continue
        value = item.get("objVal")
        if value is not None and str(value).strip():
            return value
    raise ValueError(
        "이전 MCP workflow 결과에서 필요한 objId 값을 찾지 못했습니다: "
        f"step={execution.workflow_step_code}, objId={obj_id}"
    )


def _build_interaction_request(
    *,
    request_id: str,
    subagent: SubagentResult,
    employee_id: str,
    session_id: str,
    thread_id: str,
    request_context: Mapping[str, Any],
    today: date,
) -> dict[str, Any]:
    """legacy manifest interaction을 검증하고 최종 MCP 요청을 만든다."""

    interaction = subagent.interaction
    if interaction is None:
        raise ValueError("상호작용 요청에 interaction 설정이 없습니다.")
    _validate_interaction_parameters(subagent)

    runtime = {
        "employee_id": employee_id,
        "session_id": session_id,
        "thread_id": thread_id,
    }
    date_values = {
        "current_date": today.strftime("%Y%m%d"),
        "current_year": str(today.year),
        "current_year_month": today.strftime("%Y%m"),
        "previous_year": str(today.year - 1),
    }
    arguments: dict[str, Any] = {}
    for name, definition in interaction.tool.arguments.items():
        if definition.source == "parameter":
            value = subagent.parameters.get(str(definition.key))
        elif definition.source == "runtime":
            value = runtime.get(str(definition.key))
        elif definition.source == "request_context":
            value = request_context.get(str(definition.key))
        elif definition.source == "date":
            value = date_values.get(str(definition.key))
        else:
            value = definition.value

        value = _transform_argument(value, definition.transform)
        if definition.required and (value is None or value == ""):
            raise ValueError(
                "필수 interaction MCP argument 값이 없습니다: "
                f"detail={subagent.detail_scenario_code}, argument={name}"
            )
        arguments[name] = value

    return build_mcp_tool_call_request(
        request_id=request_id,
        tool_name=interaction.tool.name,
        arguments=arguments,
    )


def _validate_interaction_parameters(subagent: SubagentResult) -> None:
    """legacy interaction의 사용자 입력 단계를 순서대로 검증한다."""

    interaction = subagent.interaction
    if interaction is None:
        return
    for step in interaction.steps:
        raw_value = subagent.parameters.get(step.parameter_name)
        normalized = _text(raw_value)
        initial_error: str | None = None
        if normalized:
            _, errors = validate_input_value(
                {step.input_code: normalized},
                input_code=step.input_code,
                expected_value=step.expected_value,
                pattern=step.pattern,
                min_length=step.min_length,
                max_length=step.max_length,
                allowed_values=step.allowed_values,
                validation_message=step.validation_message,
            )
            initial_error = errors.get(step.input_code)
        if not normalized or initial_error is not None:
            # legacy interaction도 현재 프로세스에서 동일한 input code 정책을
            # 확실히 등록한다. 이 값은 Redis에 저장하지 않는다.
            register_hitl_input_guardrail_policy(
                step.input_code,
                enabled=step.guardrail_enabled,
            )
            raise McpParameterInputRequired(
                input_code=step.input_code,
                parameter_name=step.parameter_name,
                label=step.label,
                message=step.message,
                input_type=step.input_type,
                expected_value=step.expected_value,
                pattern=step.pattern,
                min_length=step.min_length,
                max_length=step.max_length,
                allowed_values=step.allowed_values,
                validation_message=step.validation_message,
                sensitive=step.sensitive,
                initial_error=initial_error,
            )


def _transform_argument(value: Any, transform: str) -> Any:
    """legacy manifest의 간단한 argument 변환 규칙을 적용한다."""

    if transform == "none":
        return value
    normalized = _text(value)
    if transform == "upper":
        return normalized.upper()
    if transform == "digits_only":
        return "".join(character for character in normalized if character.isdigit())
    return normalized


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()

