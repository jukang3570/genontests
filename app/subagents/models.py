"""서브에이전트가 공통으로 사용하는 결과와 상호작용 모델."""

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class InteractionInputStep(BaseModel):
    """MCP 호출 전에 프론트에서 순서대로 받아야 하는 입력 한 단계."""

    model_config = ConfigDict(extra="forbid")

    parameter_name: str = Field(min_length=1)
    input_code: str = Field(min_length=1)
    label: str = Field(min_length=1)
    message: str = Field(min_length=1)
    input_type: str = Field(default="text", min_length=1)
    expected_value: str | None = None
    pattern: str | None = None
    min_length: int | None = Field(default=None, ge=1)
    max_length: int | None = Field(default=None, ge=1)
    allowed_values: list[str] | None = None
    validation_message: str | None = None
    sensitive: bool = False
    # False이면 이 HITL 입력은 humanInput에서 원문을 유지한다. 조회키처럼
    # 마스킹으로 값이 바뀌면 안 되는 필드에만 사용한다.
    guardrail_enabled: bool = True

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str | None) -> str | None:
        """잘못된 정규식은 사용자 입력 시점이 아니라 시작 시점에 거절한다."""

        if value is not None:
            re.compile(value)
        return value

    @model_validator(mode="after")
    def validate_length_and_values(self) -> "InteractionInputStep":
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length는 max_length보다 클 수 없습니다.")
        if self.allowed_values is not None:
            normalized = [value.strip() for value in self.allowed_values]
            if any(not value for value in normalized):
                raise ValueError("allowed_values에는 빈 문자열을 사용할 수 없습니다.")
            if len(normalized) != len(set(normalized)):
                raise ValueError("allowed_values에는 중복 값을 사용할 수 없습니다.")
            self.allowed_values = normalized
        return self


class InteractionToolArgument(BaseModel):
    """manifest 값에서 MCP argument 하나를 만드는 선언."""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "parameter",
        "runtime",
        "request_context",
        "date",
        "literal",
    ]
    key: str | None = None
    value: Any = None
    required: bool = False
    transform: Literal["none", "strip", "upper", "digits_only"] = "strip"

    @model_validator(mode="after")
    def validate_source(self) -> "InteractionToolArgument":
        if self.source != "literal" and not (self.key or "").strip():
            raise ValueError(f"{self.source} source에는 key가 필요합니다.")
        return self


class InteractionTool(BaseModel):
    """모든 입력 완료 후 실행할 MCP 도구와 답변 정책."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    name: str = Field(min_length=1)
    unavailable_message: str = Field(
        default="요청한 후속 작업 도구가 아직 연결되지 않았습니다.",
        min_length=1,
    )
    success_message: str = Field(
        default="요청한 작업을 완료했습니다.",
        min_length=1,
    )
    result_mode: Literal["success_message", "scenario_adapter"] = "success_message"
    arguments: dict[str, InteractionToolArgument] = Field(default_factory=dict)


class ScenarioInteraction(BaseModel):
    """한 detail의 다단계 action과 마지막 MCP 호출 선언."""

    model_config = ConfigDict(extra="forbid")

    steps: list[InteractionInputStep] = Field(min_length=1)
    tool: InteractionTool

    @model_validator(mode="after")
    def validate_unique_steps(self) -> "ScenarioInteraction":
        parameter_names = [step.parameter_name for step in self.steps]
        input_codes = [step.input_code for step in self.steps]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("interaction.steps의 parameter_name은 중복될 수 없습니다.")
        if len(input_codes) != len(set(input_codes)):
            raise ValueError("interaction.steps의 input_code는 중복될 수 없습니다.")
        return self


class McpWorkflowArgument(BaseModel):
    """순차 MCP step의 argument 하나를 만드는 선언."""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "parameter",
        "runtime",
        "request_context",
        "date",
        "literal",
        "step_result",
        "current_item",
    ]
    key: str | None = None
    value: Any = None
    step_code: str | None = None
    required: bool = False
    transform: Literal["none", "strip", "upper", "digits_only"] = "strip"

    @model_validator(mode="after")
    def validate_source(self) -> "McpWorkflowArgument":
        if self.source == "literal":
            if self.key is not None:
                raise ValueError("key는 literal source에서 사용할 수 없습니다.")
            if self.step_code is not None:
                raise ValueError("step_code는 step_result source에서만 사용할 수 있습니다.")
            return self
        if not (self.key or "").strip():
            raise ValueError(f"{self.source} source에는 key가 필요합니다.")
        if self.source == "step_result" and not (self.step_code or "").strip():
            raise ValueError("step_result source에는 step_code가 필요합니다.")
        if self.source != "step_result" and self.step_code is not None:
            raise ValueError("step_code는 step_result source에서만 사용할 수 있습니다.")
        return self


class McpWorkflowTool(BaseModel):
    """workflow step에서 호출할 MCP 도구와 argument 선언."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    name: str = Field(min_length=1)
    unavailable_message: str = Field(
        default="요청한 MCP workflow 도구가 아직 연결되지 않았습니다.",
        min_length=1,
    )
    arguments: dict[str, McpWorkflowArgument] = Field(default_factory=dict)


class McpWorkflowExecution(BaseModel):
    """workflow step의 단일·목록 fan-out·Python mapper 실행 정책."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "for_each", "mapped"] = "single"
    source_step: str | None = None
    collection_path: str = Field(default="data", min_length=1)
    error_policy: Literal["fail_fast", "continue"] = "fail_fast"
    max_items: int = Field(default=100, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_mode(self) -> "McpWorkflowExecution":
        if self.mode == "for_each" and not (self.source_step or "").strip():
            raise ValueError("for_each execution에는 source_step이 필요합니다.")
        if self.mode != "for_each" and self.source_step is not None:
            raise ValueError("source_step은 for_each execution에서만 사용합니다.")
        self.collection_path = self.collection_path.strip()
        return self


class McpWorkflowStep(BaseModel):
    """하나의 세부 시나리오 안에서 순서대로 실행할 MCP 한 단계."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    execution: McpWorkflowExecution = Field(default_factory=McpWorkflowExecution)
    # 복잡한 결과 변환은 YAML 안에 별도 DSL로 작성하지 않고, 등록된 Python
    # mapper code를 선택한다. mapper는 이전 모든 step 결과를 받아 0~N개의
    # 현재 tool arguments를 반환한다.
    input_mapper: str | None = None
    tool: McpWorkflowTool

    @model_validator(mode="after")
    def validate_input_mapper(self) -> "McpWorkflowStep":
        mapper_code = (self.input_mapper or "").strip()
        if self.execution.mode == "mapped" and not mapper_code:
            raise ValueError("mapped execution에는 input_mapper가 필요합니다.")
        if self.execution.mode != "mapped" and mapper_code:
            raise ValueError("input_mapper는 mapped execution에서만 사용합니다.")
        if mapper_code and self.tool.arguments:
            raise ValueError(
                "mapped step의 arguments는 input_mapper가 전부 생성하므로 "
                "tool.arguments를 함께 선언할 수 없습니다."
            )
        self.input_mapper = mapper_code or None
        return self


class ScenarioMcpWorkflow(BaseModel):
    """한 세부 시나리오의 1~N개 순차 MCP 호출 선언."""

    model_config = ConfigDict(extra="forbid")

    steps: list[McpWorkflowStep] = Field(min_length=1)
    final_step: str | None = None
    # 최종 화면 표현도 도구명이 아니라 세부 시나리오 workflow에 연결한다.
    # 값은 app/mcp/workflow_handlers.py에 등록된 안전한 handler code다.
    result_formatter: str | None = None

    @model_validator(mode="after")
    def validate_linear_workflow(self) -> "ScenarioMcpWorkflow":
        step_codes = [step.code for step in self.steps]
        if len(step_codes) != len(set(step_codes)):
            raise ValueError("mcp_workflow.steps의 code는 중복될 수 없습니다.")
        final_step = self.final_step or step_codes[-1]
        if final_step != step_codes[-1]:
            raise ValueError("mcp_workflow.final_step은 마지막 step이어야 합니다.")
        self.result_formatter = (self.result_formatter or "").strip() or None
        seen: set[str] = set()
        for step in self.steps:
            if (
                step.execution.mode == "for_each"
                and step.execution.source_step not in seen
            ):
                raise ValueError(
                    "for_each source_step은 현재 step보다 앞선 step이어야 합니다: "
                    f"step={step.code}, source={step.execution.source_step}"
                )
            for argument in step.tool.arguments.values():
                if (
                    argument.source == "current_item"
                    and step.execution.mode != "for_each"
                ):
                    raise ValueError(
                        "current_item source는 for_each step에서만 사용할 수 있습니다: "
                        f"step={step.code}"
                    )
                if argument.source != "step_result":
                    continue
                if argument.step_code not in seen:
                    raise ValueError(
                        "step_result는 현재 step보다 앞선 step만 참조할 수 있습니다: "
                        f"step={step.code}, reference={argument.step_code}"
                    )
            seen.add(step.code)
        self.final_step = final_step
        return self


class SubagentScenarioMatch(BaseModel):
    """한 질문에서 선택된 하나의 시나리오·세부 시나리오 결과."""

    model_config = ConfigDict(extra="forbid")

    scenario_code: str
    scenario_name: str
    detail_scenario_code: str
    detail_scenario_name: str
    # 일반 조회값은 문자열, RAG keywords는 문자열 배열을 사용한다.
    parameters: dict[str, Any]
    interaction: ScenarioInteraction | None = None
    mcp_workflow: ScenarioMcpWorkflow | None = None


class SubagentResult(BaseModel):
    """한 서브에이전트 안에서 선택된 하나 이상의 시나리오 결과."""

    model_config = ConfigDict(extra="forbid")

    agent_code: str
    prompt_version: str
    scenario_code: str
    scenario_name: str
    detail_scenario_code: str
    detail_scenario_name: str
    parameters: dict[str, Any]
    interaction: ScenarioInteraction | None = None
    mcp_workflow: ScenarioMcpWorkflow | None = None
    # 다중 시나리오의 정식 계약이다. 기존 단일 필드는 첫 번째 결과를 나타내므로
    # 기존 프론트와 API 소비자도 그대로 사용할 수 있다.
    matches: list[SubagentScenarioMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_primary_match(self) -> "SubagentResult":
        """matches가 없으면 기존 단일 필드를 첫 매칭으로 자동 변환한다."""

        if not self.matches:
            self.matches = [
                SubagentScenarioMatch(
                    scenario_code=self.scenario_code,
                    scenario_name=self.scenario_name,
                    detail_scenario_code=self.detail_scenario_code,
                    detail_scenario_name=self.detail_scenario_name,
                    parameters=self.parameters,
                    interaction=self.interaction,
                    mcp_workflow=self.mcp_workflow,
                )
            ]
        primary = self.matches[0]
        self.interaction = primary.interaction
        self.mcp_workflow = primary.mcp_workflow
        return self


def redact_interaction_parameters(
    parameters: dict[str, Any],
    interaction: ScenarioInteraction | None,
) -> dict[str, Any]:
    """로그·추적용으로 interaction의 민감 입력값을 마스킹한다."""

    redacted = dict(parameters)
    if interaction is None:
        return redacted
    for step in interaction.steps:
        if step.sensitive and redacted.get(step.parameter_name):
            redacted[step.parameter_name] = "***MASKED***"
    return redacted


def subagent_result_for_log(result: SubagentResult) -> dict[str, object]:
    """SubagentResult를 JSON 로그에 안전한 dict로 변환한다."""

    # 지연 import로 subagent 모델과 시나리오 모듈의 import cycle을 피한다.
    from app.scenario_actions import redact_scenario_action_parameters

    payload = result.model_dump(mode="json")
    payload["parameters"] = redact_scenario_action_parameters(
        result.agent_code,
        result.detail_scenario_code,
        redact_interaction_parameters(
            result.parameters,
            result.interaction,
        ),
    )
    for index, match in enumerate(result.matches):
        payload["matches"][index]["parameters"] = (
            redact_scenario_action_parameters(
                result.agent_code,
                match.detail_scenario_code,
                redact_interaction_parameters(
                    match.parameters,
                    match.interaction,
                ),
            )
        )
    return payload
