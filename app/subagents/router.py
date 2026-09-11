"""manifest 기반 Structured Output 시나리오 서브에이전트 실행기."""

from datetime import date
from typing import Annotated, Any, Literal, Protocol, Union

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.config import Settings
from app.observability import (
    async_timed_block,
    log_failure_diagnostic,
    logger,
    timed,
)
from app.subagents.models import (
    ScenarioInteraction,
    ScenarioMcpWorkflow,
    SubagentResult,
    SubagentScenarioMatch,
)
from app.subagents.prompt_loader import (
    ScenarioPromptBundle,
    SubagentPromptLoader,
)


class SubagentRouter(Protocol):
    """마스터 그래프가 구체적인 서브에이전트 구현과 무관하게 쓰는 계약."""

    def supports(self, agent_code: str) -> bool: ...

    def registered_codes(self) -> tuple[str, ...]: ...

    async def classify(
        self,
        *,
        agent_code: str,
        query: str,
    ) -> SubagentResult | None: ...

    async def classify_for_detail(
        self,
        *,
        agent_code: str,
        query: str,
        detail_scenario_code: str,
    ) -> SubagentResult | None: ...


class EmptySubagentRouter:
    """아직 구현되지 않은 에이전트와 단위 테스트를 위한 빈 라우터."""

    def supports(self, agent_code: str) -> bool:
        return False

    def registered_codes(self) -> tuple[str, ...]:
        return ()

    async def classify(
        self,
        *,
        agent_code: str,
        query: str,
    ) -> SubagentResult | None:
        return None

    async def classify_for_detail(
        self,
        *,
        agent_code: str,
        query: str,
        detail_scenario_code: str,
    ) -> SubagentResult | None:
        return None


class ScenarioSubagent:
    """한 manifest의 시나리오·세부 시나리오·파라미터를 분류한다."""

    @timed("시나리오 서브에이전트 초기화")
    def __init__(
        self,
        settings: Settings,
        bundle: ScenarioPromptBundle,
    ) -> None:
        if not settings.genos_bearer_token:
            raise ValueError("GENOS_BEARER_TOKEN 환경변수가 필요합니다.")

        self._bundle = bundle
        self._output_model = _create_output_model(bundle)
        self._scenario_by_code = {
            str(scenario["code"]): scenario for scenario in bundle.manifest["scenarios"]
        }
        self._detail_by_code: dict[str, tuple[dict, dict]] = {
            str(detail["code"]): (scenario, detail)
            for scenario in bundle.manifest["scenarios"]
            for detail in scenario["details"]
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=bundle.system_prompt),
                (
                    "human",
                    "오늘 날짜: {today}\n\n"
                    "마스터 에이전트가 보정한 사용자 질문:\n{query}\n\n"
                    "질문에 서로 독립적인 요청이 여러 개 있으면 이 서브에이전트 "
                    "안에서 해당하는 모든 세부 시나리오를 matches 배열에 "
                    "한 번씩 선택하세요. 단일 요청이면 matches에는 한 개만 "
                    "반환하세요. 다른 서브에이전트의 시나리오는 선택하지 마세요.",
                ),
            ]
        )
        # 시나리오 구조화 출력이 유효한 결과를 못 만들면 응답이 끝나지 않아
        # 평가 행이 통째로 제한시간에 걸린다. 마스터 분류기와 같은 호출 상한을
        # 공유해 그런 실행을 빠르게 끊는다.
        llm = ChatOpenAI(
            base_url=settings.genos_openai_base_url,
            model=settings.genos_model,
            api_key=settings.genos_bearer_token,
            temperature=bundle.temperature,
            **settings.llm_client_options,
        )
        self._chain = prompt | llm.with_structured_output(
            self._output_model,
            method="json_schema",
            strict=True,
        )
        self._model = settings.genos_model
        self._endpoint = settings.genos_openai_base_url
        self._max_retries = settings.llm_max_retries
        logger.info(
            "======== 시나리오 서브에이전트 준비 완료 | 에이전트=%s | "
            "프롬프트버전=%s | 시나리오개수=%d | 세부시나리오개수=%d | "
            "자동재시도=%d회 | 코드위치=app/subagents/router.py:"
            "ScenarioSubagent.classify",
            bundle.agent_code,
            bundle.version,
            len(self._scenario_by_code),
            len(self._detail_by_code),
            settings.llm_max_retries,
        )

    @timed("서브에이전트 시나리오 분류")
    async def classify(
        self,
        query: str,
        *,
        today: date | None = None,
    ) -> SubagentResult:
        """LLM으로 시나리오와 원본 조회 파라미터를 분류한다."""

        reference_date = today or date.today()
        # 단위 테스트가 네트워크 생성자 없이 구조화 체인만 주입하는 경우에도
        # 진단 로그가 테스트 자체를 방해하지 않도록 표시용 기본값을 사용한다.
        endpoint = getattr(self, "_endpoint", "테스트 주입 체인")
        model = getattr(self, "_model", "테스트 주입 모델")
        max_retries = getattr(self, "_max_retries", 0)
        logger.info(
            "======== 서브에이전트 LLM 전달 준비 | 에이전트=%s | "
            "코드위치=app/subagents/router.py:ScenarioSubagent.classify | "
            "엔드포인트=%s | 모델=%s | 기준일=%s | 질문길이=%d | "
            "자동재시도=%d회",
            self._bundle.agent_code,
            endpoint,
            model,
            reference_date.isoformat(),
            len(query),
            max_retries,
        )
        # 마스터와 마찬가지로 전체 프롬프트는 운영 로그에 출력하지 않는다.
        try:
            async with async_timed_block("서브에이전트 LLM 응답 대기"):
                logger.info(
                    "======== 서브에이전트 LLM 요청 직전 | 에이전트=%s | "
                    "보정질문=%s | 기준일=%s | StructuredOutput=%s",
                    self._bundle.agent_code,
                    query,
                    reference_date.isoformat(),
                    self._output_model.__name__,
                )
                structured = await self._chain.ainvoke(
                    {
                        "today": reference_date.isoformat(),
                        "query": query,
                    }
                )
        except Exception as exc:
            log_failure_diagnostic(
                stage=f"{self._bundle.agent_code} 서브에이전트 LLM 호출",
                code_location="app/subagents/router.py:ScenarioSubagent.classify",
                exc=exc,
                likely_cause=(
                    "GenOS LLM 연결·인증 오류 또는 선택된 서브에이전트의 "
                    "시나리오/파라미터 Structured Output 스키마 불일치"
                ),
                corrective_action=(
                    f"prompts/subagents/{self._bundle.agent_code.casefold()}의 "
                    "active.yaml, manifest.yaml, system prompt와 "
                    "app/subagents/router.py:_create_output_model을 확인하세요."
                ),
                retry_count=max_retries,
                context={
                    "agent_code": self._bundle.agent_code,
                    "query": query,
                    "today": reference_date.isoformat(),
                    "endpoint": endpoint,
                    "model": model,
                },
            )
            raise

        logger.info(
            "======== 서브에이전트 LLM 원본 구조화 결과 | 에이전트=%s | 매칭=%s",
            self._bundle.agent_code,
            [
                {
                    "scenario_code": str(selected.scenario_code),
                    "detail_scenario_code": str(selected.detail_scenario_code),
                    "parameter_keys": sorted(selected.parameters.model_dump()),
                }
                for selected in structured.matches
            ],
        )

        matches: list[SubagentScenarioMatch] = []
        seen_detail_codes: set[str] = set()
        for selected in structured.matches:
            selected_scenario_code = str(selected.scenario_code)
            detail_code = str(selected.detail_scenario_code)
            if detail_code in seen_detail_codes:
                logger.info(
                    "======== 서브에이전트 중복 세부시나리오 제거 | 코드=%s",
                    detail_code,
                )
                continue
            seen_detail_codes.add(detail_code)
            scenario, detail = self._detail_by_code[detail_code]
            # LLM의 최상위 코드가 어긋나면 더 구체적인 세부 시나리오의 실제
            # 부모로 자동 보정한다.
            scenario_code = str(scenario["code"])
            if selected_scenario_code != scenario_code:
                logger.info(
                    "======== 서브에이전트 시나리오 조합 자동 보정 | "
                    "LLM최상위=%s | 세부시나리오=%s | 적용최상위=%s",
                    selected_scenario_code,
                    detail_code,
                    scenario_code,
                )
            matches.append(
                SubagentScenarioMatch(
                    scenario_code=scenario_code,
                    scenario_name=str(scenario["name"]),
                    detail_scenario_code=detail_code,
                    detail_scenario_name=str(detail["name"]),
                    parameters=_normalize_parameters(
                        raw=selected.parameters.model_dump(),
                        manifest=self._bundle.manifest,
                        detail=detail,
                    ),
                    interaction=_parse_interaction(detail),
                    mcp_workflow=_parse_mcp_workflow(detail),
                )
            )

        # manifest가 특정 복합 표현에 필수 세부 시나리오 조합을 선언하면
        # LLM이 일부를 누락했더라도 공통 로직으로 보완한다. 업무 용어나 코드는
        # Python에 하드코딩하지 않아 다른 서브에이전트도 같은 기능을 쓸 수 있다.
        normalized_query = query.casefold()
        for rule in self._bundle.manifest.get("required_match_rules", []):
            terms = [str(term).casefold() for term in rule.get("all_terms", [])]
            if not terms or not all(term in normalized_query for term in terms):
                continue
            skip_codes = {
                str(code) for code in rule.get("skip_if_selected_detail_codes", [])
            }
            selected_skip_codes = seen_detail_codes & skip_codes
            if selected_skip_codes:
                logger.info(
                    "======== 서브에이전트 필수 다중 매칭 규칙 생략 | "
                    "에이전트=%s | 규칙=%s | 이미선택된제외코드=%s",
                    self._bundle.agent_code,
                    rule.get("name", "이름없음"),
                    sorted(selected_skip_codes),
                )
                continue
            logger.info(
                "======== 서브에이전트 필수 다중 매칭 규칙 적용 | "
                "에이전트=%s | 규칙=%s",
                self._bundle.agent_code,
                rule.get("name", "이름없음"),
            )
            for required_detail_code in rule.get("detail_codes", []):
                detail_code = str(required_detail_code)
                if detail_code in seen_detail_codes:
                    continue
                scenario, detail = self._detail_by_code[detail_code]
                seen_detail_codes.add(detail_code)
                matches.append(
                    SubagentScenarioMatch(
                        scenario_code=str(scenario["code"]),
                        scenario_name=str(scenario["name"]),
                        detail_scenario_code=detail_code,
                        detail_scenario_name=str(detail["name"]),
                        parameters=_normalize_parameters(
                            raw={
                                str(name): None
                                for name in _detail_parameter_names(
                                    self._bundle.manifest,
                                    detail,
                                )
                            },
                            manifest=self._bundle.manifest,
                            detail=detail,
                        ),
                        interaction=_parse_interaction(detail),
                        mcp_workflow=_parse_mcp_workflow(detail),
                    )
                )

        if not matches:
            raise ValueError("서브에이전트는 하나 이상의 시나리오를 선택해야 합니다.")
        primary = matches[0]
        result = SubagentResult(
            agent_code=self._bundle.agent_code,
            prompt_version=self._bundle.version,
            scenario_code=primary.scenario_code,
            scenario_name=primary.scenario_name,
            detail_scenario_code=primary.detail_scenario_code,
            detail_scenario_name=primary.detail_scenario_name,
            parameters=primary.parameters,
            interaction=primary.interaction,
            mcp_workflow=primary.mcp_workflow,
            matches=matches,
        )
        logger.info(
            "======== 서브에이전트 분류 완료 | 에이전트=%s | "
            "매칭개수=%d | 세부시나리오=%s",
            result.agent_code,
            len(result.matches),
            [match.detail_scenario_code for match in result.matches],
        )
        return result

    def result_for_detail(self, detail_scenario_code: str) -> SubagentResult | None:
        """manifest가 명시한 후속 detail 결과를 LLM 추측 없이 구성한다."""

        detail_entry = self._detail_by_code.get(detail_scenario_code)
        if detail_entry is None:
            logger.warning(
                "!!!!!!!! 추천질문 후속 세부시나리오 없음 | 에이전트=%s | "
                "세부시나리오=%s",
                self._bundle.agent_code,
                detail_scenario_code,
            )
            return None

        scenario, detail = detail_entry
        parameters = _normalize_parameters(
            raw={
                str(name): None
                for name in _detail_parameter_names(self._bundle.manifest, detail)
            },
            manifest=self._bundle.manifest,
            detail=detail,
        )
        match = SubagentScenarioMatch(
            scenario_code=str(scenario["code"]),
            scenario_name=str(scenario["name"]),
            detail_scenario_code=detail_scenario_code,
            detail_scenario_name=str(detail["name"]),
            parameters=parameters,
            interaction=_parse_interaction(detail),
            mcp_workflow=_parse_mcp_workflow(detail),
        )
        logger.info(
            "======== 추천질문 후속 세부시나리오 직접 선택 | 에이전트=%s | "
            "세부시나리오=%s | 서브LLM호출=생략",
            self._bundle.agent_code,
            detail_scenario_code,
        )
        return SubagentResult(
            agent_code=self._bundle.agent_code,
            prompt_version=self._bundle.version,
            scenario_code=match.scenario_code,
            scenario_name=match.scenario_name,
            detail_scenario_code=match.detail_scenario_code,
            detail_scenario_name=match.detail_scenario_name,
            parameters=match.parameters,
            interaction=match.interaction,
            mcp_workflow=match.mcp_workflow,
            matches=[match],
        )


class ManifestSubagentRouter:
    """registry에 등록된 여러 시나리오 서브에이전트를 공통 방식으로 실행한다."""

    def __init__(self, agents: dict[str, ScenarioSubagent]) -> None:
        self._agents = agents

    def supports(self, agent_code: str) -> bool:
        return agent_code.upper() in self._agents

    def registered_codes(self) -> tuple[str, ...]:
        return tuple(sorted(self._agents))

    async def classify(
        self,
        *,
        agent_code: str,
        query: str,
    ) -> SubagentResult | None:
        agent = self._agents.get(agent_code.upper())
        if agent is None:
            logger.info(
                "======== 서브에이전트 미등록 | 에이전트=%s | 실행생략",
                agent_code.upper(),
            )
            return None
        return await agent.classify(query)

    async def classify_for_detail(
        self,
        *,
        agent_code: str,
        query: str,
        detail_scenario_code: str,
    ) -> SubagentResult | None:
        """추천질문 metadata가 지정한 detail을 활성 manifest에서 직접 선택한다."""

        agent = self._agents.get(agent_code.upper())
        if agent is None:
            return None
        return agent.result_for_detail(detail_scenario_code)


@timed("서브에이전트 라우터 생성")
def create_subagent_router(settings: Settings) -> SubagentRouter:
    """registry의 활성 서브에이전트를 모두 생성한다."""

    bundles = SubagentPromptLoader().load_all()
    agents = {
        code: ScenarioSubagent(settings, bundle) for code, bundle in bundles.items()
    }
    return ManifestSubagentRouter(agents)


def _create_output_model(
    bundle: ScenarioPromptBundle,
) -> type[BaseModel]:
    """manifest 값으로 LLM에 전달할 엄격한 Pydantic JSON Schema를 만든다."""

    detail_codes = [
        str(detail["code"])
        for scenario in bundle.manifest["scenarios"]
        for detail in scenario["details"]
    ]
    match_models: list[type[BaseModel]] = []
    definitions = bundle.manifest["parameter_definitions"]
    for scenario in bundle.manifest["scenarios"]:
        scenario_code = str(scenario["code"])
        for detail in scenario["details"]:
            detail_code = str(detail["code"])
            parameter_fields: dict[str, tuple[Any, Field]] = {}
            for name in _detail_parameter_names(bundle.manifest, detail):
                definition = definitions[name]
                allowed_values = definition.get("allowed_values")
                value_type = str(definition.get("value_type", "string"))
                parameter_type: Any
                if value_type == "string_list":
                    parameter_type = list[str] | None
                elif value_type == "string":
                    parameter_type = str | None
                else:
                    raise ValueError(
                        "지원하지 않는 parameter value_type입니다: "
                        f"name={name}, value_type={value_type}"
                    )
                if allowed_values and value_type == "string":
                    parameter_type = Literal.__getitem__(
                        tuple(str(value) for value in allowed_values) + (None,)
                    )
                field_options: dict[str, Any] = {
                    "description": str(definition["description"]),
                }
                if value_type == "string":
                    field_options["pattern"] = definition.get("pattern")
                elif value_type == "string_list":
                    field_options["min_length"] = int(
                        definition.get("min_items", 1)
                    )
                    field_options["max_length"] = int(
                        definition.get("max_items", 10)
                    )
                parameter_fields[name] = (
                    parameter_type,
                    Field(**field_options),
                )
            safe_detail_name = "".join(
                part.title() for part in detail_code.casefold().split("_")
            )
            parameters_model = create_model(
                f"{bundle.agent_code}{safe_detail_name}Parameters",
                __config__=ConfigDict(extra="forbid"),
                **parameter_fields,
            )
            match_models.append(
                create_model(
                    f"{bundle.agent_code}{safe_detail_name}Match",
                    __config__=ConfigDict(extra="forbid"),
                    scenario_code=(
                        Literal.__getitem__(scenario_code),
                        Field(description="선택한 최상위 시나리오 코드"),
                    ),
                    detail_scenario_code=(
                        Literal.__getitem__(detail_code),
                        Field(description="선택한 세부 시나리오 코드"),
                    ),
                    parameters=(
                        parameters_model,
                        Field(
                            description=(
                                "선택한 세부 시나리오에서 허용된 파라미터만 추출"
                            )
                        ),
                    ),
                )
            )

    match_union = Union.__getitem__(tuple(match_models))
    discriminated_match = Annotated[
        match_union,
        Field(discriminator="detail_scenario_code"),
    ]
    return create_model(
        f"{bundle.agent_code}ScenarioOutput",
        __config__=ConfigDict(extra="forbid"),
        matches=(
            list[discriminated_match],
            Field(
                min_length=1,
                max_length=len(detail_codes),
                description=(
                    "질문에 포함된 독립 요청과 일치하는 모든 시나리오. "
                    "현재 선택된 서브에이전트의 코드만 사용하고 동일 세부 "
                    "시나리오는 한 번만 반환"
                ),
            ),
        ),
    )


def _normalize_parameters(
    *,
    raw: dict[str, Any],
    manifest: dict[str, Any],
    detail: dict[str, Any],
) -> dict[str, Any]:
    """LLM 누락값을 빈 문자열로 바꾸고 업무 기본값은 적용하지 않는다.

    날짜·코드 기본값, 필수값과 파라미터 우선순위는 MCP 도구마다 다르므로
    ``app/mcp/scenarios/*.py``의 해당 handler에서 최종 arguments를 만들 때
    처리한다.
    """

    normalized: dict[str, Any] = {}
    for name in _detail_parameter_names(manifest, detail):
        definition = manifest["parameter_definitions"][name]
        value = raw.get(str(name))
        if str(definition.get("value_type", "string")) == "string_list":
            items = value if isinstance(value, list) else []
            normalized_items: list[str] = []
            seen: set[str] = set()
            for item in items:
                keyword = str(item).strip()
                if not keyword or keyword in seen:
                    continue
                seen.add(keyword)
                normalized_items.append(keyword)
            normalized[str(name)] = normalized_items
            continue
        normalized[str(name)] = "" if value is None else str(value).strip()
    return normalized


def _detail_parameter_names(
    manifest: dict[str, Any],
    detail: dict[str, Any],
) -> list[str]:
    """한 detail이 LLM에서 추출하도록 허용된 파라미터 이름만 반환한다."""

    configured = detail.get("parameters", [])
    return [
        str(name)
        for name in configured
        if str(name) in manifest["parameter_definitions"]
    ]


def _parse_interaction(detail: dict[str, Any]) -> ScenarioInteraction | None:
    """manifest detail의 선택적 다단계 상호작용 선언을 모델로 변환한다."""

    raw = detail.get("interaction")
    if raw is None:
        return None
    return ScenarioInteraction.model_validate(raw)


def _parse_mcp_workflow(detail: dict[str, Any]) -> ScenarioMcpWorkflow | None:
    """manifest detail의 선택적 순차 MCP workflow 선언을 모델로 변환한다."""

    raw = detail.get("mcp_workflow")
    if raw is None:
        return None
    return ScenarioMcpWorkflow.model_validate(raw)
