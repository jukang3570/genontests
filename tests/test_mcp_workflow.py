"""세부 시나리오별 1~N단계 MCP workflow와 파라미터 schema를 검증한다."""

import unittest

from pydantic import ValidationError

from app.config import Settings
from app.mcp.client import GenosMcpToolExecutor, build_mcp_request_id
from app.mcp.models import McpExecutionResult
from app.mcp.request_builder import (
    build_mcp_workflow_step_request,
    get_mcp_workflow_iteration_items,
    get_mcp_workflow_step_count,
)
from app.mcp.workflow import aggregate_fanout_results
from app.mcp.workflow_handlers import INPUT_MAPPER_REGISTRY
from app.csv_trace import EmptyTraceRecorder
from app.graph import MasterIntentGraph
from app.subagents.models import SubagentResult
from app.subagents.prompt_loader import SubagentPromptLoader
from app.subagents.router import _create_output_model


class McpWorkflowTests(unittest.TestCase):
    def _subagent(self) -> SubagentResult:
        return SubagentResult.model_validate(
            {
                "agent_code": "TEST",
                "prompt_version": "v1",
                "scenario_code": "LOOKUP",
                "scenario_name": "연계 조회",
                "detail_scenario_code": "LOOKUP_CHAIN",
                "detail_scenario_name": "두 단계 연계 조회",
                "parameters": {"search_key": "A-100"},
                "mcp_workflow": {
                    "steps": [
                        {
                            "code": "LOOKUP_TARGET",
                            "tool": {
                                "name": "first_tool",
                                "arguments": {
                                    "searchKey": {
                                        "source": "parameter",
                                        "key": "search_key",
                                        "required": True,
                                    }
                                },
                            },
                        },
                        {
                            "code": "LOOKUP_DETAIL",
                            "tool": {
                                "name": "second_tool",
                                "arguments": {
                                    "targetId": {
                                        "source": "step_result",
                                        "step_code": "LOOKUP_TARGET",
                                        "key": "targetId",
                                        "required": True,
                                    }
                                },
                            },
                        },
                    ],
                    "final_step": "LOOKUP_DETAIL",
                },
            }
        )

    def test_mapped_step_requires_registered_handler_shape(self) -> None:
        with self.assertRaises(ValidationError):
            SubagentResult.model_validate(
                {
                    "agent_code": "TEST",
                    "prompt_version": "v1",
                    "scenario_code": "LOOKUP",
                    "scenario_name": "연계 조회",
                    "detail_scenario_code": "LOOKUP_CHAIN",
                    "detail_scenario_name": "mapper 검증",
                    "mcp_workflow": {
                        "steps": [
                            {
                                "code": "MAPPED",
                                "execution": {"mode": "mapped"},
                                "tool": {"name": "test_tool"},
                            }
                        ]
                    },
                }
            )

    def test_builds_second_payload_from_first_step_result(self) -> None:
        subagent = self._subagent()
        self.assertEqual(get_mcp_workflow_step_count(subagent), 2)
        first_payload = build_mcp_workflow_step_request(
            request_id="request:LOOKUP_TARGET",
            subagent=subagent,
            employee_id="K3003980",
            session_id="session-1",
            thread_id="thread-1",
            request_context={},
            step_index=0,
        )
        self.assertEqual(first_payload["params"]["name"], "first_tool")
        self.assertEqual(
            first_payload["params"]["arguments"],
            {"searchKey": "A-100"},
        )

        first_result = McpExecutionResult(
            backend="http",
            tool_name="first_tool",
            request_id="request:LOOKUP_TARGET",
            arguments={"searchKey": "A-100"},
            succeeded=True,
            result={
                "data": [
                    {"objId": "targetId", "objType": "String", "objVal": "T-1"}
                ]
            },
            workflow_step_code="LOOKUP_TARGET",
            workflow_step_index=0,
            workflow_step_count=2,
            workflow_is_final=False,
        )
        second_payload = build_mcp_workflow_step_request(
            request_id="request:LOOKUP_DETAIL",
            subagent=subagent,
            employee_id="K3003980",
            session_id="session-1",
            thread_id="thread-1",
            request_context={},
            step_index=1,
            previous_results=[first_result],
        )
        self.assertEqual(second_payload["params"]["name"], "second_tool")
        self.assertEqual(
            second_payload["params"]["arguments"],
            {"targetId": "T-1"},
        )

    def test_request_id_distinguishes_workflow_steps(self) -> None:
        common = {
            "project_code": "acqsc",
            "employee_id": "K3003980",
            "session_id": "session-1",
            "thread_id": "thread-1",
            "detail_scenario_code": "LOOKUP_CHAIN",
        }
        first = build_mcp_request_id(
            **common,
            workflow_step_code="LOOKUP_TARGET",
        )
        second = build_mcp_request_id(
            **common,
            workflow_step_code="LOOKUP_DETAIL",
        )
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(":LOOKUP_TARGET"))
        self.assertTrue(second.endswith(":LOOKUP_DETAIL"))

    def test_fanout_item_can_supply_multiple_arguments_with_request_context(
        self,
    ) -> None:
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "TEST",
                "prompt_version": "v1",
                "scenario_code": "LOOKUP",
                "scenario_name": "연계 조회",
                "detail_scenario_code": "LOOKUP_CHAIN",
                "detail_scenario_name": "목록 상세 조회",
                "parameters": {},
                "mcp_workflow": {
                    "steps": [
                        {"code": "LOOKUP_TARGETS", "tool": {"name": "list_tool"}},
                        {
                            "code": "LOOKUP_DETAILS",
                            "execution": {
                                "mode": "for_each",
                                "source_step": "LOOKUP_TARGETS",
                                "collection_path": "data",
                                "error_policy": "continue",
                            },
                            "tool": {
                                "name": "detail_tool",
                                "arguments": {
                                    "customerId": {
                                        "source": "current_item",
                                        "key": "customerId",
                                        "required": True,
                                    },
                                    "contractId": {
                                        "source": "current_item",
                                        "key": "contractId",
                                        "required": True,
                                    },
                                    "bearerToken": {
                                        "source": "request_context",
                                        "key": "access_token",
                                        "required": True,
                                    },
                                    "employeeId": {
                                        "source": "runtime",
                                        "key": "employee_id",
                                        "required": True,
                                    },
                                },
                            },
                        },
                    ]
                },
            }
        )
        first_result = McpExecutionResult(
            backend="http",
            tool_name="list_tool",
            request_id="request:list",
            arguments={},
            succeeded=True,
            result={
                "data": [
                    {"customerId": "C-001", "contractId": "CT-001"},
                    {"customerId": "C-002", "contractId": "CT-002"},
                ]
            },
            workflow_step_code="LOOKUP_TARGETS",
            workflow_step_count=2,
            workflow_is_final=False,
        )
        items = get_mcp_workflow_iteration_items(
            subagent=subagent,
            step_index=1,
            previous_results=[first_result],
        )
        self.assertEqual(len(items), 2)

        payload = build_mcp_workflow_step_request(
            request_id="request:detail:2",
            subagent=subagent,
            employee_id="K3003980",
            session_id="session-1",
            thread_id="thread-1",
            request_context={"access_token": "access-123"},
            step_index=1,
            previous_results=[first_result],
            current_item=items[1],
        )
        self.assertEqual(
            payload["params"]["arguments"],
            {
                "customerId": "C-002",
                "contractId": "CT-002",
                "bearerToken": "access-123",
                "employeeId": "K3003980",
            },
        )

    def test_request_id_distinguishes_fanout_items(self) -> None:
        common = {
            "project_code": "acqsc",
            "employee_id": "K3003980",
            "session_id": "session-1",
            "thread_id": "thread-1",
            "detail_scenario_code": "LOOKUP_CHAIN",
            "workflow_step_code": "LOOKUP_DETAILS",
        }
        first = build_mcp_request_id(**common, workflow_item_index=0)
        second = build_mcp_request_id(**common, workflow_item_index=1)
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(":ITEM-0001"))
        self.assertTrue(second.endswith(":ITEM-0002"))

    def test_continue_policy_aggregates_success_and_failure_batches(self) -> None:
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "TEST",
                "prompt_version": "v1",
                "scenario_code": "LOOKUP",
                "scenario_name": "연계 조회",
                "detail_scenario_code": "LOOKUP_CHAIN",
                "detail_scenario_name": "목록 상세 조회",
                "parameters": {},
                "mcp_workflow": {
                    "steps": [
                        {"code": "LIST", "tool": {"name": "list_tool"}},
                        {
                            "code": "DETAIL",
                            "execution": {
                                "mode": "for_each",
                                "source_step": "LIST",
                                "error_policy": "continue",
                            },
                            "tool": {"name": "detail_tool"},
                        },
                    ]
                },
            }
        )
        step = subagent.mcp_workflow.steps[1]
        success = McpExecutionResult(
            backend="http",
            tool_name="detail_tool",
            request_id="detail:1",
            arguments={"targetId": "T-1"},
            succeeded=True,
            result={"data": [{"objId": "name", "objVal": "첫 번째"}]},
            workflow_step_code="DETAIL",
            workflow_step_index=1,
            workflow_step_count=2,
            workflow_is_final=False,
            workflow_execution_mode="for_each",
            workflow_item_index=0,
            workflow_item_count=2,
        )
        failure = McpExecutionResult(
            backend="http",
            tool_name="detail_tool",
            request_id="detail:2",
            arguments={"targetId": "T-2"},
            succeeded=False,
            outcome="ERROR",
            error="timeout",
            workflow_step_code="DETAIL",
            workflow_step_index=1,
            workflow_step_count=2,
            workflow_is_final=False,
            workflow_execution_mode="for_each",
            workflow_item_index=1,
            workflow_item_count=2,
        )

        aggregate = aggregate_fanout_results(
            step=step,
            workflow_step_index=1,
            workflow_step_count=2,
            item_count=2,
            results=[success, failure],
            aggregate_request_id="detail:aggregate",
        )

        self.assertTrue(aggregate.succeeded)
        self.assertEqual(aggregate.outcome, "SUCCESS")
        self.assertTrue(aggregate.workflow_is_aggregate)
        self.assertEqual(aggregate.result["fanout"]["successCount"], 1)
        self.assertEqual(aggregate.result["fanout"]["errorCount"], 1)
        self.assertEqual(len(aggregate.result["data"]), 1)
        self.assertEqual(len(aggregate.result["batches"]), 2)

    def test_all_fanout_no_data_becomes_business_code_1001(self) -> None:
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "TEST",
                "prompt_version": "v1",
                "scenario_code": "LOOKUP",
                "scenario_name": "복수 조회",
                "detail_scenario_code": "LOOKUP_MANY",
                "detail_scenario_name": "복수 조회",
                "parameters": {},
                "mcp_workflow": {
                    "steps": [
                        {
                            "code": "LIST",
                            "tool": {"name": "list_tool"},
                        },
                        {
                            "code": "DETAIL",
                            "execution": {
                                "mode": "for_each",
                                "source_step": "LIST",
                                "error_policy": "continue",
                            },
                            "tool": {"name": "detail_tool"},
                        }
                    ]
                },
            }
        )
        step = subagent.mcp_workflow.steps[1]
        no_data_results = [
            McpExecutionResult(
                backend="http",
                tool_name="detail_tool",
                request_id=f"detail:{index}",
                arguments={"targetId": f"T-{index}"},
                succeeded=True,
                outcome="NO_DATA",
                business_code="1001",
                result={"data": []},
                workflow_item_index=index - 1,
                workflow_item_count=2,
            )
            for index in (1, 2)
        ]

        aggregate = aggregate_fanout_results(
            step=step,
            workflow_step_index=0,
            workflow_step_count=1,
            item_count=2,
            results=no_data_results,
            aggregate_request_id="detail:aggregate",
        )

        self.assertTrue(aggregate.succeeded)
        self.assertEqual(aggregate.outcome, "NO_DATA")
        self.assertEqual(aggregate.business_code, "1001")
        self.assertEqual(aggregate.result["fanout"]["callCount"], 2)
        self.assertEqual(aggregate.result["fanout"]["noDataCount"], 2)
        self.assertEqual(aggregate.result["data"], [])

        partial_success = no_data_results[0].model_copy(
            update={
                "outcome": "SUCCESS",
                "business_code": None,
                "result": {
                    "data": [{"objId": "score", "objVal": "10"}]
                },
            }
        )
        mixed = aggregate_fanout_results(
            step=step,
            workflow_step_index=0,
            workflow_step_count=1,
            item_count=2,
            results=[partial_success, no_data_results[1]],
            aggregate_request_id="detail:mixed-aggregate",
        )

        self.assertEqual(mixed.outcome, "SUCCESS")
        self.assertIsNone(mixed.error)
        self.assertEqual(mixed.result["fanout"]["successCount"], 1)
        self.assertEqual(mixed.result["fanout"]["noDataCount"], 1)
        self.assertEqual(len(mixed.result["data"]), 1)


class DetailParameterSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bundle = SubagentPromptLoader().load_one(directory="performance-fee")
        cls.output_model = _create_output_model(bundle)

    def test_one_llm_output_uses_only_selected_detail_parameters(self) -> None:
        parsed = self.output_model.model_validate(
            {
                "matches": [
                    {
                        "scenario_code": "PERFORMANCE_SUMMARY",
                        "detail_scenario_code": "PERFORMANCE_SUMMARY_TOTAL",
                        "parameters": {
                            "closing_year_month": None,
                            "reference_date": None,
                        },
                    },
                    {
                        "scenario_code": "FEE_DETAILS",
                        "detail_scenario_code": "WITHHOLDING_TAX",
                        "parameters": {"reference_year": "2025"},
                    },
                ]
            }
        )
        self.assertEqual(len(parsed.matches), 2)

    def test_irrelevant_fax_parameter_is_forbidden_for_performance(self) -> None:
        with self.assertRaises(ValidationError):
            self.output_model.model_validate(
                {
                    "matches": [
                        {
                            "scenario_code": "PERFORMANCE_SUMMARY",
                            "detail_scenario_code": "PERFORMANCE_SUMMARY_TOTAL",
                            "parameters": {
                                "closing_year_month": None,
                                "reference_date": None,
                                "fax_number": "02-1234-5678",
                            },
                        }
                    ]
                }
            )


class _SequentialExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    async def execute(
        self,
        *,
        subagent: SubagentResult,
        employee_id: str,
        session_id: str,
        thread_id: str,
        request_context: dict | None = None,
        workflow_step_index: int = 0,
        previous_results: tuple[McpExecutionResult, ...] | list[McpExecutionResult] = (),
    ) -> McpExecutionResult:
        self.calls.append((workflow_step_index, len(previous_results)))
        is_final = workflow_step_index == 1
        data = (
            [
                {"objId": "column1", "objType": "String", "objVal": "100"},
                {"objId": "column2", "objType": "String", "objVal": "2"},
            ]
            if is_final
            else [{"objId": "targetId", "objType": "String", "objVal": "T-1"}]
        )
        return McpExecutionResult(
            backend="test",
            tool_name="second_tool" if is_final else "first_tool",
            request_id=f"request:{workflow_step_index}",
            arguments={},
            succeeded=True,
            result={"data": data},
            workflow_step_code="LOOKUP_DETAIL" if is_final else "LOOKUP_TARGET",
            workflow_step_index=workflow_step_index,
            workflow_step_count=2,
            workflow_is_final=is_final,
        )

    async def aclose(self) -> None:
        return None


class GraphWorkflowExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_keeps_all_steps_but_only_one_final_result(self) -> None:
        executor = _SequentialExecutor()
        graph = MasterIntentGraph.__new__(MasterIntentGraph)
        graph._mcp_executor = executor
        graph._trace_recorder = EmptyTraceRecorder()
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "PERFORMANCE_FEE",
                "prompt_version": "v1",
                "scenario_code": "COMPOSITE_CONVERSION",
                "scenario_name": "복합환산조회",
                "detail_scenario_code": "COMPOSITE_CONVERSION_SCORE",
                "detail_scenario_name": "복합환산 점수",
                "parameters": {
                    "closing_year_month": "202608",
                    "reference_date": "",
                },
                "mcp_workflow": {
                    "steps": [
                        {"code": "LOOKUP_TARGET", "tool": {"name": "first_tool"}},
                        {"code": "LOOKUP_DETAIL", "tool": {"name": "second_tool"}},
                    ],
                    "final_step": "LOOKUP_DETAIL",
                },
            }
        )
        state = {
            "subagent": subagent.model_dump(mode="json"),
            "employee_id": "K3003980",
            "session_id": "session-1",
            "thread_id": "thread-1",
            "request_context": {},
        }

        update = await graph._call_mcp(state)

        self.assertEqual(executor.calls, [(0, 0), (1, 1)])
        self.assertEqual(len(update["mcp_workflow_results"]), 2)
        self.assertEqual(len(update["mcp_results"]), 1)
        self.assertEqual(update["mcp_results"][0]["tool_name"], "second_tool")
        self.assertEqual(update["mcp_results"][0]["result_format"], "query.v1")


class _FanoutExecutor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(
        self,
        *,
        subagent: SubagentResult,
        employee_id: str,
        session_id: str,
        thread_id: str,
        request_context: dict | None = None,
        workflow_step_index: int = 0,
        previous_results: tuple[McpExecutionResult, ...]
        | list[McpExecutionResult] = (),
        current_item=None,
        workflow_item_index: int | None = None,
        workflow_item_count: int | None = None,
    ) -> McpExecutionResult:
        self.calls.append(
            {
                "step": workflow_step_index,
                "item": workflow_item_index,
                "current_item": current_item,
                "access_token": (request_context or {}).get("access_token"),
            }
        )
        if workflow_step_index == 0:
            return McpExecutionResult(
                backend="test",
                tool_name="list_tool",
                request_id="request:list",
                arguments={},
                succeeded=True,
                result={
                    "data": [
                        {"customerId": "C-001", "contractId": "CT-001"},
                        {"customerId": "C-002", "contractId": "CT-002"},
                    ]
                },
                workflow_step_code="LOOKUP_TARGETS",
                workflow_step_index=0,
                workflow_step_count=2,
                workflow_is_final=False,
            )
        assert isinstance(current_item, dict)
        arguments = {
            "customerId": current_item["customerId"],
            "contractId": current_item["contractId"],
            "bearerToken": (request_context or {})["access_token"],
        }
        return McpExecutionResult(
            backend="test",
            tool_name="detail_tool",
            request_id=f"request:detail:{workflow_item_index}",
            arguments=arguments,
            succeeded=True,
            result={
                "data": [
                    {
                        "objId": "detailValue",
                        "objType": "String",
                        "objVal": f"DETAIL-{current_item['customerId']}",
                    }
                ]
            },
            workflow_step_code="LOOKUP_DETAILS",
            workflow_step_index=1,
            workflow_step_count=2,
            workflow_is_final=False,
            workflow_execution_mode="for_each",
            workflow_item_index=workflow_item_index,
            workflow_item_count=workflow_item_count,
            workflow_source_step_code="LOOKUP_TARGETS",
        )

    async def aclose(self) -> None:
        return None


class GraphFanoutExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_graph_fans_out_merges_and_passes_all_data_to_formatter(
        self,
    ) -> None:
        executor = _FanoutExecutor()
        graph = MasterIntentGraph.__new__(MasterIntentGraph)
        graph._mcp_executor = executor
        graph._trace_recorder = EmptyTraceRecorder()
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "PERFORMANCE_FEE",
                "prompt_version": "v1",
                "scenario_code": "PERFORMANCE_SUMMARY",
                "scenario_name": "실적 종합조회",
                "detail_scenario_code": "PERFORMANCE_SUMMARY_TOTAL",
                "detail_scenario_name": "실적 종합 조회",
                "parameters": {
                    "closing_year_month": "202608",
                    "reference_date": "",
                },
                "mcp_workflow": {
                    "steps": [
                        {"code": "LOOKUP_TARGETS", "tool": {"name": "list_tool"}},
                        {
                            "code": "LOOKUP_DETAILS",
                            "execution": {
                                "mode": "for_each",
                                "source_step": "LOOKUP_TARGETS",
                                "error_policy": "continue",
                            },
                            "tool": {"name": "detail_tool"},
                        },
                    ]
                },
            }
        )
        state = {
            "subagent": subagent.model_dump(mode="json"),
            "employee_id": "K3003980",
            "session_id": "session-1",
            "thread_id": "thread-1",
            "request_context": {"access_token": "access-123"},
        }

        update = await graph._call_mcp(state)

        self.assertEqual(
            [(call["step"], call["item"]) for call in executor.calls],
            [(0, None), (1, 0), (1, 1)],
        )
        self.assertEqual(
            [call["access_token"] for call in executor.calls],
            ["access-123", "access-123", "access-123"],
        )
        self.assertEqual(len(update["mcp_workflow_results"]), 4)
        self.assertEqual(len(update["mcp_results"]), 1)
        final = update["mcp_results"][0]
        self.assertTrue(final["workflow_is_aggregate"])
        self.assertTrue(final["workflow_is_final"])
        self.assertEqual(final["result"]["fanout"]["callCount"], 2)
        self.assertEqual(len(final["result"]["data"]), 2)
        self.assertEqual(
            [item["objVal"] for item in final["formatted_result"]["data"]],
            ["DETAIL-C-001", "DETAIL-C-002"],
        )
        self.assertEqual(
            len(
                final["formatted_result"]["request_context"]["mcp_workflow"][
                    "batches"
                ]
            ),
            2,
        )


class ManifestMappedWorkflowExampleTests(unittest.IsolatedAsyncioTestCase):
    """manifest의 test_tool 3단계 예제가 실제 공통 executor에서 동작한다."""

    @staticmethod
    def _settings() -> Settings:
        return Settings(
            genos_url="https://example.invalid",
            genos_serving_id=1,
            genos_model="test",
            genos_bearer_token=None,
            prompt_version=None,
            history_backend="memory",
            redis_url="redis://localhost:6379/0",
            redis_history_key_prefix="test:history",
            history_limit=10,
            redis_dedupe_ttl_seconds=60,
            mcp_backend="mock",
        )

    async def test_composite_excluded_runs_pagination_handler(
        self,
    ) -> None:
        bundle = SubagentPromptLoader().load_one(directory="performance-fee")
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "PERFORMANCE_FEE",
                "prompt_version": bundle.version,
                "scenario_code": "COMPOSITE_CONVERSION",
                "scenario_name": "복합환산조회",
                "detail_scenario_code": "COMPOSITE_CONVERSION_EXCLUDED",
                "detail_scenario_name": "환산 미반영 내역 조회",
                "parameters": {"closing_year_month": "202608"},
            }
        )
        executor = GenosMcpToolExecutor(self._settings())
        graph = MasterIntentGraph.__new__(MasterIntentGraph)
        graph._mcp_executor = executor
        graph._trace_recorder = EmptyTraceRecorder()
        try:
            update = await graph._call_mcp(
                {
                    "subagent": subagent.model_dump(mode="json"),
                    "employee_id": "K3003980",
                    "session_id": "session-mapped",
                    "thread_id": "thread-mapped",
                    "request_context": {"access_token": "secret-access-token"},
                }
            )
        finally:
            await executor.aclose()

        workflow_results = update["mcp_workflow_results"]
        self.assertEqual(len(workflow_results), 2)
        self.assertEqual(
            {item["workflow_step_code"] for item in workflow_results},
            {"EXCLUDED_PAGES"},
        )
        self.assertEqual(
            {item["workflow_execution_mode"] for item in workflow_results},
            {"pagination"},
        )
        self.assertTrue(
            all(
                item["workflow_handler_code"]
                == "performance_fee.composite_conversion_excluded.v1"
                for item in workflow_results
            )
        )
        final = update["mcp_results"][0]
        self.assertTrue(final["workflow_is_aggregate"])
        formatted = final["formatted_result"]
        self.assertEqual(
            formatted["result_formatter_code"],
            "performance_fee.composite_conversion_excluded_output.v1",
        )
        formatter_workflow = formatted["request_context"]["mcp_workflow"]
        self.assertEqual(len(formatter_workflow["results"]), 2)
        self.assertEqual(
            set(formatter_workflow["by_step"]),
            {"EXCLUDED_PAGES"},
        )
        table = formatted["renderables"][0]
        self.assertEqual(len(table["data"]["rows"]), 0)
        self.assertNotIn("secret-access-token", str(formatted))

    async def test_mapper_failure_becomes_safe_mcp_result(self) -> None:
        mapper_code = "test.mapper.failure"

        def fail_mapper(_context):
            raise ValueError("예제 mapper 변환 실패")

        INPUT_MAPPER_REGISTRY[mapper_code] = fail_mapper
        subagent = SubagentResult.model_validate(
            {
                "agent_code": "PERFORMANCE_FEE",
                "prompt_version": "v1",
                "scenario_code": "COMPOSITE_CONVERSION",
                "scenario_name": "복합환산조회",
                "detail_scenario_code": "COMPOSITE_CONVERSION_EXCLUDED",
                "detail_scenario_name": "환산 미반영 내역 조회",
                "parameters": {"closing_year_month": "202608"},
                "mcp_workflow": {
                    "steps": [
                        {
                            "code": "FAIL_MAPPING",
                            "execution": {"mode": "mapped"},
                            "input_mapper": mapper_code,
                            "tool": {"name": "test_tool"},
                        }
                    ]
                },
            }
        )
        executor = GenosMcpToolExecutor(self._settings())
        graph = MasterIntentGraph.__new__(MasterIntentGraph)
        graph._mcp_executor = executor
        graph._trace_recorder = EmptyTraceRecorder()
        try:
            update = await graph._call_mcp(
                {
                    "subagent": subagent.model_dump(mode="json"),
                    "employee_id": "K3003980",
                    "session_id": "session-mapper-error",
                    "thread_id": "thread-mapper-error",
                    "request_context": {},
                }
            )
        finally:
            INPUT_MAPPER_REGISTRY.pop(mapper_code, None)
            await executor.aclose()

        final = update["mcp_results"][0]
        self.assertFalse(final["succeeded"])
        self.assertEqual(final["backend"], "mapper")
        self.assertEqual(final["outcome"], "ERROR")
        self.assertNotIn("예제 mapper 변환 실패", final["user_message"])


if __name__ == "__main__":
    unittest.main()
