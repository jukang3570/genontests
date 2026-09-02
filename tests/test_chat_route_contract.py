"""공개 채팅 API와 목업이 단일 SSE 경로를 사용하는지 검증한다."""

import ast
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


ROOT = Path(__file__).resolve().parents[1]


class ChatRouteContractTests(unittest.TestCase):
    @staticmethod
    def _verification_app():
        settings = Settings(
            genos_url="http://localhost",
            genos_serving_id=1,
            genos_model="test-model",
            genos_bearer_token=None,
            prompt_version=None,
            history_backend="memory",
            redis_url="redis://localhost:6379/0",
            redis_history_key_prefix="chat:history",
            history_limit=10,
            redis_dedupe_ttl_seconds=60,
        )
        unused_dependency = object()
        return create_app(
            settings=settings,
            classifier=unused_dependency,
            history_store=unused_dependency,
            hitl_store=unused_dependency,
            subagent_router=unused_dependency,
            mcp_executor=unused_dependency,
            trace_recorder=unused_dependency,
            answer_service=unused_dependency,
            recommended_question_registry=unused_dependency,
            guardrail_client=unused_dependency,
        )

    def test_code_serving_verification_returns_json_without_auth(self) -> None:
        response = TestClient(self._verification_app()).post(
            "/chat",
            json={"question": "__verify__"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(
            response.json(),
            {"code": 0, "data": {"text": "verified"}},
        )

    def test_normal_chat_still_requires_bearer_auth(self) -> None:
        response = TestClient(self._verification_app()).post(
            "/chat",
            json={"message": "실적을 알려줘"},
        )

        self.assertEqual(response.status_code, 401)

    def test_chat_is_the_only_post_route(self) -> None:
        api_path = ROOT / "app" / "api.py"
        tree = ast.parse(api_path.read_text(encoding="utf-8"))
        post_paths: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                function = decorator.func
                if not (
                    isinstance(function, ast.Attribute)
                    and isinstance(function.value, ast.Name)
                    and function.value.id == "app"
                    and function.attr == "post"
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    continue
                post_paths.append(str(decorator.args[0].value))

        self.assertEqual(post_paths, ["/chat"])

    def test_mock_screens_use_the_chat_sse_endpoint(self) -> None:
        intent_tester = (ROOT / "static" / "intent_tester.html").read_text(
            encoding="utf-8"
        )
        chatting = (ROOT / "static" / "chatting.html").read_text(encoding="utf-8")

        for page in (intent_tester, chatting):
            self.assertNotIn("/v1/chat/stream", page)
            self.assertNotIn('"v1/chat/stream"', page)

        self.assertIn('requestStream("chat"', intent_tester)
        self.assertIn('"X-Debug-Trace": "true"', intent_tester)
        self.assertIn('event === "trace"', intent_tester)
        self.assertIn("renderStateInspector", intent_tester)
        self.assertIn('trace?.kind === "state_transition"', intent_tester)
        self.assertIn("MCP requests", intent_tester)
        self.assertIn("mcpWorkflowResults", intent_tester)
        self.assertIn("순차 MCP workflow 실행", intent_tester)
        self.assertIn("FAN-IN 집계", intent_tester)
        self.assertIn("step.executionMode", intent_tester)
        self.assertIn('const chatEndpoint = `${basePath}/chat`;', chatting)
        self.assertIn('"Accept": "text/event-stream"', chatting)
        self.assertIn('"X-Debug-Trace": "true"', chatting)
        self.assertIn('frame?.event === "trace"', chatting)


if __name__ == "__main__":
    unittest.main()
