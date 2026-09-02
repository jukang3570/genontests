"""동일한 POST /chat에서 코드서빙 JSON과 WAS SSE 계약을 분기한다."""

import unittest

from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings
from app.domain import ClassificationType, IntentClassification
from app.history import InMemoryChatHistoryStore
from app.hitl_store import InMemoryHitlStateStore
from app.mcp.client import EmptyMcpToolExecutor
from app.subagents.router import EmptySubagentRouter


class OutOfScopeClassifier:
    """외부 LLM 없이 JSON/SSE 응답 계약을 확인하는 분류기."""

    async def classify(self, message, history, frontend_agent_code=None):
        return IntentClassification(
            refined_query=message,
            classification_type=ClassificationType.OUT_OF_SCOPE,
            agent_code=None,
        )


def make_test_settings() -> Settings:
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


class ChatDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        app = create_app(
            settings=make_test_settings(),
            classifier=OutOfScopeClassifier(),
            history_store=InMemoryChatHistoryStore(),
            hitl_store=InMemoryHitlStateStore(),
            subagent_router=EmptySubagentRouter(),
            mcp_executor=EmptyMcpToolExecutor(),
        )
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)

    def test_code_serving_verification_returns_json_without_authorization(self) -> None:
        response = self.client.post(
            "/chat",
            json={"question": "__verify__", "history": [], "overrideConfig": {}},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.json(), {"code": 0, "data": {"text": "verified"}})

    def test_code_serving_question_collects_internal_sse_as_json(self) -> None:
        response = self.client.post(
            "/chat",
            json={
                "question": "오늘 날씨를 알려줘",
                "history": [],
                "overrideConfig": {
                    "employee_id": "EMP001",
                    "session_id": "workflow-session",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.json()["code"], 0)
        self.assertTrue(response.json()["data"]["text"])

    def test_was_message_keeps_the_existing_sse_contract(self) -> None:
        response = self.client.post(
            "/chat",
            headers={
                "Authorization": "Bearer test-token",
                "Accept": "text/event-stream",
            },
            json={"message": "오늘 날씨를 알려줘", "session_id": "was-session"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn('"event":"token"', response.text)
        self.assertIn('"event":"end"', response.text)

    def test_was_message_still_requires_bearer_authorization(self) -> None:
        response = self.client.post("/chat", json={"message": "질문"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "INVALID_AUTHORIZATION")

    def test_question_and_message_cannot_be_mixed(self) -> None:
        response = self.client.post(
            "/chat",
            json={"question": "코드서빙 질문", "message": "WAS 질문"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "AMBIGUOUS_CHAT_REQUEST")


if __name__ == "__main__":
    unittest.main()
