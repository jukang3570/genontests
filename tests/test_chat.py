import json
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from main import app


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GENOS_BEARER_TOKEN": "EXAMPLE_TEST_TOKEN"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def mock_llm(self, handler):
        real_client = httpx.AsyncClient
        return patch("main.httpx.AsyncClient", side_effect=lambda **kwargs: real_client(
            transport=httpx.MockTransport(handler), **kwargs
        ))

    def test_health_and_verify_do_not_call_llm(self):
        with patch.dict(os.environ, {"GENOS_BEARER_TOKEN": ""}), patch("main.httpx.AsyncClient") as llm:
            self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
            self.assertEqual(self.client.post("/chat", json={"question": "**verify**"}).json(),
                             {"code": 0, "data": {"text": "verified"}})
            llm.assert_not_called()
        self.assertEqual({route.path for route in app.routes}, {"/health", "/chat"})

    def test_empty_question_matches_sample(self):
        for body in ({}, {"question": None}, {"question": ""}, {"question": "   "}):
            with self.subTest(body=body):
                response = self.client.post("/chat", json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"code": 0, "data": {"text": "[ERROR] question is empty"}})
        self.assertEqual(self.client.post("/chat", content="null").json()["code"], 0)

    def test_invalid_input(self):
        for body in ([], "text", {"question": 12}, {"question": False}):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/chat", json=body).status_code, 400)
        self.assertEqual(self.client.post("/chat", content="{").status_code, 400)

    def test_llm_contract(self):
        calls = []

        def handler(request):
            calls.append(request)
            self.assertEqual(str(request.url), "https://genos.genon.ai/api/gateway/rep/serving/850/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer EXAMPLE_TEST_TOKEN")
            payload = json.loads(request.content)
            self.assertEqual(payload, {"model": "qwen/qwen3.7-flash", "messages": [
                {"role": "user", "content": "안녕하세요"}
            ], "stream": False, "max_tokens": 4096})
            return httpx.Response(200, json={"choices": [{"message": {"content": "안녕하세요!"}}]})

        with self.mock_llm(handler):
            response = self.client.post("/chat", json={"question": "안녕하세요"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"code": 0, "data": {"text": "안녕하세요!"}})
        self.assertEqual(len(calls), 1)

    def test_missing_or_invalid_configuration(self):
        for config in ({"GENOS_BEARER_TOKEN": ""}, {"LLM_TIMEOUT_SECONDS": "nan"},
                       {"GENOS_SERVING_ID": "bad"}, {"LLM_MAX_TOKENS": "0"}):
            with self.subTest(config=config), patch.dict(os.environ, config), patch("main.httpx.AsyncClient") as llm:
                self.assertEqual(self.client.post("/chat", json={"question": "hello"}).status_code, 503)
                llm.assert_not_called()

    def test_upstream_http_error_is_sanitized(self):
        for status in (401, 429, 500):
            with self.subTest(status=status), self.mock_llm(lambda request: httpx.Response(status, text="EXAMPLE_PRIVATE_ERROR")):
                response = self.client.post("/chat", json={"question": "hello"})
            self.assertEqual(response.status_code, 502)
            self.assertEqual(response.json()["code"], 502)
            self.assertNotIn("EXAMPLE_PRIVATE_ERROR", response.text)

    def test_network_failures(self):
        for error, status in ((httpx.ReadTimeout, 504), (httpx.ConnectError, 502)):
            def handler(request):
                raise error("EXAMPLE_PRIVATE_ERROR", request=request)
            with self.subTest(error=error), self.mock_llm(handler):
                response = self.client.post("/chat", json={"question": "hello"})
            self.assertEqual(response.status_code, status)
            self.assertEqual(response.json()["code"], status)
            self.assertNotIn("EXAMPLE_PRIVATE_ERROR", response.text)

    def test_invalid_llm_responses(self):
        for payload in ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]},
                        {"choices": [{"message": {"content": " "}}]}, []):
            with self.subTest(payload=payload), self.mock_llm(lambda request: httpx.Response(200, json=payload)):
                self.assertEqual(self.client.post("/chat", json={"question": "hello"}).status_code, 502)
        with self.mock_llm(lambda request: httpx.Response(200, text="not JSON")):
            self.assertEqual(self.client.post("/chat", json={"question": "hello"}).status_code, 502)


if __name__ == "__main__":
    unittest.main()
