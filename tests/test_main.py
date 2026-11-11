import unittest

from fastapi.testclient import TestClient

from main import app


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.json(), {"status": "ok"})

    def test_verify(self) -> None:
        response = self.client.post("/chat", json={"question": "__verify__"})
        self.assertEqual(response.json(), {"code": 0, "data": {"text": "verified"}})

    def test_echoes_headers_and_masks_authorization(self) -> None:
        response = self.client.post(
            "/predict?x=1",
            json={"question": "hi"},
            headers={
                "Authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
                "X-Test": "hello",
            },
        )
        request = response.json()["data"]["request"]
        self.assertEqual(request["path"], "/predict")
        self.assertEqual(request["query"], "x=1")
        self.assertEqual(request["headers"]["x-test"], "hello")
        self.assertEqual(request["headers"]["authorization"], "Bearer abcdef...wxyz(len=26)")
        self.assertEqual(request["body"], {"question": "hi"})


if __name__ == "__main__":
    unittest.main()
