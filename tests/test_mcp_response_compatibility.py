"""운영계 MCP structuredContent 응답 계약 테스트."""

import unittest

import httpx

from app.mcp.client import _extract_mcp_structured_content, _parse_mcp_response


class McpResponseCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [
            {
                "objId": "artNm",
                "objType": "String",
                "objNm": "유치자명",
                "objVal": "홍길동",
            },
            {
                "objId": "amount",
                "objType": "Number",
                "objNm": "금액",
                "objVal": 10000,
            },
        ]

    def test_extracts_production_structured_content(self) -> None:
        response = httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "jsonrpc": "2.0",
                "result": {
                    "structuredContent": {"data": self.items},
                    "isError": False,
                },
            },
        )

        envelope = _parse_mcp_response(response)
        normalized = _extract_mcp_structured_content(envelope["result"])

        self.assertEqual(normalized, {"data": self.items})

    def test_rejects_development_content_text_fallback(self) -> None:
        result = {
            "content": {"text": '[{"objId":"artNm","objVal":"홍길동"}]'},
            "isError": False,
        }

        self.assertIsNone(_extract_mcp_structured_content(result))

    def test_requires_structured_content_to_be_an_object(self) -> None:
        self.assertIsNone(
            _extract_mcp_structured_content(
                {"structuredContent": [{"data": self.items}]}
            )
        )


if __name__ == "__main__":
    unittest.main()
