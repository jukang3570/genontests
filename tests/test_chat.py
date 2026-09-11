import asyncio
import unittest

from main import chat


class ChatTests(unittest.TestCase):
    def test_revision_one_response(self):
        self.assertEqual(
            asyncio.run(chat()),
            {"code": 0, "data": {"text": "코드서빙 리비전 버전 1입니다."}},
        )


if __name__ == "__main__":
    unittest.main()
