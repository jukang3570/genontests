"""GenOS 코드서빙 요청 확인용 최소 API.

어떤 경로·메서드로 요청이 오든 받은 요청 정보(헤더 포함)를 컨테이너 로그에
출력하고, 같은 내용을 응답 JSON으로 그대로 돌려준다.
"""

import json
import logging
import os
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, Request


# 토큰·쿠키 원문이 로그와 응답에 남지 않도록 앞뒤 일부와 길이만 보여준다.
# 원문이 꼭 필요하면 MASK_SENSITIVE_HEADERS=false로 실행한다.
SENSITIVE_HEADERS = {
    "authorization",
    "proxy-authorization",
    "cookie",
    "x-api-key",
    "api-key",
}
MASK_SENSITIVE = os.getenv("MASK_SENSITIVE_HEADERS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")

logger = logging.getLogger("request_probe")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def mask(value: str) -> str:
    """'Bearer abcdef...wxyz(len=40)' 형태로 값을 가린다."""

    scheme, _, token = value.partition(" ") if " " in value else ("", "", value)
    prefix = f"{scheme} " if scheme else ""
    if len(token) <= 12:
        return f"{prefix}***(len={len(token)})"
    return f"{prefix}{token[:6]}...{token[-4:]}(len={len(token)})"


def collect_headers(request: Request) -> dict[str, Any]:
    """모든 요청 헤더를 모은다. 같은 이름이 여러 번 오면 리스트로 묶는다."""

    headers: dict[str, Any] = {}
    for name, value in request.headers.items():
        if MASK_SENSITIVE and name in SENSITIVE_HEADERS:
            value = mask(value)
        if name not in headers:
            headers[name] = value
        elif isinstance(headers[name], list):
            headers[name].append(value)
        else:
            headers[name] = [headers[name], value]
    return headers


async def read_body(request: Request) -> Any:
    raw = await request.body()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", errors="replace")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def probe(request: Request, path: str) -> dict[str, Any]:
    client = request.client
    body = await read_body(request)
    info = {
        "method": request.method,
        "path": request.url.path,
        "query": request.url.query or None,
        "client": f"{client.host}:{client.port}" if client else None,
        "root_path": request.scope.get("root_path") or None,
        "header_count": len(request.headers.raw),
        "headers": collect_headers(request),
        "body": body,
    }
    logger.info("요청 수신\n%s", json.dumps(info, ensure_ascii=False, indent=2))

    # GenOS 배포 검증 요청에는 기존 에이전트와 같은 고정 응답을 준다.
    if isinstance(body, dict) and body.get("question") == "__verify__":
        return {"code": 0, "data": {"text": "verified"}}
    return {"code": 0, "data": {"text": "요청 수신 완료", "request": info}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
