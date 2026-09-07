"""GenOS code-serving: question -> LLM -> {code, data: {text}}."""

import math
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def result(text: str, *, code: int = 0, status: int = 200) -> JSONResponse:
    return JSONResponse({"code": code, "data": {"text": text}}, status_code=status)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return result("[ERROR] invalid JSON", code=400, status=400)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        return result("[ERROR] body must be an object", code=400, status=400)

    question = body.get("question")
    if question is None or question == "":
        return result("[ERROR] question is empty")
    if not isinstance(question, str):
        return result("[ERROR] question must be a string", code=400, status=400)
    if question == "**verify**":
        return result("verified")
    if not question.strip():
        return result("[ERROR] question is empty")

    # Preserve the existing GenOS serving settings; inject secrets at deployment.
    token = os.getenv("GENOS_BEARER_TOKEN", "").strip()
    if not token or token == "<secret>":
        return result("[ERROR] GENOS_BEARER_TOKEN is required", code=503, status=503)
    try:
        serving_id = int(os.getenv("GENOS_SERVING_ID", "850"))
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        if serving_id <= 0 or not math.isfinite(timeout) or timeout <= 0 or max_tokens <= 0:
            raise ValueError
        base_url = os.getenv("GENOS_URL", "https://genos.genon.ai").strip().rstrip("/")
        model = os.getenv("GENOS_MODEL", "qwen/qwen3.7-flash").strip()
        if not model or not base_url.startswith(("https://", "http://")):
            raise ValueError
    except ValueError:
        return result("[ERROR] invalid LLM configuration", code=503, status=503)

    url = f"{base_url}/api/gateway/rep/serving/{serving_id}/v1/chat/completions"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": question}],
                    "stream": False,
                    "max_tokens": max_tokens,
                },
            )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError
    except httpx.TimeoutException:
        return result("[ERROR] LLM request timed out", code=504, status=504)
    except httpx.HTTPStatusError as exc:
        return result(
            f"[ERROR] LLM returned HTTP {exc.response.status_code}", code=502, status=502
        )
    except httpx.RequestError:
        return result("[ERROR] LLM connection failed", code=502, status=502)
    except httpx.InvalidURL:
        return result("[ERROR] invalid LLM URL", code=503, status=503)
    except (ValueError, KeyError, IndexError, TypeError):
        return result("[ERROR] invalid LLM response", code=502, status=502)
    return result(text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
