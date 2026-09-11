"""Code-serving revision test API."""

import os

import uvicorn
from fastapi import FastAPI


REVISION_MESSAGE = "코드서빙입니다 1"

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat():
    return {"code": 0, "data": {"text": REVISION_MESSAGE}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
