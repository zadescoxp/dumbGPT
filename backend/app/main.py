from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import LocalModel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
app = FastAPI(title="dumbGPT API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
model = LocalModel()
last_message_at: dict[str, float] = defaultdict(float)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    chat_id: str = Field(min_length=1, max_length=100)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=1000)


model = LocalModel()
last_message_at: dict[str, float] = defaultdict(float)


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "model": model.status()}


@app.post("/v1/chat")
def chat(payload: ChatRequest, request: Request, x_user_id: Optional[str] = Header(default=None)) -> dict[str, str]:
    user_id = (x_user_id or (request.client.host if request.client else "anonymous"))[:100]
    now = time.monotonic()
    if now - last_message_at[user_id] < 1.0:
        raise HTTPException(status_code=429, detail="Please wait one second between messages")
    last_message_at[user_id] = now
    try:
        answer = model.generate(
            payload.message,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"chat_id": payload.chat_id, "message": answer}
