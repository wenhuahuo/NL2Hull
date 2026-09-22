"""OpenRouter client for the TypeSafe Jev decision API."""

from __future__ import annotations

import os
from typing import Any

import requests


JEV_MODEL = "~typesafe/jev-latest"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"


class JevClient:
    def __init__(self, *, api_key: str | None = None, timeout: int = 180) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for Jev evaluation")
        self.timeout = timeout

    def decide(self, state: str, questions: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            JEV_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "NURBS Ship FFD Benchmark",
            },
            json={
                "model": JEV_MODEL,
                "state": state,
                "questions": questions,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise RuntimeError("Jev response has no answers object")
        return {
            "answers": answers,
            "model": payload.get("model"),
            "usage": payload.get("usage"),
            "request_id": payload.get("id"),
        }
