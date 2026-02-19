from __future__ import annotations

import logging
import time

import httpx

from vocab_trainer.providers.base import LLMProvider

log = logging.getLogger("vocab_trainer.llm")


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        log.info("── PROMPT (%s) ──\n%s", self.model, prompt)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        elapsed = time.monotonic() - t0
        response = data["response"]
        tokens = data.get("eval_count", "?")
        log.info("── RESPONSE (%.1fs, %s tokens) ──\n%s", elapsed, tokens, response)
        return response

    def name(self) -> str:
        return f"ollama/{self.model}"
