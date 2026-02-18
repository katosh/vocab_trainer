from __future__ import annotations

import httpx

from vocab_trainer.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
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
            return resp.json()["response"]

    def name(self) -> str:
        return f"ollama/{self.model}"
