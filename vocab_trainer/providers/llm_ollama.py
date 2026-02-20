from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from vocab_trainer.providers.base import LLMProvider

log = logging.getLogger("vocab_trainer.llm")


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen3:8b"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, temperature: float = 0.7, thinking: bool = True) -> str:
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
                    "think": thinking,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        elapsed = time.monotonic() - t0
        response = data["response"]
        tokens = data.get("eval_count", "?")
        log.info("── RESPONSE (%.1fs, %s tokens) ──\n%s", elapsed, tokens, response)
        return response

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, system: str | None = None, thinking: bool = True
    ) -> AsyncIterator[str]:
        """Stream tokens from Ollama, stripping <think>...</think> blocks."""
        log.info("── STREAM PROMPT (%s) ──\n%s", self.model, prompt)
        if system:
            log.info("── SYSTEM ──\n%s", system)
        t0 = time.monotonic()
        buf = ""
        in_think = False

        body: dict = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True,
            "think": thinking,
        }
        if system:
            body["system"] = system

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    token = data.get("response", "")
                    if not token:
                        continue

                    buf += token

                    # Filter <think>...</think> blocks from Qwen3
                    while True:
                        if in_think:
                            idx = buf.find("</think>")
                            if idx >= 0:
                                buf = buf[idx + 8:]
                                in_think = False
                            else:
                                buf = ""
                                break
                        else:
                            idx = buf.find("<think>")
                            if idx >= 0:
                                if idx > 0:
                                    yield buf[:idx]
                                buf = buf[idx + 7:]
                                in_think = True
                            else:
                                # Hold back last 6 chars for partial tag detection
                                if len(buf) > 6:
                                    yield buf[:-6]
                                    buf = buf[-6:]
                                break

        # Flush remaining buffer
        if buf and not in_think:
            yield buf

        elapsed = time.monotonic() - t0
        log.info("── STREAM COMPLETE (%.1fs) ──", elapsed)

    def name(self) -> str:
        return f"ollama/{self.model}"
