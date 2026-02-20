from __future__ import annotations

import os

from vocab_trainer.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        import openai
        self.client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        self.model = model

    async def generate(self, prompt: str, temperature: float = 0.7, thinking: bool = True) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def name(self) -> str:
        return f"openai/{self.model}"
