from __future__ import annotations

import os

from vocab_trainer.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        self.model = model

    async def generate(self, prompt: str, temperature: float = 0.7, thinking: bool = True) -> str:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def name(self) -> str:
        return f"anthropic/{self.model}"
