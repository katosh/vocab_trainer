from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7, thinking: bool = True) -> str:
        ...

    async def generate_stream(
        self, prompt: str, temperature: float = 0.7, system: str | None = None, thinking: bool = True
    ) -> AsyncIterator[str]:
        """Stream response tokens. Default: yield full response at once."""
        result = await self.generate(prompt, temperature, thinking=thinking)
        yield result

    @abstractmethod
    def name(self) -> str:
        ...


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str, output_path: Path) -> Path:
        ...

    @abstractmethod
    def name(self) -> str:
        ...
