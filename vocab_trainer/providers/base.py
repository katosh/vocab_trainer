from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        ...

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
