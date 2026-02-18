from __future__ import annotations

from pathlib import Path

from vocab_trainer.providers.base import TTSProvider


class EdgeTTSProvider(TTSProvider):
    def __init__(self, voice: str = "en-US-GuyNeural"):
        self.voice = voice

    async def synthesize(self, text: str, output_path: Path) -> Path:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))
        return output_path

    def name(self) -> str:
        return f"edge-tts/{self.voice}"
