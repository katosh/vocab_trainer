from __future__ import annotations

import os
from pathlib import Path

from vocab_trainer.providers.base import TTSProvider


class ElevenLabsProvider(TTSProvider):
    def __init__(self, voice_id: str = "lfBVYbXnblkOddWFfEIg"):
        from elevenlabs import ElevenLabs
        self.client = ElevenLabs(
            api_key=os.environ.get("ELEVEN_LABS_API_KEY", ""),
        )
        self.voice_id = voice_id

    async def synthesize(self, text: str, output_path: Path) -> Path:
        from elevenlabs.types import VoiceSettings
        import asyncio

        def _generate():
            audio = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_v3",
                output_format="mp3_44100_128",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.3,
                    speed=0.95,
                ),
            )
            # audio is a generator of bytes
            with open(output_path, "wb") as f:
                for chunk in audio:
                    f.write(chunk)
            return output_path

        return await asyncio.get_event_loop().run_in_executor(None, _generate)

    def name(self) -> str:
        return f"elevenlabs/{self.voice_id}"
