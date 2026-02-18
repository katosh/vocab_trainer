from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from vocab_trainer.providers.base import TTSProvider


class PiperTTSProvider(TTSProvider):
    def __init__(self, model: str = "en_US-lessac-medium"):
        self.model = model
        if not shutil.which("piper"):
            raise RuntimeError(
                "Piper not found. Install from https://github.com/rhasspy/piper"
            )

    async def synthesize(self, text: str, output_path: Path) -> Path:
        wav_path = output_path.with_suffix(".wav")
        proc = await asyncio.create_subprocess_exec(
            "piper",
            "--model", self.model,
            "--output_file", str(wav_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate(input=text.encode())
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed with code {proc.returncode}")

        # Convert WAV to MP3 via ffmpeg
        proc2 = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            str(output_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc2.communicate()
        wav_path.unlink(missing_ok=True)
        return output_path

    def name(self) -> str:
        return f"piper/{self.model}"
