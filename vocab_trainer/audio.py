"""TTS audio caching and serving."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocab_trainer.db import Database
    from vocab_trainer.providers.base import TTSProvider


def sentence_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


async def get_or_create_audio(
    text: str,
    tts: TTSProvider,
    db: Database,
    cache_dir: Path,
) -> Path | None:
    """Get cached audio or generate new TTS audio."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = sentence_hash(text)

    # Check DB cache
    cached_path = db.get_audio_cache(h)
    if cached_path:
        p = Path(cached_path)
        if p.exists():
            return p

    # Generate new audio
    output_path = cache_dir / f"{h}.mp3"
    try:
        await tts.synthesize(text, output_path)
        db.set_audio_cache(h, str(output_path), tts.name())
        return output_path
    except Exception as e:
        print(f"TTS error: {e}")
        return None
