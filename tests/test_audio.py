"""Tests for audio caching."""
from __future__ import annotations

from pathlib import Path

import pytest

from vocab_trainer.audio import get_or_create_audio, sentence_hash


class FakeTTS:
    """Simple fake TTS that avoids AsyncMock's `name` attribute issue."""

    def __init__(self, side_effect=None):
        self._side_effect = side_effect
        self.synthesize_called = 0

    async def synthesize(self, text: str, output_path: Path) -> Path:
        self.synthesize_called += 1
        if self._side_effect:
            if isinstance(self._side_effect, Exception):
                raise self._side_effect
            return await self._side_effect(text, output_path)
        output_path.write_bytes(b"fake mp3 data")
        return output_path

    def name(self) -> str:
        return "fake-tts"


class TestSentenceHash:
    def test_deterministic(self):
        h1 = sentence_hash("Hello world")
        h2 = sentence_hash("Hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = sentence_hash("Hello world")
        h2 = sentence_hash("Goodbye world")
        assert h1 != h2

    def test_hash_length(self):
        h = sentence_hash("test")
        assert len(h) == 16

    def test_hex_chars_only(self):
        h = sentence_hash("any text here")
        assert all(c in "0123456789abcdef" for c in h)


class TestGetOrCreateAudio:
    @pytest.mark.asyncio
    async def test_creates_new_audio(self, tmp_db, tmp_path):
        cache_dir = tmp_path / "audio"
        tts = FakeTTS()

        result = await get_or_create_audio("Hello world", tts, tmp_db, cache_dir)

        assert result is not None
        assert result.exists()
        assert result.suffix == ".mp3"
        assert tts.synthesize_called == 1

    @pytest.mark.asyncio
    async def test_returns_cached(self, tmp_db, tmp_path):
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        h = sentence_hash("Hello world")
        cached_file = cache_dir / f"{h}.mp3"
        cached_file.write_bytes(b"cached data")
        tmp_db.set_audio_cache(h, str(cached_file), "mock-tts")

        tts = FakeTTS()
        result = await get_or_create_audio("Hello world", tts, tmp_db, cache_dir)

        assert result == cached_file
        assert tts.synthesize_called == 0

    @pytest.mark.asyncio
    async def test_regenerates_if_file_missing(self, tmp_db, tmp_path):
        cache_dir = tmp_path / "audio"
        h = sentence_hash("Hello world")
        # DB says it's cached, but file doesn't exist
        tmp_db.set_audio_cache(h, str(tmp_path / "nonexistent.mp3"), "mock-tts")

        tts = FakeTTS()
        result = await get_or_create_audio("Hello world", tts, tmp_db, cache_dir)

        assert result is not None
        assert result.exists()
        assert tts.synthesize_called == 1

    @pytest.mark.asyncio
    async def test_tts_failure_returns_none(self, tmp_db, tmp_path):
        cache_dir = tmp_path / "audio"
        tts = FakeTTS(side_effect=RuntimeError("TTS unavailable"))

        result = await get_or_create_audio("Hello world", tts, tmp_db, cache_dir)
        assert result is None
