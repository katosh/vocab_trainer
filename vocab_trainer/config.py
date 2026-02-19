from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

DEFAULTS = {
    "llm_provider": "ollama",
    "llm_model": "qwen3:8b",
    "tts_provider": "edge-tts",
    "tts_voice": "en-US-GuyNeural",
    "elevenlabs_voice_id": "lfBVYbXnblkOddWFfEIg",
    "session_size": 20,
    "vocab_files": ["../vocabulary.md", "../vocabulary_distinctions.md", "../vocabulary_upper_intermediate_unknown.md"],
    "audio_cache_dir": "audio_cache",
    "db_path": "progress.db",
    "ollama_url": "http://localhost:11434",
    "min_ready_questions": 3,
    "max_active_words": 20,
    "archive_interval_days": 21,
}


@dataclass
class Settings:
    llm_provider: str = DEFAULTS["llm_provider"]
    llm_model: str = DEFAULTS["llm_model"]
    tts_provider: str = DEFAULTS["tts_provider"]
    tts_voice: str = DEFAULTS["tts_voice"]
    elevenlabs_voice_id: str = DEFAULTS["elevenlabs_voice_id"]
    session_size: int = DEFAULTS["session_size"]
    vocab_files: list[str] = field(default_factory=lambda: list(DEFAULTS["vocab_files"]))
    audio_cache_dir: str = DEFAULTS["audio_cache_dir"]
    db_path: str = DEFAULTS["db_path"]
    ollama_url: str = DEFAULTS["ollama_url"]
    min_ready_questions: int = DEFAULTS["min_ready_questions"]
    max_active_words: int = DEFAULTS["max_active_words"]
    archive_interval_days: int = DEFAULTS["archive_interval_days"]

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def db_full_path(self) -> Path:
        return self.project_root / self.db_path

    @property
    def audio_cache_full_path(self) -> Path:
        return self.project_root / self.audio_cache_dir

    def resolved_vocab_files(self) -> list[Path]:
        root = self.project_root
        return [root / f for f in self.vocab_files]

    def to_dict(self) -> dict:
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "tts_provider": self.tts_provider,
            "tts_voice": self.tts_voice,
            "elevenlabs_voice_id": self.elevenlabs_voice_id,
            "session_size": self.session_size,
            "vocab_files": self.vocab_files,
            "audio_cache_dir": self.audio_cache_dir,
            "db_path": self.db_path,
            "ollama_url": self.ollama_url,
            "min_ready_questions": self.min_ready_questions,
            "max_active_words": self.max_active_words,
            "archive_interval_days": self.archive_interval_days,
        }


def load_settings() -> Settings:
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text())
        known = {f.name for f in Settings.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known}
        return Settings(**filtered)
    return Settings()


def save_settings(settings: Settings) -> None:
    CONFIG_PATH.write_text(json.dumps(settings.to_dict(), indent=4) + "\n")
