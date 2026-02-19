# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run tests (requires --extra all for optional provider deps)
uv run --extra all python -m pytest tests/ -v

# Run a single test file or test
uv run --extra all python -m pytest tests/test_db.py -v
uv run --extra all python -m pytest tests/test_app.py::test_session_flow -v

# Start the web server (default port 8765)
uv run python -m vocab_trainer serve

# Start without auto-importing vocabulary on startup
uv run python -m vocab_trainer serve --no-auto-import

# Import vocabulary from markdown files into SQLite
uv run python -m vocab_trainer import

# Pre-generate questions via LLM
uv run python -m vocab_trainer generate --count 20
```

## Architecture

FastAPI backend + vanilla JS SPA + SQLite database. No build step for frontend.

### Backend (`vocab_trainer/`)

- **app.py** — FastAPI server with all API routes and session management (~830 lines). Houses global state (DB connection, active sessions dict) and background task orchestration.
- **db.py** — SQLite CRUD layer with WAL mode. Schema creation and migrations in `_migrate()`. Tables: `words`, `clusters`, `cluster_words`, `questions`, `reviews`, `sessions`, `audio_cache`, `file_mtimes`.
- **question_generator.py** — LLM orchestration: builds prompts, calls provider, extracts/validates JSON response. Three question types: `fill_blank` (60%), `best_fit` (25%), `distinction` (15%). Retries up to 3 times per question.
- **srs.py** — SM-2 spaced repetition algorithm. Quality scores derived from correctness + response time.
- **config.py** — `Settings` dataclass, reads/writes `config.json`.
- **models.py** — Core dataclasses: `VocabWord`, `Question`, `DistinctionCluster`, `DistinctionEntry`.
- **prompts.py** — Prompt templates per question type.
- **audio.py** — TTS caching system (SHA256 hash of sentence text).

### Provider Pattern (`vocab_trainer/providers/`)

ABC base classes in `base.py` (`LLMProvider`, `TTSProvider`). Implementations:
- LLM: `llm_ollama.py` (default, local Qwen3), `llm_anthropic.py`, `llm_openai.py`
- TTS: `tts_edge.py` (default, free), `tts_elevenlabs.py`, `tts_piper.py`

Providers are instantiated via factory functions `_get_llm()` / `_get_tts()` in `app.py` based on `config.json`.

### Parsers (`vocab_trainer/parsers/`)

Custom regex-based markdown parsers for two formats:
- `vocabulary_parser.py` — 2-column tables (word | definition) from `vocabulary.md`
- `distinctions_parser.py` — 3-column tables (word | meaning | distinction) grouped into clusters from `vocabulary_distinctions.md`

### Frontend (`vocab_trainer/static/`)

Single-page app: `index.html` (shell), `app.js` (~1400 lines, all view logic), `style.css`. Views: Dashboard, Quiz, Library, Settings. Served as static files by FastAPI.

### Session Pipeline

Critical flow in `app.py`:
1. `/api/session/start` — pools questions (due reviews → new words → reinforcement → pending generation), spawns background `_pregenerate_session_questions()` task
2. `/api/session/next` — polls for pending generation completion (up to 90s), shuffles choices
3. `/api/session/answer` — records result, updates SM-2, generates TTS, auto-archives mastered questions

Pre-generation runs **sequentially** (one LLM call at a time) to prevent GPU resource contention. The main question-serving path never calls the LLM directly.

## Testing

pytest with `pytest-asyncio` (async mode: `auto`). Key fixtures in `tests/conftest.py`:
- `tmp_db` — fresh temporary SQLite database
- `populated_db` — DB pre-loaded with sample words and clusters
- `sample_words`, `sample_cluster`, `sample_question` — test data objects

App tests (`test_app.py`) use FastAPI's `TestClient` with a test lifespan that creates an isolated temp DB.

## Key Conventions

- Python 3.10+, async throughout (FastAPI async routes, async providers)
- `uv` for dependency management (not pip). Optional deps grouped as extras: `elevenlabs`, `anthropic`, `openai`, `all`, `test`
- SQLite with WAL mode; parameterized queries for all DB operations
- Config lives in `config.json` at project root; `Settings` dataclass enforces defaults
- Database file (`progress.db`) and `audio_cache/` are gitignored
- Bundled vocabulary files live in `data/` and are auto-discovered when `vocab_files` is empty in config
