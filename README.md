# Vocab Trainer

A vocabulary trainer that teaches the precise differences between similar words. Instead of drilling isolated definitions, it groups near-synonyms and close alternatives into clusters and generates quiz scenarios that force you to distinguish between them — building an intuitive feel for which word fits which context.

## Learning Strategy

English is full of words that seem interchangeable until they aren't. *Mitigate* vs *alleviate* vs *ameliorate*. *Rebuke* vs *admonish* vs *chastise*. This trainer targets exactly that gap:

- **Synonym clusters** — Words are organized into groups of near-synonyms, each annotated with its specific shade of meaning and how it differs from the others.
- **Contextual quizzes** — An LLM generates scenarios where only one word in the cluster is the right fit, teaching you to recognize the precise contexts where each word belongs.
- **Three question types** — Fill-in-the-blank sentences (60%), best-fit scenarios (25%), and explicit distinction questions (15%), each testing a different facet of word knowledge.
- **Spaced repetition (SM-2)** — Words you struggle with come back sooner; words you know well space out over days and weeks.

## AI Tutoring

After each answer, the trainer becomes an interactive tutor:

- **Auto-compare** — Automatically asks the AI to compare all four answer choices, explaining when you'd use each one and giving example sentences. Streams the response in real time.
- **Automated narration** — TTS reads the completed sentence aloud and narrates the AI's explanations, so you absorb the material by listening as well as reading.
- **Context on demand** — One-click prompts generate example sentences at different complexity levels (simple, intermediate, advanced, literary) or explore a word's etymology and history.
- **Free-form chat** — Ask the AI anything about the words in the current question: subtle connotations, register differences, common collocations, or anything else.

## Prerequisites

- **Python 3.10+**
- **Ollama** — local LLM inference (or set an API key for Anthropic/OpenAI instead)

[uv](https://docs.astral.sh/uv/) is the recommended way to run the project — it handles the virtualenv and installs automatically. All commands below use `uv run`, which can be dropped if you install manually into any virtualenv:

```bash
pip install -e .               # then use "python -m vocab_trainer ..." instead of "uv run python -m vocab_trainer ..."
```

## Quick Start

```bash
cd vocab_trainer

# 1. Start Ollama + pull a model (one-time setup)
brew install ollama          # macOS
brew services start ollama   # run as background service
ollama pull qwen3:8b         # ~5 GB download

# 2. Launch the web UI (auto-imports vocabulary on first start)
uv run python -m vocab_trainer serve
# Open http://localhost:8765
```

The SQLite database and all tables are created automatically on first run. The bundled vocabulary files in `data/` are imported automatically on startup. You can also pre-generate questions from the Dashboard or via CLI:

```bash
uv run python -m vocab_trainer generate --count 20
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `uv run python -m vocab_trainer serve [--port PORT]` | Launch web app (default port 8765) |
| `uv run python -m vocab_trainer serve --no-auto-import` | Launch without re-importing changed vocab files |
| `uv run python -m vocab_trainer import` | Manually import vocabulary files into SQLite |
| `uv run python -m vocab_trainer generate --count N` | Pre-generate N questions using the configured LLM |
| `uv run python -m vocab_trainer stats` | Print progress summary to terminal |

## Vocabulary Data

Bundled vocabulary files live in `data/` and are auto-imported on startup when they change. When `vocab_files` in `config.json` is empty (the default), all `*.md` files in `data/` are discovered automatically. Set explicit paths to use files from elsewhere.

The parser recognizes two markdown table formats:

**Word lists** — 2-column tables (`Word | Definition`) under any heading. Each heading becomes a category. Words are imported individually for general vocabulary building.

```markdown
## Cognition and Insight

| Word | Definition |
|------|------------|
| **perspicacious** | having keen insight; perceptive |
| **sagacious** | having keen practical wisdom and judgment |
```

**Distinction clusters** — 3-column tables (`Word | Meaning | Distinction`) under a cluster heading. These form the synonym groups that drive quiz generation. The heading names the cluster, and each row explains how that word differs from the others.

```markdown
## Criticizing Someone

| Word | What it really means | Key distinction |
|------|---------------------|-----------------|
| **admonish** | warn or counsel firmly but gently | corrective, not punitive |
| **rebuke** | criticize sharply and sternly | a sudden, direct verbal blow |
```

See [`data/README.md`](data/README.md) for more details on editing or replacing the bundled files.

## How It Works

### Question Types

Questions are generated by an LLM from the distinction clusters:

- **Fill in the Blank** (60%) — A context sentence with one blank; 4 choices from the same cluster. Tests contextual fitness.
- **Best Fit** (25%) — A short scenario; pick the word that best applies. Tests shade of meaning.
- **Distinction** (15%) — "Which word specifically implies X?" Tests explicit knowledge of near-synonym differences.

### Spaced Repetition (SM-2)

Each word tracks an easiness factor, interval, and repetition count. After each answer:

- **Correct**: interval grows (1d -> 6d -> interval * EF)
- **Wrong**: interval resets to 1 day
- Response time maps to a quality score (instant = 5, slow = 3, wrong = 1)

Sessions prioritize overdue words first, then introduce new words up to a configurable limit.

### Keyboard Shortcuts

During a quiz session:

- **A/B/C/D** or **1/2/3/4** — Select an answer
- **Enter** or **Space** — Next question (after answer reveal)

## Architecture

```
vocab_trainer/
├── pyproject.toml
├── config.json                  # Provider settings, session params
├── data/                        # Bundled vocabulary files (auto-imported)
├── vocab_trainer/
│   ├── __main__.py              # CLI entry point
│   ├── app.py                   # FastAPI routes + session management
│   ├── config.py                # Settings dataclass, load/save config.json
│   ├── models.py                # Core dataclasses (VocabWord, Question, etc.)
│   ├── db.py                    # SQLite schema + CRUD
│   ├── srs.py                   # SM-2 spaced repetition
│   ├── question_generator.py    # LLM orchestration + JSON validation
│   ├── prompts.py               # Prompt templates per question type
│   ├── audio.py                 # TTS caching (hash-based)
│   ├── parsers/
│   │   ├── vocabulary_parser.py       # Parse vocabulary.md
│   │   └── distinctions_parser.py     # Parse vocabulary_distinctions.md
│   ├── providers/
│   │   ├── base.py              # ABCs: LLMProvider, TTSProvider
│   │   ├── llm_ollama.py        # Ollama (Qwen3) via httpx
│   │   ├── llm_anthropic.py     # Claude API
│   │   ├── llm_openai.py        # OpenAI API
│   │   ├── tts_edge.py          # edge-tts (free, default)
│   │   ├── tts_elevenlabs.py    # ElevenLabs API
│   │   └── tts_piper.py         # Piper (fully offline)
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── app.js
├── audio_cache/                 # Generated TTS files (gitignored)
└── progress.db                  # SQLite database (gitignored)
```

## Provider Configuration

All providers are swappable via `config.json` or the Settings page in the web UI.

### LLM Providers

| Provider | Config value | Requirements |
|----------|-------------|--------------|
| Ollama (default) | `"ollama"` | `brew services start ollama`, model pulled |
| Anthropic | `"anthropic"` | `ANTHROPIC_API_KEY` env var |
| OpenAI | `"openai"` | `OPENAI_API_KEY` env var |

### TTS Providers

| Provider | Config value | Requirements |
|----------|-------------|--------------|
| Edge TTS (default) | `"edge-tts"` | None (free Microsoft voices) |
| ElevenLabs | `"elevenlabs"` | `ELEVEN_LABS_API_KEY` env var |
| Piper | `"piper"` | `piper` binary on PATH |

## Running Tests

```bash
uv run --extra all python -m pytest tests/ -v
```

## License

[MIT](LICENSE)
