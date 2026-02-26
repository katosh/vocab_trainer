# <img src="vocab_trainer/static/img/logo.png" width="36" height="36" align="top"> Wiseacre

A vocabulary trainer that teaches the precise differences between similar words. Instead of drilling isolated definitions, it groups near-synonyms and close alternatives into clusters and generates quiz scenarios that force you to distinguish between them — building an intuitive feel for which word fits which context.

## Learning Strategy

English is full of words that seem interchangeable until they aren't. *Mitigate* vs *alleviate* vs *ameliorate*. *Rebuke* vs *admonish* vs *chastise*. This trainer targets exactly that gap:

- **Synonym clusters** — Words are organized into groups of near-synonyms, each annotated with its specific shade of meaning and how it differs from the others.
- **LLM-generated questions** — Every question is freshly crafted by an LLM playing the role of an expert assessment writer. The prompts include the full cluster context — each word's meaning and its key distinction — so the LLM can construct scenarios where only one word genuinely fits. Because the LLM generates new material each time, you never see the same sentence twice for the same word, and the variety of contexts builds a richer, more intuitive understanding than memorizing fixed flashcards.
- **Three question types** — Fill-in-the-blank sentences (60%) test whether you can read contextual clues, best-fit scenarios (25%) describe a situation and ask you to pick the right word, and distinction questions (15%) directly probe your knowledge of what sets one near-synonym apart from the others. The mix ensures you learn words from multiple angles: reading comprehension, semantic matching, and explicit conceptual recall.
- **Spaced repetition (SM-2)** — Words you struggle with come back sooner; words you know well space out over days and weeks.

## AI Tutoring and Narration

After each answer, the trainer becomes an interactive tutor with built-in text-to-speech:

- **Auto-narrate** — After you answer, the AI automatically compares all four choices in flowing prose: why the correct word fits, exactly why each alternative fails, and when you'd use each one instead. The explanation streams in real time and is narrated aloud sentence by sentence as it arrives, so you hear the words in context and absorb pronunciation, stress patterns, and natural phrasing alongside the semantic distinctions.
- **Context on demand** — One-tap buttons generate example sentences for any word at three complexity levels (simple everyday usage, professional/academic, literary/sophisticated) or explore its etymology and historical roots. Every generated example is narrated aloud, letting you hear how the word sounds in different registers.
- **Free-form chat** — Ask the AI tutor anything about the current question: subtle connotations, register differences, common collocations, why your wrong answer almost worked, or anything else. Responses are narrated automatically.
- **Pausable narration** — All narration can be paused, resumed, or stopped mid-stream. TTS runs through multiple providers (free Edge TTS by default, or ElevenLabs/Piper for higher quality).

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

### Question Generation

Questions are generated by an LLM that receives the full cluster context — every word's meaning and key distinction — along with few-shot examples of high-quality assessment items. The LLM is prompted to write questions where context alone reveals the answer, with distractors that are genuinely tempting near-synonyms. Each question also includes a mini-lesson explanation and a context sentence used for TTS narration.

Because questions are generated fresh rather than drawn from a static bank, the same word appears in different scenarios across sessions, building robust understanding instead of pattern-matching to a single memorized sentence.

- **Fill in the Blank** (60%) — A polished prose sentence with one blank; 4 choices from the same cluster. The surrounding context is crafted to make only one word defensible.
- **Best Fit** (25%) — A scenario description ending with "which word best describes..."; tests shade of meaning by embedding the target word's distinction into the situation.
- **Distinction** (15%) — Direct probes like "which word specifically implies X?" with varied formats (scenarios, definitional contrasts, usage probes). Tests explicit conceptual recall.

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

[AGPL-3.0](LICENSE) — You're free to use, modify, and distribute this software, but any derivative work (including hosting it as a network service) must be released under the same license with source code available.
