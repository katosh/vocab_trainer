"""FastAPI application with all routes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vocab_trainer.audio import get_or_create_audio, sentence_hash
from vocab_trainer.config import Settings, load_settings, save_settings
from vocab_trainer.db import Database
from vocab_trainer.models import Question
from vocab_trainer.parsers.distinctions_parser import parse_distinctions_file
from vocab_trainer.parsers.vocabulary_parser import parse_vocabulary_file
from vocab_trainer.question_generator import generate_batch, generate_question
from vocab_trainer.srs import quality_from_answer, record_review, select_session_words

app = FastAPI(title="Vocab Trainer")

# Global state (initialized in lifespan)
_db: Database | None = None
_settings: Settings | None = None
_active_sessions: dict[int, dict] = {}  # session_id -> session state


def get_db() -> Database:
    assert _db is not None
    return _db


def get_settings() -> Settings:
    assert _settings is not None
    return _settings


def _get_llm():
    s = get_settings()
    if s.llm_provider == "ollama":
        from vocab_trainer.providers.llm_ollama import OllamaProvider
        return OllamaProvider(base_url=s.ollama_url, model=s.llm_model)
    elif s.llm_provider == "anthropic":
        from vocab_trainer.providers.llm_anthropic import AnthropicProvider
        return AnthropicProvider()
    elif s.llm_provider == "openai":
        from vocab_trainer.providers.llm_openai import OpenAIProvider
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")


def _get_tts():
    s = get_settings()
    if s.tts_provider == "edge-tts":
        from vocab_trainer.providers.tts_edge import EdgeTTSProvider
        return EdgeTTSProvider(voice=s.tts_voice)
    elif s.tts_provider == "elevenlabs":
        from vocab_trainer.providers.tts_elevenlabs import ElevenLabsProvider
        return ElevenLabsProvider(voice_id=s.tts_voice, model_id=s.elevenlabs_model)
    elif s.tts_provider == "piper":
        from vocab_trainer.providers.tts_piper import PiperTTSProvider
        return PiperTTSProvider(model=s.tts_voice)
    raise ValueError(f"Unknown TTS provider: {s.tts_provider}")


_bg_generating = False
_bg_log = logging.getLogger("vocab_trainer.bg")

# Background LLM tasks — tracked so chat can cancel them for GPU priority.
_bg_tasks: set[asyncio.Task] = set()


async def _ensure_question_buffer():
    """Kick off background generation if the non-archived bank is too low."""
    global _bg_generating
    if _bg_generating:
        _bg_log.debug("Buffer check: skipped (generation already running)")
        return
    db = get_db()
    s = get_settings()
    ready = db.get_ready_question_count()
    _bg_log.info("Buffer check: %d ready, %d required", ready, s.min_ready_questions)
    if ready >= s.min_ready_questions:
        return
    need = s.min_ready_questions - ready
    _bg_generating = True
    task = asyncio.create_task(_generate_in_background(need))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _generate_in_background(count: int):
    global _bg_generating
    try:
        _bg_log.info("Background generation: creating %d questions", count)
        db = get_db()
        llm = _get_llm()
        questions = await generate_batch(llm, db, count=count)
        _bg_log.info(
            "Background generation: created %d questions, bank now %d ready",
            len(questions), db.get_ready_question_count(),
        )
    except asyncio.CancelledError:
        _bg_log.info("Background generation cancelled (chat priority)")
    except Exception as e:
        _bg_log.warning("Background generation failed: %s", e)
    finally:
        _bg_generating = False
        try:
            await _ensure_question_buffer()
        except Exception:
            pass  # DB may be closed during shutdown


_pregen_log = logging.getLogger("vocab_trainer.pregen")


async def _pregenerate_session_questions(session_id: int, pending_indices: list[int]):
    """Generate pending session questions sequentially in queue order.

    pending_indices is already sorted by session position, so the
    next question the user will need is always generated first.
    Only this task calls the LLM — _get_next_question just waits.
    Cancelled when a chat request needs GPU priority; the remaining
    indices are re-queued on resume.
    """
    llm = _get_llm()
    db = get_db()

    try:
        for idx in pending_indices:
            session = _active_sessions.get(session_id)
            if session is None:
                _pregen_log.info("Session %d gone, stopping pre-generation", session_id)
                return

            q_data = session["questions"][idx]
            if "pending_cluster" not in q_data:
                continue  # Already handled

            cluster, word_info = q_data.pop("pending_cluster"), q_data.pop("pending_word")
            q_data["generating"] = True
            try:
                q = await generate_question(
                    llm, db,
                    cluster=cluster,
                    target_word_info=word_info,
                )
                if q:
                    db.save_question(q)
                    session["questions"][idx] = {
                        "id": q.id,
                        "question_type": q.question_type,
                        "stem": q.stem,
                        "choices_json": json.dumps(q.choices),
                        "correct_index": q.correct_index,
                        "correct_word": q.correct_word,
                        "explanation": q.explanation,
                        "context_sentence": q.context_sentence,
                        "cluster_title": q.cluster_title,
                        "choice_explanations_json": json.dumps(q.choice_explanations),
                        "choice_details_json": json.dumps(q.choice_details),
                    }
                    _pregen_log.info("Pre-generated question %d/%d for session %d",
                                     idx + 1, len(session["questions"]), session_id)
                else:
                    q_data.pop("generating", None)
                    q_data["generation_failed"] = True
                    _pregen_log.warning("Pre-generation returned None for index %d", idx)
            except asyncio.CancelledError:
                # Put the pending data back so it can be retried
                q_data.pop("generating", None)
                q_data["pending_cluster"] = cluster
                q_data["pending_word"] = word_info
                raise
            except Exception as e:
                _pregen_log.warning("Pre-generation failed for index %d: %s", idx, e)
                q_data.pop("generating", None)
                q_data["generation_failed"] = True
    except asyncio.CancelledError:
        _pregen_log.info("Session %d pre-generation cancelled (chat priority)", session_id)
        # Re-queue remaining pending indices after chat completes
        remaining = [i for i in pending_indices if "pending_cluster" in
                     (_active_sessions.get(session_id, {}).get("questions", [{}] * (i + 1))[i])]
        if remaining and _active_sessions.get(session_id):
            _pregen_log.info("Will resume %d pending questions after chat", len(remaining))
            _active_sessions[session_id]["_resume_indices"] = remaining


def _auto_import_if_changed(db: Database, settings: Settings) -> None:
    """Re-import vocab files whose mtime has changed since the last import."""
    log = logging.getLogger("auto-import")
    for vf in settings.resolved_vocab_files():
        if not vf.exists():
            continue
        current_mtime = vf.stat().st_mtime_ns
        stored_mtime = db.get_file_mtime(str(vf))
        if stored_mtime == current_mtime:
            continue
        log.info("Changed: %s — re-importing", vf.name)
        if "distinctions" in vf.name:
            db.delete_clusters_by_source(vf.name)
            clusters = parse_distinctions_file(vf)
            n = db.import_clusters(clusters)
            log.info("  %d clusters imported", n)
        else:
            db.delete_words_by_source(vf.name)
            words = parse_vocabulary_file(vf)
            n = db.import_words(words)
            log.info("  %d words imported", n)
        db.set_file_mtime(str(vf), current_mtime)


@app.on_event("startup")
async def startup():
    global _db, _settings
    if _db is not None:
        return  # Already initialized (e.g. by tests)
    _settings = load_settings()
    _db = Database(_settings.db_full_path)
    if not os.environ.get("VOCAB_TRAINER_NO_AUTO_IMPORT"):
        _auto_import_if_changed(_db, _settings)
    await _ensure_question_buffer()


@app.on_event("shutdown")
async def shutdown():
    if _db:
        _db.close()


# ── Static files ──────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/style.css")
async def style():
    return FileResponse(static_dir / "style.css", media_type="text/css")


@app.get("/app.js")
async def script():
    return FileResponse(static_dir / "app.js", media_type="application/javascript")


# ── API: Stats ────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    return get_db().get_stats()


# ── API: Import ───────────────────────────────────────────────────────────

@app.post("/api/import")
async def api_import():
    db = get_db()
    s = get_settings()

    total_words = 0
    total_clusters = 0

    for vf in s.resolved_vocab_files():
        if not vf.exists():
            continue
        if "distinctions" in vf.name:
            db.delete_clusters_by_source(vf.name)
            clusters = parse_distinctions_file(vf)
            total_clusters += db.import_clusters(clusters)
        else:
            db.delete_words_by_source(vf.name)
            words = parse_vocabulary_file(vf)
            total_words += db.import_words(words)
        db.set_file_mtime(str(vf), vf.stat().st_mtime_ns)

    return {
        "words_imported": total_words,
        "clusters_imported": total_clusters,
        "total_words": db.get_word_count(),
        "total_clusters": db.get_cluster_count(),
    }


# ── API: Generate questions ───────────────────────────────────────────────

@app.post("/api/generate")
async def api_generate(request: Request):
    body = await request.json() if await request.body() else {}
    count = body.get("count", 10)

    db = get_db()
    llm = _get_llm()

    questions = await generate_batch(llm, db, count=count)
    return {
        "generated": len(questions),
        "bank_size": db.get_question_bank_size(),
    }


# ── API: Session management ──────────────────────────────────────────────

@app.post("/api/session/start")
async def api_session_start():
    db = get_db()
    s = get_settings()

    # Start replenishing if needed (runs in background)
    await _ensure_question_buffer()

    session_id = db.start_session()

    # 1. Load ALL due review questions (up to 200) — not capped at session_size.
    #    The session target is soft; users can keep going past it.
    review_qs = db.get_review_questions(limit=200)
    questions_data = list(review_qs)
    review_count = len(questions_data)
    seen_words: set[str] = {q["target_word"].lower() for q in questions_data}

    # 2. Fill remaining slots (up to session_size) with new banked questions
    new_count = 0
    remaining_slots = max(0, s.session_size - len(questions_data))
    if remaining_slots > 0:
        new_qs = db.get_new_questions(limit=remaining_slots)
        for q in new_qs:
            if q["target_word"].lower() not in seen_words:
                questions_data.append(q)
                seen_words.add(q["target_word"].lower())
                new_count += 1

    # 3. Fill remaining slots with unseen questions for already-active words
    remaining_slots = max(0, s.session_size - len(questions_data))
    if remaining_slots > 0:
        active_qs = db.get_active_word_new_questions(limit=remaining_slots, exclude_words=seen_words)
        for q in active_qs:
            if q["target_word"].lower() not in seen_words:
                questions_data.append(q)
                seen_words.add(q["target_word"].lower())

    # Shuffle banked questions for variety
    random.shuffle(questions_data)

    # 4. Build generation queue for remaining slots (ordered, not shuffled)
    pending_queue: list[dict] = []
    remaining_slots = max(0, s.session_size - len(questions_data))
    if remaining_slots > 0:
        targets = select_session_words(db, remaining_slots + 10)
        all_clusters = db.get_all_clusters()
        for word in targets:
            if word.lower() in seen_words:
                continue
            for cl in all_clusters:
                cw = db.get_cluster_words(cl["id"])
                word_info = next((w for w in cw if w["word"].lower() == word.lower()), None)
                if word_info and len(cw) >= 4:
                    pending_queue.append({
                        "pending_cluster": cl,
                        "pending_word": word_info,
                    })
                    seen_words.add(word.lower())
                    new_count += 1
                    break
            if len(pending_queue) >= remaining_slots:
                break

    if not questions_data and not pending_queue:
        return {"error": "No questions available. Import vocabulary and generate questions first.", "session_id": None}

    # Banked questions first (shuffled), then pending questions in generation
    # order. This gives pre-generation maximum lead time: the first pending
    # question won't be needed until all banked ones are served.
    final: list[dict] = questions_data + pending_queue

    # Store session state
    _active_sessions[session_id] = {
        "questions": final,
        "current_index": 0,
        "total": 0,
        "correct": 0,
        "review_count": review_count,
        "new_count": new_count,
        "target": s.session_size,
    }

    # Background task generates pending questions sequentially —
    # indices are in session order so nearest-needed is generated first.
    pending_indices = [i for i, q in enumerate(final) if "pending_cluster" in q]
    if pending_indices:
        task = asyncio.create_task(_pregenerate_session_questions(session_id, pending_indices))
        _bg_tasks.add(task)
        task.add_done_callback(_bg_tasks.discard)

    # Return first question
    return await _get_next_question(session_id)


@app.post("/api/session/answer")
async def api_session_answer(request: Request):
    body = await request.json()
    session_id = body["session_id"]
    selected_index = body["selected_index"]
    time_seconds = body.get("time_seconds")

    if session_id not in _active_sessions:
        raise HTTPException(404, "Session not found")

    session = _active_sessions[session_id]
    current_q = session.get("current_question")
    if not current_q:
        raise HTTPException(400, "No current question")

    correct = selected_index == current_q["correct_index"]
    session["total"] += 1
    if correct:
        session["correct"] += 1

    # SRS update first (so archive check sees updated interval)
    db = get_db()
    s = get_settings()
    quality = quality_from_answer(correct, time_seconds)
    record_review(db, current_q["correct_word"], quality)

    # Record in DB + archive decision
    archive_info = {"archived": False, "reason": "", "question_id": ""}
    if current_q.get("id"):
        archive_info = db.record_question_shown(
            current_q["id"], correct, s.archive_interval_days
        )

    # Generate TTS for explanation + context sentence
    explanation_audio_hash = None
    context_audio_hash = None
    try:
        tts = _get_tts()
        s = get_settings()
        for text, attr in [
            (current_q["explanation"], "explanation_audio_hash"),
            (current_q["context_sentence"], "context_audio_hash"),
        ]:
            path = await get_or_create_audio(text, tts, db, s.audio_cache_full_path)
            if path:
                if attr == "explanation_audio_hash":
                    explanation_audio_hash = sentence_hash(text)
                else:
                    context_audio_hash = sentence_hash(text)
    except Exception:
        pass

    # Check if we need to replenish the question buffer
    await _ensure_question_buffer()

    result = {
        "correct": correct,
        "correct_index": current_q["correct_index"],
        "correct_word": current_q["correct_word"],
        "explanation": current_q["explanation"],
        "context_sentence": current_q["context_sentence"],
        "explanation_audio_hash": explanation_audio_hash,
        "context_audio_hash": context_audio_hash,
        "archive": archive_info,
        "session_progress": {
            "answered": session["total"],
            "correct": session["correct"],
            "remaining": len(session["questions"]) - session["current_index"],
            "target": session.get("target", len(session["questions"])),
        },
    }

    # Advance to next question
    session["current_index"] += 1
    if session["current_index"] >= len(session["questions"]):
        # Session complete
        db.end_session(session_id, session["total"], session["correct"])
        result["session_complete"] = True
        result["summary"] = {
            "total": session["total"],
            "correct": session["correct"],
            "accuracy": round(session["correct"] / max(session["total"], 1) * 100, 1),
            "review_count": session.get("review_count", 0),
            "new_count": session.get("new_count", 0),
        }
        del _active_sessions[session_id]
    else:
        result["session_complete"] = False

    return result


async def _get_next_question(session_id: int) -> dict:
    if session_id not in _active_sessions:
        raise HTTPException(404, "Session not found")

    session = _active_sessions[session_id]
    idx = session["current_index"]

    if idx >= len(session["questions"]):
        return {"session_complete": True, "session_id": session_id}

    q_data = session["questions"][idx]

    # Wait for background generation (queued or in-flight).
    # Never call the LLM here — the single background task owns all
    # generation so it doesn't compete with Ollama for the GPU.
    if q_data.get("generating") or "pending_cluster" in q_data:
        for _ in range(180):  # up to ~90s
            await asyncio.sleep(0.5)
            q_data = session["questions"][idx]
            if not q_data.get("generating") and "pending_cluster" not in q_data:
                break
        else:
            # Timed out or generation failed — skip this question
            session["current_index"] += 1
            return await _get_next_question(session_id)

    # Generation failed — skip
    if q_data.get("generation_failed"):
        session["current_index"] += 1
        return await _get_next_question(session_id)

    # Parse choices
    choices = q_data.get("choices_json", "[]")
    if isinstance(choices, str):
        choices = json.loads(choices)

    cluster_title = q_data.get("cluster_title", "")
    db = get_db()

    # Parse stored choice_details (always populated — backfill runs on startup)
    raw_details = q_data.get("choice_details_json", "[]")
    if isinstance(raw_details, str):
        try:
            choice_details = json.loads(raw_details)
        except (json.JSONDecodeError, TypeError):
            choice_details = []
    else:
        choice_details = raw_details or []

    # Shuffle choices so the correct answer isn't always in the same slot
    correct_index = q_data["correct_index"]
    indices = list(range(len(choices)))
    random.shuffle(indices)
    choices = [choices[i] for i in indices]
    if choice_details:
        choice_details = [choice_details[i] for i in indices]
    correct_index = indices.index(correct_index)

    target_word = q_data.get("correct_word", q_data.get("target_word", ""))
    has_review = db.get_review(target_word) is not None
    question = {
        "session_id": session_id,
        "question_type": q_data.get("question_type", "fill_blank"),
        "stem": q_data["stem"],
        "choices": choices,
        "choice_details": choice_details,
        "correct_index": correct_index,
        "correct_word": target_word,
        "explanation": q_data.get("explanation", ""),
        "context_sentence": q_data.get("context_sentence", ""),
        "cluster_title": cluster_title,
        "id": q_data.get("id", ""),
        "is_new": not has_review,
        "progress": {
            "current": idx + 1,
            "total": len(session["questions"]),
            "answered": session["total"],
            "correct": session["correct"],
            "target": session.get("target", len(session["questions"])),
        },
    }

    # Store for answer checking
    session["current_question"] = question
    return question


@app.post("/api/session/next")
async def api_session_next(request: Request):
    body = await request.json()
    session_id = body["session_id"]
    return await _get_next_question(session_id)


@app.post("/api/session/finish")
async def api_session_finish(request: Request):
    """End a session early (e.g. after reaching the soft target)."""
    body = await request.json()
    session_id = body["session_id"]

    if session_id not in _active_sessions:
        raise HTTPException(404, "Session not found")

    session = _active_sessions.pop(session_id)
    db = get_db()
    db.end_session(session_id, session["total"], session["correct"])

    return {
        "session_complete": True,
        "summary": {
            "total": session["total"],
            "correct": session["correct"],
            "accuracy": round(session["correct"] / max(session["total"], 1) * 100, 1),
            "review_count": session.get("review_count", 0),
            "new_count": session.get("new_count", 0),
        },
    }


@app.get("/api/session/summary")
async def api_session_summary():
    db = get_db()
    history = db.get_session_history(limit=10)
    return {"sessions": history}


# ── API: Question archive override ──────────────────────────────────────

@app.post("/api/question/{question_id}/archive")
async def api_question_archive(question_id: str, request: Request):
    body = await request.json()
    archived = body.get("archived", True)
    db = get_db()
    db.set_question_archived(question_id, archived)
    # Replenish buffer if un-archiving reduced nothing, but archiving might
    await _ensure_question_buffer()
    return {"question_id": question_id, "archived": archived}


# ── API: Question library ─────────────────────────────────────────────────

@app.get("/api/questions/active")
async def api_questions_active():
    return get_db().get_active_questions()


@app.get("/api/questions/archived")
async def api_questions_archived():
    return get_db().get_archived_questions()


@app.post("/api/questions/reset-due")
async def api_questions_reset_due(request: Request):
    body = await request.json()
    word = body.get("word", "")
    if not word:
        raise HTTPException(400, "No word provided")
    get_db().reset_word_due(word)
    return {"ok": True}


# ── API: Audio ────────────────────────────────────────────────────────────

@app.get("/api/audio/{audio_hash}.mp3")
async def api_audio(audio_hash: str):
    s = get_settings()
    audio_path = s.audio_cache_full_path / f"{audio_hash}.mp3"
    if not audio_path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.post("/api/tts/generate")
async def api_tts_generate(request: Request):
    """Generate TTS for arbitrary text (e.g. chat responses). Returns audio hash."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(400, "No text provided")

    # Strip markdown formatting for cleaner speech
    import re
    clean = text
    clean = re.sub(r"\*\*(.+?)\*\*", r"\1", clean)  # bold
    clean = re.sub(r"\*(.+?)\*", r"\1", clean)       # italic
    clean = re.sub(r"`(.+?)`", r"\1", clean)          # inline code
    clean = re.sub(r"^#{1,4}\s+", "", clean, flags=re.MULTILINE)  # headings
    clean = re.sub(r"\n{2,}", "\n", clean)             # collapse blank lines

    try:
        tts = _get_tts()
        s = get_settings()
        db = get_db()
        audio_path = await get_or_create_audio(clean, tts, db, s.audio_cache_full_path)
        if audio_path:
            audio_hash = sentence_hash(clean)
            return {"audio_hash": audio_hash}
        raise HTTPException(500, "TTS generation failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}")


# ── API: Settings ─────────────────────────────────────────────────────────

@app.get("/api/settings")
async def api_get_settings():
    return get_settings().to_dict()


@app.put("/api/settings")
async def api_update_settings(request: Request):
    body = await request.json()
    s = get_settings()
    known = {f.name for f in Settings.__dataclass_fields__.values()}
    for k, v in body.items():
        if k in known:
            setattr(s, k, v)
    save_settings(s)
    return s.to_dict()


# ── API: Chat (LLM tutor) ───────────────────────────────────────────────

def _build_chat_prompt(context: dict, history: list[dict], message: str) -> str:
    """Build a single prompt for the chat LLM call.

    First message (empty history): a rich preamble with full quiz context,
    word definitions, and output expectations — primes the model so
    thoroughly it doesn't need chain-of-thought reasoning.

    Follow-up messages: just the conversation history as Student/Tutor
    turns plus the new message. No preamble repetition.
    """
    if history:
        # Follow-up: compact history + new message
        lines: list[str] = []
        for msg in history:
            role = "Student" if msg["role"] == "user" else "Tutor"
            lines.append(f"{role}: {msg['content']}")
            lines.append("")
        lines.append(f"Student: {message}")
        lines.append("")
        lines.append("Tutor:")
        return "\n".join(lines)

    # First message: rich preamble with full context
    q_type = context.get("question_type", "fill_blank")
    type_labels = {
        "fill_blank": "fill-in-the-blank",
        "best_fit": "best-fit",
        "distinction": "distinction",
    }
    stem = context.get("stem", "")
    choices = context.get("choices", [])
    correct_word = context.get("correct_word", "")
    explanation = context.get("explanation", "")
    context_sentence = context.get("context_sentence", "")
    cluster_title = context.get("cluster_title", "")
    selected_idx = context.get("selected_index")
    was_correct = context.get("was_correct")
    details = context.get("choice_details", [])

    labels = ["A", "B", "C", "D"]
    choices_str = ", ".join(f"{labels[i]}) {c}" for i, c in enumerate(choices))
    chosen = choices[selected_idx] if selected_idx is not None and selected_idx < len(choices) else "?"

    # Build word definitions block
    defs_lines = []
    for d in details:
        if d.get("meaning"):
            line = f"  - {d['word']}: {d['meaning']}"
            if d.get("distinction"):
                line += f" — {d['distinction']}"
            defs_lines.append(line)
    defs_block = "\n".join(defs_lines) if defs_lines else ""

    if was_correct:
        outcome = f'The student chose "{chosen}" — correct.'
        task = f"Deepen their understanding of why **{correct_word}** fits and how it differs from the alternatives."
    else:
        outcome = f'The student chose "{chosen}" — wrong. The correct answer is "{correct_word}".'
        task = f"Clarify why **{chosen}** doesn't fit and what makes **{correct_word}** the right choice. Then illuminate the broader distinctions."

    preamble = f"""You are a vocabulary tutor. Your mission is to teach impeccable vocabulary use — the student should walk away knowing not just what each word means, but exactly when to reach for it instead of its near-synonyms, what collocations it naturally forms, and where substituting a close alternative would sound wrong to a native ear.

Your first sentence always delivers substance — a specific semantic distinction, an etymological root, a revealing collocational pattern. You write in flowing prose with **bold** highlights and example sentences woven in naturally. No lists, no tables.

Here is an example of the quality of insight you produce:
"**Beatific** and **blissful** both describe profound happiness, but they diverge in register and reach. **Beatific** carries a specifically religious resonance — from Latin *beatus* (blessed) — and implies serene, transcendent joy: you'd write 'a beatific smile' but never 'beatific ignorance.' **Blissful** works in secular contexts and pairs naturally with unawareness — 'blissful ignorance,' 'blissful sleep' — a happiness that's felt rather than radiated outward. **Elated** moves into a different register entirely: it's active, momentary, tied to a specific occasion, where the other two describe sustained states."

Context for this question:
{type_labels.get(q_type, q_type)} quiz{" — cluster: " + cluster_title if cluster_title else ""}
Stem: {stem}
Choices: {choices_str}
{outcome}
Explanation: {explanation}
Context sentence: {context_sentence}
"""

    if defs_block:
        preamble += f"""Word definitions:
{defs_block}
"""

    preamble += f"""{task}

Student: {message}

Tutor: **"""

    return preamble


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    message = body.get("message", "")
    context = body.get("context", {})
    history = body.get("history", [])
    thinking = body.get("thinking", False)

    if not message:
        raise HTTPException(400, "No message provided")

    prompt = _build_chat_prompt(context, history, message)
    llm = _get_llm()

    # Cancel in-flight background LLM tasks to free the GPU immediately
    cancelled_tasks = set(_bg_tasks)
    for t in cancelled_tasks:
        t.cancel()
    for t in cancelled_tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    async def stream():
        try:
            first_token = True
            if thinking:
                yield f"data: {json.dumps({'thinking': True})}\n\n"
            async for token in llm.generate_stream(prompt, temperature=0.7, thinking=thinking):
                if first_token and thinking:
                    yield f"data: {json.dumps({'thinking': False})}\n\n"
                    first_token = False
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Resume pre-generation for any active sessions
            for sid, session in _active_sessions.items():
                resume = session.pop("_resume_indices", None)
                if resume:
                    task = asyncio.create_task(
                        _pregenerate_session_questions(sid, resume))
                    _bg_tasks.add(task)
                    task.add_done_callback(_bg_tasks.discard)
            # Resume background buffer generation
            await _ensure_question_buffer()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
