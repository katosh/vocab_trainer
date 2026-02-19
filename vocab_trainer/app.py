"""FastAPI application with all routes."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
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
        return ElevenLabsProvider(voice_id=s.elevenlabs_voice_id)
    elif s.tts_provider == "piper":
        from vocab_trainer.providers.tts_piper import PiperTTSProvider
        return PiperTTSProvider()
    raise ValueError(f"Unknown TTS provider: {s.tts_provider}")


@app.on_event("startup")
async def startup():
    global _db, _settings
    if _db is not None:
        return  # Already initialized (e.g. by tests)
    _settings = load_settings()
    _db = Database(_settings.db_full_path)


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
            clusters = parse_distinctions_file(vf)
            total_clusters += db.import_clusters(clusters)
        else:
            words = parse_vocabulary_file(vf)
            total_words += db.import_words(words)

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

    session_id = db.start_session()

    # 1. Draw from the question bank (SRS-prioritized)
    banked = db.get_session_questions(limit=s.session_size)
    questions_data = list(banked)
    seen_words: set[str] = {q["target_word"].lower() for q in questions_data}

    # 2. Fill remaining slots with on-the-fly generation targets
    remaining = s.session_size - len(questions_data)
    if remaining > 0:
        targets = select_session_words(db, remaining, s.new_words_per_session)
        for word in targets:
            if word.lower() in seen_words:
                continue
            # Find the word's cluster for on-the-fly generation
            all_clusters = db.get_all_clusters()
            for cl in all_clusters:
                cw = db.get_cluster_words(cl["id"])
                word_info = next((w for w in cw if w["word"].lower() == word.lower()), None)
                if word_info and len(cw) >= 4:
                    questions_data.append({
                        "pending_cluster": cl,
                        "pending_word": word_info,
                    })
                    seen_words.add(word.lower())
                    break
            if len(questions_data) >= s.session_size:
                break

    if not questions_data:
        return {"error": "No questions available. Import vocabulary and generate questions first.", "session_id": None}

    # Store session state
    _active_sessions[session_id] = {
        "questions": questions_data,
        "current_index": 0,
        "total": 0,
        "correct": 0,
    }

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

    # Record in DB
    db = get_db()
    if current_q.get("id"):
        db.record_question_shown(current_q["id"], correct)

    # SRS update
    quality = quality_from_answer(correct, time_seconds)
    record_review(db, current_q["correct_word"], quality)

    # Generate TTS for the context sentence
    audio_hash = None
    try:
        tts = _get_tts()
        s = get_settings()
        audio_path = await get_or_create_audio(
            current_q["context_sentence"], tts, db, s.audio_cache_full_path
        )
        if audio_path:
            audio_hash = sentence_hash(current_q["context_sentence"])
    except Exception:
        pass

    result = {
        "correct": correct,
        "correct_index": current_q["correct_index"],
        "correct_word": current_q["correct_word"],
        "explanation": current_q["explanation"],
        "context_sentence": current_q["context_sentence"],
        "audio_hash": audio_hash,
        "session_progress": {
            "answered": session["total"],
            "correct": session["correct"],
            "remaining": len(session["questions"]) - session["current_index"],
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

    # Handle pending (on-the-fly generation)
    if "pending_cluster" in q_data:
        try:
            llm = _get_llm()
            db = get_db()
            q = await generate_question(
                llm, db,
                cluster=q_data["pending_cluster"],
                target_word_info=q_data["pending_word"],
            )
            if q:
                db.save_question(q)
                q_data = {
                    "id": q.id,
                    "question_type": q.question_type,
                    "stem": q.stem,
                    "choices_json": json.dumps(q.choices),
                    "correct_index": q.correct_index,
                    "correct_word": q.correct_word,
                    "explanation": q.explanation,
                    "context_sentence": q.context_sentence,
                    "cluster_title": q.cluster_title,
                }
                session["questions"][idx] = q_data
            else:
                # Generation failed — skip this question
                session["current_index"] += 1
                if session["current_index"] >= len(session["questions"]):
                    del _active_sessions[session_id]
                    return {
                        "error": "Question generation failed. Check that your LLM provider is running and configured.",
                        "session_id": session_id,
                    }
                return await _get_next_question(session_id)
        except Exception as e:
            session["current_index"] += 1
            if session["current_index"] >= len(session["questions"]):
                del _active_sessions[session_id]
                return {
                    "error": f"LLM provider error: {e}",
                    "session_id": session_id,
                }
            return await _get_next_question(session_id)

    # Parse choices
    choices = q_data.get("choices_json", "[]")
    if isinstance(choices, str):
        choices = json.loads(choices)

    # Look up meanings/distinctions for all choices from the cluster
    choice_details = []
    cluster_title = q_data.get("cluster_title", "")
    if cluster_title:
        db = get_db()
        cluster = db.get_cluster_by_title(cluster_title)
        if cluster:
            cw = db.get_cluster_words(cluster["id"])
            cw_map = {w["word"].lower(): w for w in cw}
            for c in choices:
                info = cw_map.get(c.lower())
                if info:
                    choice_details.append({
                        "word": c,
                        "meaning": info.get("meaning", ""),
                        "distinction": info.get("distinction", ""),
                    })
                else:
                    choice_details.append({"word": c, "meaning": "", "distinction": ""})

    question = {
        "session_id": session_id,
        "question_type": q_data.get("question_type", "fill_blank"),
        "stem": q_data["stem"],
        "choices": choices,
        "choice_details": choice_details,
        "correct_index": q_data["correct_index"],
        "correct_word": q_data.get("correct_word", q_data.get("target_word", "")),
        "explanation": q_data.get("explanation", ""),
        "context_sentence": q_data.get("context_sentence", ""),
        "cluster_title": cluster_title,
        "id": q_data.get("id", ""),
        "progress": {
            "current": idx + 1,
            "total": len(session["questions"]),
            "answered": session["total"],
            "correct": session["correct"],
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


@app.get("/api/session/summary")
async def api_session_summary():
    db = get_db()
    history = db.get_session_history(limit=10)
    return {"sessions": history}


# ── API: Audio ────────────────────────────────────────────────────────────

@app.get("/api/audio/{audio_hash}.mp3")
async def api_audio(audio_hash: str):
    s = get_settings()
    audio_path = s.audio_cache_full_path / f"{audio_hash}.mp3"
    if not audio_path.exists():
        raise HTTPException(404, "Audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


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
