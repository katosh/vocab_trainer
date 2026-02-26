"""Tests for the FastAPI application routes."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from vocab_trainer import app as app_module
from vocab_trainer.app import app
from vocab_trainer.config import Settings
from vocab_trainer.db import Database
from vocab_trainer.models import Question


class FakeLLM:
    """Fake LLM that returns valid question JSON for any cluster word."""

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        return json.dumps({
            "stem": "The ___ answer was surprisingly brief.",
            "choices": ["terse", "concise", "pithy", "laconic"],
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "The terse answer was surprisingly brief.",
        })

    def name(self) -> str:
        return "fake-llm"


@pytest.fixture
def test_app(tmp_path):
    """Set up test app with temporary database and settings."""
    db = Database(tmp_path / "test.db")
    settings = Settings(
        db_path=str(tmp_path / "test.db"),
        audio_cache_dir=str(tmp_path / "audio"),
        vocab_files=[],
    )

    # Set globals BEFORE creating TestClient so startup() is a no-op
    app_module._db = db
    app_module._settings = settings
    app_module._active_sessions.clear()
    app_module._bg_generating = False

    # Patch save_settings and _get_llm so tests never hit real config/LLM
    with patch("vocab_trainer.app.save_settings"), \
         patch("vocab_trainer.app._get_llm", return_value=FakeLLM()):
        client = TestClient(app, raise_server_exceptions=False)
        yield client, db, settings
        client.close()

    db.close()
    app_module._db = None
    app_module._settings = None
    app_module._active_sessions.clear()
    app_module._bg_generating = False


@pytest.fixture
def test_app_with_data(test_app, sample_words, sample_cluster, sample_question):
    """Test app with data pre-loaded."""
    client, db, settings = test_app
    db.import_words(sample_words)
    db.import_clusters([sample_cluster])
    db.save_question(sample_question)
    return client, db, settings


class TestStaticFiles:
    def test_index(self, test_app):
        client, _, _ = test_app
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Wiseacre" in resp.text

    def test_css(self, test_app):
        client, _, _ = test_app
        resp = client.get("/style.css")
        assert resp.status_code == 200
        assert "var(--bg)" in resp.text

    def test_js(self, test_app):
        client, _, _ = test_app
        resp = client.get("/app.js")
        assert resp.status_code == 200
        assert "refreshStats" in resp.text


class TestStatsAPI:
    def test_empty_stats(self, test_app):
        client, _, _ = test_app
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_words"] == 0
        assert data["accuracy"] == 0

    def test_stats_with_data(self, test_app_with_data):
        client, _, _ = test_app_with_data
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_words"] > 0
        assert data["total_clusters"] == 1


class TestSettingsAPI:
    def test_get_settings(self, test_app):
        client, _, _ = test_app
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_provider" in data
        assert "session_size" in data

    def test_update_settings(self, test_app):
        client, _, settings = test_app
        resp = client.put("/api/settings", json={"session_size": 30})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_size"] == 30

    def test_preference_fields_in_settings(self, test_app):
        """Settings API includes auto_narrate and context_level."""
        client, _, _ = test_app
        resp = client.get("/api/settings")
        data = resp.json()
        assert data["auto_narrate"] is True
        assert data["context_level"] == "simple"

    def test_update_preference_fields(self, test_app):
        """PUT /api/settings updates preference fields."""
        client, _, _ = test_app
        resp = client.put("/api/settings", json={
            "auto_narrate": False,
            "context_level": "advanced",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["auto_narrate"] is False
        assert data["context_level"] == "advanced"


class TestSessionAPI:
    def test_start_session_no_data(self, test_app):
        client, _, _ = test_app
        resp = client.post("/api/session/start")
        assert resp.status_code == 200
        data = resp.json()
        # No words imported, so should get error or empty session
        assert data.get("error") is not None or data.get("session_complete") is True

    def test_start_session_with_pregenerated(self, test_app_with_data):
        client, db, settings = test_app_with_data
        settings.session_size = 1

        resp = client.post("/api/session/start")
        assert resp.status_code == 200
        data = resp.json()

        # Should get a question (we have a pregenerated one for "terse")
        if "stem" in data:
            assert "choices" in data
            assert "session_id" in data

    def test_session_question_includes_choice_details(self, test_app_with_data):
        """Session questions include choice_details with meaning/distinction/why."""
        client, db, settings = test_app_with_data
        settings.session_size = 1

        resp = client.post("/api/session/start")
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["choice_details"]) == 4
        for detail in data["choice_details"]:
            assert "word" in detail
            assert "base_word" in detail
            assert "meaning" in detail
            assert "distinction" in detail
            assert "why" in detail

    def test_answer_missing_session(self, test_app):
        client, _, _ = test_app
        resp = client.post("/api/session/answer", json={
            "session_id": 9999,
            "selected_index": 0,
        })
        assert resp.status_code == 404

    def test_session_summary(self, test_app_with_data):
        client, db, _ = test_app_with_data
        sid = db.start_session()
        db.end_session(sid, 5, 3)

        resp = client.get("/api/session/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 1

    def test_session_finish(self, test_app_with_data):
        """POST /api/session/finish ends session early and returns summary."""
        client, db, settings = test_app_with_data
        session_id = db.start_session()
        app_module._active_sessions[session_id] = {
            "questions": [],
            "current_index": 0,
            "total": 5,
            "correct": 3,
            "review_count": 4,
            "new_count": 1,
            "target": 20,
        }

        resp = client.post("/api/session/finish", json={"session_id": session_id})
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_complete"] is True
        assert data["summary"]["total"] == 5
        assert data["summary"]["correct"] == 3
        assert data["summary"]["accuracy"] == 60.0
        # Session should be removed from active sessions
        assert session_id not in app_module._active_sessions

    def test_session_finish_missing(self, test_app):
        """POST /api/session/finish with unknown session returns 404."""
        client, _, _ = test_app
        resp = client.post("/api/session/finish", json={"session_id": 9999})
        assert resp.status_code == 404

    def test_session_active_no_session(self, test_app):
        """GET /api/session/active returns inactive when no sessions."""
        client, _, _ = test_app
        resp = client.get("/api/session/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is False

    def test_session_active_with_unanswered_question(self, test_app_with_data):
        """GET /api/session/active returns current question when unanswered."""
        client, db, settings = test_app_with_data
        session_id = db.start_session()
        current_q = {
            "session_id": session_id,
            "stem": "Her ___ reply left no room.",
            "choices": ["terse", "concise", "pithy", "laconic"],
            "correct_index": 0,
            "correct_word": "terse",
        }
        app_module._active_sessions[session_id] = {
            "questions": [{"id": "q1", "target_word": "terse"}],
            "current_index": 0,
            "total": 0,
            "correct": 0,
            "target": 1,
            "seen_ids": {"q1"},
            "seen_pairs": {("terse", "being brief")},
            "current_question": current_q,
            "question_answered": False,
        }

        resp = client.get("/api/session/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["session_id"] == session_id
        assert data["question_answered"] is False
        assert "current_question" in data
        assert data["current_question"]["stem"] == "Her ___ reply left no room."

    def test_session_active_after_answer(self, test_app_with_data):
        """GET /api/session/active omits question after answer."""
        client, db, settings = test_app_with_data
        session_id = db.start_session()
        app_module._active_sessions[session_id] = {
            "questions": [{"id": "q1", "target_word": "terse"}],
            "current_index": 1,
            "total": 1,
            "correct": 1,
            "target": 1,
            "seen_ids": {"q1"},
            "seen_pairs": {("terse", "being brief")},
            "current_question": {"stem": "old"},
            "question_answered": True,
        }

        resp = client.get("/api/session/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["question_answered"] is True
        assert "current_question" not in data


class TestAudioAPI:
    def test_missing_audio(self, test_app):
        client, _, _ = test_app
        resp = client.get("/api/audio/nonexistent.mp3")
        assert resp.status_code == 404

    def test_serve_audio(self, test_app):
        client, _, settings = test_app
        cache_dir = Path(settings.audio_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        audio_file = cache_dir / "testhash.mp3"
        audio_file.write_bytes(b"fake mp3 data")

        resp = client.get("/api/audio/testhash.mp3")
        assert resp.status_code == 200
        assert resp.content == b"fake mp3 data"


class TestFullSessionFlow:
    """Test a complete question-answer flow with manually constructed session state."""

    def _setup_session(self, db):
        """Create a session with a known question in _active_sessions."""
        session_id = db.start_session()
        q_data = {
            "id": "test-q-001",
            "question_type": "fill_blank",
            "stem": "Her ___ reply left no room for pleasantries.",
            "choices_json": json.dumps(["terse", "concise", "pithy", "laconic"]),
            "correct_index": 0,
            "correct_word": "terse",
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply left no room for pleasantries.",
            "cluster_title": "Being Brief",
        }

        app_module._active_sessions[session_id] = {
            "questions": [q_data],
            "current_index": 0,
            "total": 0,
            "correct": 0,
            "target": 1,
            "seen_ids": {"test-q-001"},
            "seen_pairs": {("terse", "being brief")},
            "current_question": {
                **q_data,
                "choices": ["terse", "concise", "pithy", "laconic"],
                "session_id": session_id,
                "progress": {"current": 1, "total": 1, "answered": 0, "correct": 0},
            },
        }
        return session_id

    def test_answer_correct(self, test_app_with_data):
        client, db, settings = test_app_with_data
        session_id = self._setup_session(db)

        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS in test")), \
             patch("vocab_trainer.app._ensure_question_buffer", new_callable=AsyncMock):
            resp = client.post("/api/session/answer", json={
                "session_id": session_id,
                "selected_index": 0,
                "time_seconds": 2.0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct"] is True
        assert data["correct_word"] == "terse"
        assert data["session_complete"] is True
        assert data["summary"]["accuracy"] == 100.0

        # Should have created word_progress entry
        wp = db.get_word_progress("terse", "Being Brief")
        assert wp is not None
        assert wp["total_correct"] == 1

    def test_answer_wrong(self, test_app_with_data):
        client, db, settings = test_app_with_data
        session_id = self._setup_session(db)

        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS in test")), \
             patch("vocab_trainer.app._ensure_question_buffer", new_callable=AsyncMock):
            resp = client.post("/api/session/answer", json={
                "session_id": session_id,
                "selected_index": 2,  # pithy, not terse
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct"] is False
        assert data["summary"]["accuracy"] == 0.0

    def test_answer_marks_question_answered(self, test_app_with_data):
        """Answer endpoint marks the question with answered_at."""
        client, db, settings = test_app_with_data
        db.save_question(Question(
            id="test-q-001",
            question_type="fill_blank",
            stem="Her ___ reply left no room.",
            choices=["terse", "concise", "pithy", "laconic"],
            correct_index=0,
            correct_word="terse",
            explanation="Terse implies rudeness.",
            context_sentence="Her terse reply was cold.",
            cluster_title="Being Brief",
            llm_provider="test",
        ))
        session_id = self._setup_session(db)

        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS")), \
             patch("vocab_trainer.app._ensure_question_buffer", new_callable=AsyncMock):
            client.post("/api/session/answer", json={
                "session_id": session_id,
                "selected_index": 0,
            })

        q = db.get_questions_for_word("terse")
        answered = [r for r in q if r["answered_at"] is not None]
        assert len(answered) >= 1

    def test_answer_includes_archive_info(self, test_app_with_data):
        """Answer response includes archive info with word+cluster."""
        client, db, settings = test_app_with_data
        settings.archive_interval_days = 21
        # Pre-set high interval so it archives
        db.upsert_word_progress("terse", "Being Brief", 2.6, 25.0, 5, "2020-01-01T00:00:00+00:00", True)

        session_id = self._setup_session(db)

        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS")), \
             patch("vocab_trainer.app._ensure_question_buffer", new_callable=AsyncMock):
            resp = client.post("/api/session/answer", json={
                "session_id": session_id,
                "selected_index": 0,  # correct
                "time_seconds": 2.0,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "archive" in data
        assert data["archive"]["word"] == "terse"
        assert data["archive"]["cluster_title"] == "Being Brief"


class TestWordProgressAPI:
    """Tests for the word-progress archive endpoint."""

    def test_archive_word(self, test_app_with_data):
        client, db, settings = test_app_with_data
        db.upsert_word_progress("terse", "Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)

        resp = client.post("/api/word-progress/archive", json={
            "word": "terse",
            "cluster_title": "Being Brief",
            "archived": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["archived"] is True

        wp = db.get_word_progress("terse", "Being Brief")
        assert wp["archived"] == 1

    def test_restore_word(self, test_app_with_data):
        client, db, settings = test_app_with_data
        db.upsert_word_progress("terse", "Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        db.set_word_archived("terse", "Being Brief", True)

        resp = client.post("/api/word-progress/archive", json={
            "word": "terse",
            "cluster_title": "Being Brief",
            "archived": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["archived"] is False


class TestLibraryAPI:
    """Tests for the question library endpoints."""

    def test_active_empty(self, test_app_with_data):
        client, db, _ = test_app_with_data
        resp = client.get("/api/questions/active")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_active_with_progress(self, test_app_with_data):
        client, db, _ = test_app_with_data
        db.upsert_word_progress("terse", "Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)

        resp = client.get("/api/questions/active")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["target_word"] == "terse"
        assert "interval_days" in data[0]

    def test_archived_empty(self, test_app_with_data):
        client, db, _ = test_app_with_data
        resp = client.get("/api/questions/archived")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_archived_with_data(self, test_app_with_data):
        client, db, settings = test_app_with_data
        db.upsert_word_progress("terse", "Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        db.set_word_archived("terse", "Being Brief", True)

        resp = client.get("/api/questions/archived")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["target_word"] == "terse"

    def test_reset_due(self, test_app_with_data):
        client, db, _ = test_app_with_data
        db.upsert_word_progress("terse", "Being Brief", 2.6, 25.0, 5, "2099-01-01T00:00:00+00:00", True)

        resp = client.post("/api/questions/reset-due", json={
            "word": "terse",
            "cluster_title": "Being Brief",
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        wp = db.get_word_progress("terse", "Being Brief")
        assert wp["interval_days"] == 1.0

    def test_reset_due_missing_word(self, test_app):
        client, _, _ = test_app
        resp = client.post("/api/questions/reset-due", json={"word": ""})
        assert resp.status_code == 400

    def test_stats_include_active_words(self, test_app_with_data):
        """Stats API returns active_words count."""
        client, db, _ = test_app_with_data
        resp = client.get("/api/stats")
        data = resp.json()
        assert "active_words" in data
        assert data["active_words"] == 0

    def test_stats_ready_and_archived(self, test_app_with_data):
        """Stats API returns ready and archived counts."""
        client, db, _ = test_app_with_data
        resp = client.get("/api/stats")
        data = resp.json()
        assert data["questions_ready"] == 1
        assert data["questions_archived"] == 0

        # Archive the word-cluster
        db.upsert_word_progress("terse", "Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        db.set_word_archived("terse", "Being Brief", True)
        resp = client.get("/api/stats")
        data = resp.json()
        assert data["questions_archived"] == 1
