"""Tests for the FastAPI application routes."""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import patch

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
            "choice_explanations": [
                "Correct: terse implies brevity bordering on rudeness, fitting the cold tone.",
                "Concise means brief but clear — no negative connotation here.",
                "Pithy means concise and meaningful — doesn't imply coldness.",
                "Laconic means using few words, but without the rude edge of terse.",
            ],
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
        assert "Vocab Trainer" in resp.text

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

        # Patch _get_tts to avoid real TTS calls
        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS in test")):
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

    def test_answer_wrong(self, test_app_with_data):
        client, db, settings = test_app_with_data
        session_id = self._setup_session(db)

        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS in test")):
            resp = client.post("/api/session/answer", json={
                "session_id": session_id,
                "selected_index": 2,  # pithy, not terse
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["correct"] is False
        assert data["summary"]["accuracy"] == 0.0


class TestQuestionBuffer:
    """Tests for the background question buffer and SRS-based archival."""

    def _make_question(self, word="terse", qid=None):
        return Question(
            id=qid or str(uuid.uuid4()),
            question_type="fill_blank",
            stem="The ___ reply was cold.",
            choices=["terse", "concise", "pithy", "laconic"],
            correct_index=0,
            correct_word=word,
            explanation="Terse implies rudeness.",
            context_sentence=f"The {word} reply was cold.",
            cluster_title="Being Brief",
            llm_provider="test",
        )

    def test_no_archive_on_first_correct(self, test_app_with_data):
        """First correct answer does NOT archive (SRS interval too low)."""
        client, db, _ = test_app_with_data
        q = db.get_session_questions(limit=1)[0]
        assert q["archived"] == 0

        info = db.record_question_shown(q["id"], correct=True)
        assert info["archived"] is False

    def test_no_archive_on_first_wrong(self, test_app_with_data):
        """Wrong answer does not archive."""
        client, db, _ = test_app_with_data
        q = db.get_session_questions(limit=1)[0]

        info = db.record_question_shown(q["id"], correct=False)
        assert info["archived"] is False

    def test_archived_excluded_from_session(self, test_app_with_data):
        """Archived questions are not returned by get_session_questions."""
        client, db, _ = test_app_with_data
        q = db.get_session_questions(limit=1)[0]

        db.set_question_archived(q["id"], True)
        remaining = db.get_session_questions(limit=10)
        assert all(r["id"] != q["id"] for r in remaining)

    def test_unarchive_keeps_srs_data(self, test_app_with_data):
        """Un-archiving preserves SRS data; word doesn't re-archive without high interval."""
        client, db, _ = test_app_with_data
        q = db.get_session_questions(limit=1)[0]

        # Set a high interval and archive
        db.upsert_review("terse", 2.6, 25.0, 5, "2020-01-01T00:00:00+00:00", True)
        db.record_question_shown(q["id"], correct=True)  # archives (interval >= 21)
        db.set_question_archived(q["id"], False)  # un-archive

        # Reset review interval low — should NOT re-archive
        db.upsert_review("terse", 2.5, 6.0, 2, "2020-01-01T00:00:00+00:00", True)
        info = db.record_question_shown(q["id"], correct=True)
        assert info["archived"] is False

    def test_archive_override_via_api(self, test_app_with_data):
        """POST /api/question/{id}/archive toggles archive status."""
        client, db, settings = test_app_with_data
        settings.min_ready_questions = 0  # disable buffer so it doesn't regenerate
        q = db.get_session_questions(limit=1)[0]

        # Archive via API
        resp = client.post(f"/api/question/{q['id']}/archive", json={"archived": True})
        assert resp.status_code == 200
        assert resp.json()["archived"] is True
        # get_ready_question_count now counts times_shown=0 only; question was never shown
        # so archiving it removes it from the new pool
        assert db.get_ready_question_count() == 0

        # Un-archive via API
        resp = client.post(f"/api/question/{q['id']}/archive", json={"archived": False})
        assert resp.status_code == 200
        assert resp.json()["archived"] is False
        assert db.get_ready_question_count() == 1

    def test_answer_response_includes_archive_info(self, test_app_with_data):
        """The answer API response includes archive info with SRS-based archival."""
        client, db, settings = test_app_with_data
        settings.session_size = 1
        settings.archive_interval_days = 21

        # Set up a review with interval >= 21 so it will archive on correct
        db.upsert_review("terse", 2.6, 25.0, 5, "2020-01-01T00:00:00+00:00", True)

        session_id = db.start_session()
        q = db.get_session_questions(limit=1)[0]
        app_module._active_sessions[session_id] = {
            "questions": [q],
            "current_index": 0,
            "total": 0,
            "correct": 0,
            "current_question": {
                "id": q["id"],
                "correct_index": q["correct_index"],
                "correct_word": q["target_word"],
                "explanation": q["explanation"],
                "context_sentence": q["context_sentence"],
            },
        }

        with patch("vocab_trainer.app._get_tts", side_effect=RuntimeError("no TTS")):
            resp = client.post("/api/session/answer", json={
                "session_id": session_id,
                "selected_index": 0,  # correct
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "archive" in data
        assert data["archive"]["archived"] is True
        assert data["archive"]["question_id"] == q["id"]

    def test_buffer_triggers_generation(self, test_app_with_data):
        """When ready count drops below min, _ensure_question_buffer schedules generation."""
        client, db, settings = test_app_with_data
        settings.min_ready_questions = 3

        # We have 1 question (never shown), need 3 → should trigger
        generated = []

        async def mock_generate_batch(llm, db, count=10, **kw):
            for i in range(count):
                q = self._make_question()
                db.save_question(q)
                generated.append(q)
            return generated

        with patch("vocab_trainer.app.generate_batch", side_effect=mock_generate_batch):
            # Run the buffer check
            asyncio.get_event_loop().run_until_complete(
                app_module._ensure_question_buffer()
            )
            # Let the background task complete
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))

        assert len(generated) == 2  # had 1, need 3 → generate 2
        assert db.get_ready_question_count() == 3

    def test_buffer_no_generation_when_sufficient(self, test_app_with_data):
        """No generation triggered when ready count >= min."""
        client, db, settings = test_app_with_data
        settings.min_ready_questions = 1

        # We have 1 question (never shown), need 1 → should NOT trigger
        with patch("vocab_trainer.app.generate_batch") as mock_gen:
            asyncio.get_event_loop().run_until_complete(
                app_module._ensure_question_buffer()
            )
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))
            mock_gen.assert_not_called()

    def test_stats_include_active_words(self, test_app_with_data):
        """Stats API returns active_words count."""
        client, db, _ = test_app_with_data
        resp = client.get("/api/stats")
        data = resp.json()
        assert "active_words" in data
        assert "new_questions" in data
        assert data["active_words"] == 0
        assert data["new_questions"] == 1  # 1 never-shown question

    def test_stats_ready_and_archived(self, test_app_with_data):
        """Stats API returns ready and archived question counts."""
        client, db, _ = test_app_with_data
        resp = client.get("/api/stats")
        data = resp.json()
        assert data["questions_ready"] == 1
        assert data["questions_archived"] == 0

        # Archive the question
        q = db.get_session_questions(limit=1)[0]
        db.set_question_archived(q["id"], True)
        resp = client.get("/api/stats")
        data = resp.json()
        assert data["questions_ready"] == 0
        assert data["questions_archived"] == 1
