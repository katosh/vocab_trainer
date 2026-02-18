"""Tests for the FastAPI application routes."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vocab_trainer import app as app_module
from vocab_trainer.app import app
from vocab_trainer.config import Settings
from vocab_trainer.db import Database


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

    client = TestClient(app, raise_server_exceptions=False)

    yield client, db, settings

    client.close()
    db.close()
    app_module._db = None
    app_module._settings = None
    app_module._active_sessions.clear()


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
        # Patch save_settings so it doesn't write to the real config.json
        with patch("vocab_trainer.app.save_settings"):
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
        settings.new_words_per_session = 1

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
