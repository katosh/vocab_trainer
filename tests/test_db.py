"""Tests for the database layer."""
from __future__ import annotations

import json

import pytest

from vocab_trainer.db import Database
from vocab_trainer.models import (
    DistinctionCluster,
    DistinctionEntry,
    Question,
    VocabWord,
)


class TestImport:
    def test_import_words(self, tmp_db, sample_words):
        count = tmp_db.import_words(sample_words)
        assert count == 5
        assert tmp_db.get_word_count() == 5

    def test_import_words_idempotent(self, tmp_db, sample_words):
        tmp_db.import_words(sample_words)
        tmp_db.import_words(sample_words)
        assert tmp_db.get_word_count() == 5

    def test_import_clusters(self, tmp_db, sample_cluster):
        count = tmp_db.import_clusters([sample_cluster])
        assert count == 1
        assert tmp_db.get_cluster_count() == 1

    def test_cluster_words_added_to_words_table(self, tmp_db, sample_cluster):
        tmp_db.import_clusters([sample_cluster])
        # Cluster entries should be in the words table
        assert tmp_db.get_word_count() == 6  # 6 entries in sample_cluster

    def test_import_clusters_idempotent(self, tmp_db, sample_cluster):
        tmp_db.import_clusters([sample_cluster])
        tmp_db.import_clusters([sample_cluster])
        assert tmp_db.get_cluster_count() == 1


class TestWords:
    def test_get_all_words(self, populated_db):
        words = populated_db.get_all_words()
        assert len(words) > 0
        assert all("word" in w for w in words)

    def test_word_fields(self, populated_db):
        words = populated_db.get_all_words()
        w = next(w for w in words if w["word"] == "perspicacious")
        assert w["definition"] == "having keen insight"
        assert w["section"] == "Cognition"
        assert w["source_file"] == "vocabulary.md"


class TestClusters:
    def test_get_cluster_by_title(self, populated_db):
        c = populated_db.get_cluster_by_title("Being Brief")
        assert c is not None
        assert c["title"] == "Being Brief"

    def test_get_cluster_by_title_not_found(self, populated_db):
        c = populated_db.get_cluster_by_title("Nonexistent")
        assert c is None

    def test_get_random_cluster(self, populated_db):
        c = populated_db.get_random_cluster()
        assert c is not None
        assert c["title"] == "Being Brief"

    def test_get_cluster_words(self, populated_db):
        c = populated_db.get_cluster_by_title("Being Brief")
        words = populated_db.get_cluster_words(c["id"])
        assert len(words) == 6
        word_names = [w["word"] for w in words]
        assert "concise" in word_names
        assert "terse" in word_names


class TestQuestions:
    def test_save_and_retrieve(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        assert populated_db.get_question_bank_size() == 1

    def test_get_questions_for_word(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        qs = populated_db.get_questions_for_word("terse")
        assert len(qs) == 1

    def test_choice_details_round_trip(self, populated_db):
        """choice_details_json is persisted and retrievable."""
        q = Question(
            id="test-cd-001",
            question_type="fill_blank",
            stem="Her ___ reply left no room.",
            choices=["terse", "concise", "pithy", "laconic"],
            correct_index=0,
            correct_word="terse",
            explanation="Terse implies rudeness.",
            context_sentence="Her terse reply left no room.",
            cluster_title="Being Brief",
            llm_provider="test",
            choice_details=[
                {"word": "terse", "meaning": "brief and rude", "distinction": "personality trait"},
                {"word": "concise", "meaning": "brief and clear", "distinction": "skillful compression"},
                {"word": "pithy", "meaning": "concise and meaningful", "distinction": "best kind of brief"},
                {"word": "laconic", "meaning": "few words habitually", "distinction": "temperamental"},
            ],
        )
        populated_db.save_question(q)
        rows = populated_db.get_questions_for_word("terse")
        row = next(r for r in rows if r["id"] == "test-cd-001")
        details = json.loads(row["choice_details_json"])
        assert len(details) == 4
        assert details[0]["word"] == "terse"
        assert details[0]["meaning"] == "brief and rude"
        assert details[1]["distinction"] == "skillful compression"

    def test_mark_question_answered(self, populated_db, sample_question):
        """Answered questions have answered_at set."""
        populated_db.save_question(sample_question)
        populated_db.mark_question_answered("test-q-001", 0, True, 2000, 1)
        qs = populated_db.get_questions_for_word("terse")
        q = next(q for q in qs if q["id"] == "test-q-001")
        assert q["answered_at"] is not None
        assert q["was_correct"] == 1
        assert q["chosen_index"] == 0

    def test_answered_not_in_ready(self, populated_db, sample_question):
        """Answered questions don't appear in ready count."""
        populated_db.save_question(sample_question)
        assert populated_db.get_ready_question_count() == 1
        populated_db.mark_question_answered("test-q-001", 0, True)
        assert populated_db.get_ready_question_count() == 0


class TestClusterProgress:
    """Tests for the cluster_progress SRS table."""

    def test_new_cluster_no_progress(self, populated_db):
        r = populated_db.get_cluster_progress("Being Brief")
        assert r is None

    def test_upsert_creates_progress(self, populated_db):
        populated_db.upsert_cluster_progress(
            cluster_title="Being Brief",
            easiness_factor=2.5,
            interval_days=1.0,
            repetitions=1,
            next_review="2026-02-19T00:00:00+00:00",
            correct=True,
        )
        r = populated_db.get_cluster_progress("Being Brief")
        assert r is not None
        assert r["easiness_factor"] == 2.5
        assert r["total_correct"] == 1
        assert r["total_incorrect"] == 0

    def test_upsert_updates_progress(self, populated_db):
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-19T00:00:00+00:00", True)
        populated_db.upsert_cluster_progress("Being Brief", 2.6, 6.0, 2, "2026-02-25T00:00:00+00:00", True)

        r = populated_db.get_cluster_progress("Being Brief")
        assert r["easiness_factor"] == 2.6
        assert r["interval_days"] == 6.0
        assert r["total_correct"] == 2

    def test_upsert_incorrect(self, populated_db):
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 0, "2026-02-19T00:00:00+00:00", False)
        r = populated_db.get_cluster_progress("Being Brief")
        assert r["total_incorrect"] == 1
        assert r["total_correct"] == 0

    def test_set_cluster_archived(self, populated_db):
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-19T00:00:00+00:00", True)
        populated_db.set_cluster_archived("Being Brief", True)
        r = populated_db.get_cluster_progress("Being Brief")
        assert r["archived"] == 1

    def test_archived_excludes_from_active(self, populated_db):
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-19T00:00:00+00:00", True)
        assert populated_db.get_active_cluster_count() == 1
        populated_db.set_cluster_archived("Being Brief", True)
        assert populated_db.get_active_cluster_count() == 0


class TestArchival:
    """Tests for cluster-level archival via cluster_progress."""

    def test_archive_shows_in_archived_list(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.6, 25.0, 5, "2020-01-01T00:00:00+00:00", True)
        populated_db.set_cluster_archived("Being Brief", True)

        archived = populated_db.get_archived_clusters()
        assert len(archived) == 1
        assert archived[0]["cluster_title"] == "Being Brief"

    def test_restore_returns_to_active(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 6.0, 2, "2020-01-01T00:00:00+00:00", True)
        populated_db.set_cluster_archived("Being Brief", True)
        populated_db.set_cluster_archived("Being Brief", False)

        active = populated_db.get_active_clusters()
        assert len(active) == 1
        assert active[0]["cluster_title"] == "Being Brief"

    def test_archived_excluded_from_session(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2020-01-01T00:00:00+00:00", True)
        populated_db.set_cluster_archived("Being Brief", True)

        session_qs = populated_db.get_session_questions(limit=10)
        assert all(q["cluster_title"] != "Being Brief" for q in session_qs)


class TestLibraryQueries:
    """Tests for get_active_clusters, get_archived_clusters, reset_cluster_due."""

    def test_active_empty_when_no_progress(self, populated_db, sample_question):
        """No cluster_progress entries → empty active list."""
        populated_db.save_question(sample_question)
        assert len(populated_db.get_active_clusters()) == 0

    def test_active_appears_after_progress(self, populated_db, sample_question):
        """Clusters with progress appear in active list."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)

        active = populated_db.get_active_clusters()
        assert len(active) == 1
        assert active[0]["cluster_title"] == "Being Brief"
        assert active[0]["interval_days"] == 1.0
        assert active[0]["next_review"] is not None

    def test_archived_clusters_empty_initially(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        assert len(populated_db.get_archived_clusters()) == 0

    def test_reset_cluster_due(self, populated_db, sample_question):
        """reset_cluster_due sets next_review to now and interval to 1."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.6, 25.0, 5, "2099-01-01T00:00:00+00:00", True)
        populated_db.reset_cluster_due("Being Brief")
        r = populated_db.get_cluster_progress("Being Brief")
        assert r["interval_days"] == 1.0
        assert r["repetitions"] == 0


class TestQuestionPools:
    """Tests for review/new question pool queries."""

    def test_get_review_questions_due(self, populated_db, sample_question):
        """Ready questions for due clusters appear in review pool."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress(
            "Being Brief", 2.5, 1.0, 1, "2020-01-01T00:00:00+00:00", True
        )
        review_qs = populated_db.get_review_questions(limit=10)
        assert len(review_qs) == 1
        assert review_qs[0]["id"] == "test-q-001"

    def test_get_review_questions_not_due(self, populated_db, sample_question):
        """Questions for not-yet-due clusters don't appear in review pool."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress(
            "Being Brief", 2.5, 1.0, 1, "2099-01-01T00:00:00+00:00", True
        )
        review_qs = populated_db.get_review_questions(limit=10)
        assert len(review_qs) == 0

    def test_get_new_questions(self, populated_db, sample_question):
        """Ready questions with no cluster_progress appear in new pool."""
        populated_db.save_question(sample_question)
        new_qs = populated_db.get_new_questions(limit=10)
        assert len(new_qs) == 1
        assert new_qs[0]["id"] == "test-q-001"

    def test_get_new_questions_excludes_progressed(self, populated_db, sample_question):
        """Questions for clusters with progress don't appear in new pool."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        new_qs = populated_db.get_new_questions(limit=10)
        assert len(new_qs) == 0

    def test_answered_excluded_from_session(self, populated_db, sample_question):
        """Answered questions don't appear in session pool."""
        populated_db.save_question(sample_question)
        populated_db.mark_question_answered("test-q-001", 0, True)
        session_qs = populated_db.get_session_questions(limit=10)
        assert len(session_qs) == 0

    def test_active_cluster_count(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        assert populated_db.get_active_cluster_count() == 0
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        assert populated_db.get_active_cluster_count() == 1

    def test_ready_count(self, populated_db, sample_question):
        """Ready count reflects unanswered questions."""
        populated_db.save_question(sample_question)
        assert populated_db.get_ready_question_count() == 1
        populated_db.mark_question_answered("test-q-001", 0, True)
        assert populated_db.get_ready_question_count() == 0


class TestClusterQuestionCounts:
    def test_returns_all_clusters(self, populated_db):
        """Returns a row for every cluster with >= 4 words."""
        clusters = populated_db.get_cluster_question_counts()
        titles = {c["cluster_title"] for c in clusters}
        assert "Being Brief" in titles
        assert len(clusters) == 1  # one cluster

    def test_counts_ready_questions(self, populated_db, sample_question):
        """Question count reflects only unanswered questions."""
        populated_db.save_question(sample_question)
        clusters = populated_db.get_cluster_question_counts()
        bb = next(c for c in clusters if c["cluster_title"] == "Being Brief")
        assert bb["question_count"] == 1

    def test_answered_excluded_from_count(self, populated_db, sample_question):
        """Answered questions don't count as ready."""
        populated_db.save_question(sample_question)
        populated_db.mark_question_answered("test-q-001", 0, True)
        clusters = populated_db.get_cluster_question_counts()
        bb = next(c for c in clusters if c["cluster_title"] == "Being Brief")
        assert bb["question_count"] == 0

    def test_archived_excluded(self, populated_db, sample_question):
        """Archived clusters are excluded."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        populated_db.set_cluster_archived("Being Brief", True)
        clusters = populated_db.get_cluster_question_counts()
        bb_rows = [c for c in clusters if c["cluster_title"] == "Being Brief"]
        assert len(bb_rows) == 0

    def test_empty_db(self, tmp_db):
        assert tmp_db.get_cluster_question_counts() == []


class TestClustersNeedingQuestions:
    def test_active_with_no_ready_question(self, populated_db, sample_question):
        """Active clusters with no unanswered question need questions."""
        populated_db.save_question(sample_question)
        # Create progress for Being Brief, answer the question
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        populated_db.mark_question_answered("test-q-001", 0, True)

        needing = populated_db.get_clusters_needing_questions()
        titles = {n["cluster_title"] for n in needing}
        assert "Being Brief" in titles

    def test_active_with_ready_question(self, populated_db, sample_question):
        """Active clusters with a ready question don't need generation."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)

        needing = populated_db.get_clusters_needing_questions()
        titles = {n["cluster_title"] for n in needing}
        assert "Being Brief" not in titles

    def test_archived_excluded(self, populated_db, sample_question):
        """Archived clusters don't need questions."""
        populated_db.save_question(sample_question)
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 1.0, 1, "2026-02-20T00:00:00+00:00", True)
        populated_db.mark_question_answered("test-q-001", 0, True)
        populated_db.set_cluster_archived("Being Brief", True)

        needing = populated_db.get_clusters_needing_questions()
        titles = {n["cluster_title"] for n in needing}
        assert "Being Brief" not in titles


class TestClusterWordAccuracy:
    def test_no_history(self, populated_db):
        """No answered questions → empty accuracy."""
        acc = populated_db.get_cluster_word_accuracy("Being Brief")
        assert acc == []

    def test_with_answered_questions(self, populated_db, sample_question):
        """Accuracy reflects answered questions per word."""
        populated_db.save_question(sample_question)
        populated_db.mark_question_answered("test-q-001", 0, True)
        acc = populated_db.get_cluster_word_accuracy("Being Brief")
        assert len(acc) == 1
        assert acc[0]["word"] == "terse"
        assert acc[0]["total"] == 1
        assert acc[0]["correct"] == 1


class TestMigration:
    def test_word_progress_migration(self, tmp_path):
        """Old word_progress data is migrated to cluster_progress."""
        db_path = tmp_path / "migrate.db"
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        # Create old schema with word_progress
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS words (word TEXT PRIMARY KEY, definition TEXT, section TEXT, source_file TEXT);
            CREATE TABLE IF NOT EXISTS clusters (id INTEGER PRIMARY KEY, title TEXT UNIQUE, preamble TEXT, commentary TEXT, source_file TEXT);
            CREATE TABLE IF NOT EXISTS cluster_words (cluster_id INTEGER, word TEXT, meaning TEXT, distinction TEXT, PRIMARY KEY (cluster_id, word));
            CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, question_type TEXT, target_word TEXT, stem TEXT, choices_json TEXT, correct_index INTEGER, explanation TEXT, context_sentence TEXT, cluster_title TEXT, llm_provider TEXT, generated_at TEXT, choice_details_json TEXT DEFAULT '[]', answered_at TEXT, chosen_index INTEGER, was_correct INTEGER, response_time_ms INTEGER, session_id INTEGER);
            CREATE TABLE IF NOT EXISTS word_progress (word TEXT NOT NULL, cluster_title TEXT NOT NULL, archived INTEGER DEFAULT 0, easiness_factor REAL DEFAULT 2.5, interval_days REAL DEFAULT 1.0, repetitions INTEGER DEFAULT 0, next_review TEXT, last_review TEXT, total_correct INTEGER DEFAULT 0, total_incorrect INTEGER DEFAULT 0, PRIMARY KEY (word, cluster_title));
            CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, started_at TEXT, ended_at TEXT, questions_total INTEGER DEFAULT 0, questions_correct INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS audio_cache (sentence_hash TEXT PRIMARY KEY, file_path TEXT, tts_provider TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS file_mtimes (file_path TEXT PRIMARY KEY, mtime_ns INTEGER);
        """)
        # Insert old data
        conn.execute("INSERT INTO word_progress VALUES ('concise', 'Being Brief', 0, 2.3, 6.0, 3, '2026-03-01T00:00:00+00:00', '2026-02-28T00:00:00+00:00', 5, 2)")
        conn.execute("INSERT INTO word_progress VALUES ('terse', 'Being Brief', 0, 2.5, 10.0, 4, '2026-03-05T00:00:00+00:00', '2026-02-28T00:00:00+00:00', 3, 1)")
        conn.commit()
        conn.close()

        # Open with Database class (triggers migration)
        db = Database(db_path)
        cp = db.get_cluster_progress("Being Brief")
        assert cp is not None
        # Pessimistic: MIN EF, MIN interval, MIN reps
        assert cp["easiness_factor"] == 2.3
        assert cp["interval_days"] == 6.0
        assert cp["repetitions"] == 3
        # Earliest next_review
        assert cp["next_review"] == "2026-03-01T00:00:00+00:00"
        # Summed totals
        assert cp["total_correct"] == 8
        assert cp["total_incorrect"] == 3
        db.close()


class TestSessions:
    def test_start_session(self, populated_db):
        sid = populated_db.start_session()
        assert sid >= 1

    def test_end_session(self, populated_db):
        sid = populated_db.start_session()
        populated_db.end_session(sid, total=10, correct=7)

        history = populated_db.get_session_history()
        assert len(history) == 1
        assert history[0]["questions_total"] == 10
        assert history[0]["questions_correct"] == 7
        assert history[0]["ended_at"] is not None

    def test_session_history_order(self, populated_db):
        s1 = populated_db.start_session()
        populated_db.end_session(s1, 5, 3)
        s2 = populated_db.start_session()
        populated_db.end_session(s2, 10, 8)

        history = populated_db.get_session_history()
        assert history[0]["id"] == s2  # most recent first


class TestStats:
    def test_empty_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert stats["total_words"] == 0
        assert stats["accuracy"] == 0

    def test_stats_with_data(self, populated_db):
        stats = populated_db.get_stats()
        assert stats["total_words"] > 0
        assert stats["total_clusters"] == 1
        assert stats["clusters_new"] == stats["total_clusters"]

    def test_stats_with_cluster_progress(self, populated_db):
        """Stats accuracy comes from cluster_progress."""
        from datetime import datetime, timezone
        next_rev = datetime.now(timezone.utc).isoformat()
        for i in range(7):
            populated_db.upsert_cluster_progress(f"cluster{i}", 2.5, 1.0, 1, next_rev, correct=True)
        for i in range(3):
            populated_db.upsert_cluster_progress(f"wrong{i}", 2.5, 1.0, 0, next_rev, correct=False)
        stats = populated_db.get_stats()
        assert stats["total_questions_answered"] == 10
        assert stats["accuracy"] == 70.0


class TestAudioCache:
    def test_get_missing(self, tmp_db):
        assert tmp_db.get_audio_cache("nonexistent") is None

    def test_set_and_get(self, tmp_db):
        tmp_db.set_audio_cache("abc123", "/path/to/audio.mp3", "edge-tts")
        assert tmp_db.get_audio_cache("abc123") == "/path/to/audio.mp3"

    def test_overwrite(self, tmp_db):
        tmp_db.set_audio_cache("abc123", "/path/old.mp3", "edge-tts")
        tmp_db.set_audio_cache("abc123", "/path/new.mp3", "piper")
        assert tmp_db.get_audio_cache("abc123") == "/path/new.mp3"
