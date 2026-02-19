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

    def test_get_unused_questions(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        unused = populated_db.get_unused_questions(limit=10)
        assert len(unused) == 1
        assert unused[0]["target_word"] == "terse"

    def test_get_questions_for_word(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        qs = populated_db.get_questions_for_word("terse")
        assert len(qs) == 1

    def test_record_question_correct(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        populated_db.record_question_shown("test-q-001", correct=True)

        qs = populated_db.get_questions_for_word("terse")
        assert qs[0]["times_shown"] == 1
        assert qs[0]["times_correct"] == 1

    def test_record_question_wrong(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        populated_db.record_question_shown("test-q-001", correct=False)

        qs = populated_db.get_questions_for_word("terse")
        assert qs[0]["times_shown"] == 1
        assert qs[0]["times_correct"] == 0

    def test_used_questions_not_in_unused(self, populated_db, sample_question):
        populated_db.save_question(sample_question)
        populated_db.record_question_shown("test-q-001", correct=True)
        unused = populated_db.get_unused_questions(limit=10)
        assert len(unused) == 0


class TestArchival:
    """Tests for SRS-interval-based archival."""

    def test_archive_when_interval_reached(self, populated_db, sample_question):
        """Word archives when SRS interval >= threshold after correct answer."""
        populated_db.save_question(sample_question)
        # Simulate a word that has been reviewed many times (interval 25 days)
        populated_db.upsert_review(
            "terse", 2.6, 25.0, 5, "2020-01-01T00:00:00+00:00", True
        )
        info = populated_db.record_question_shown("test-q-001", correct=True, archive_interval_days=21)
        assert info["archived"] is True
        assert "Mastered" in info["reason"]

    def test_no_archive_below_threshold(self, populated_db, sample_question):
        """Word with short interval stays in rotation."""
        populated_db.save_question(sample_question)
        populated_db.upsert_review(
            "terse", 2.5, 6.0, 2, "2020-01-01T00:00:00+00:00", True
        )
        info = populated_db.record_question_shown("test-q-001", correct=True, archive_interval_days=21)
        assert info["archived"] is False

    def test_no_archive_on_wrong_even_high_interval(self, populated_db, sample_question):
        """Wrong answer never archives, even with high interval."""
        populated_db.save_question(sample_question)
        populated_db.upsert_review(
            "terse", 2.6, 30.0, 5, "2020-01-01T00:00:00+00:00", True
        )
        info = populated_db.record_question_shown("test-q-001", correct=False, archive_interval_days=21)
        assert info["archived"] is False

    def test_no_archive_first_correct(self, populated_db, sample_question):
        """First correct answer does NOT archive (no review data yet)."""
        populated_db.save_question(sample_question)
        info = populated_db.record_question_shown("test-q-001", correct=True)
        assert info["archived"] is False


class TestQuestionPools:
    """Tests for review/new question pool queries."""

    def test_get_review_questions_due(self, populated_db, sample_question):
        """Questions for due words with times_shown > 0 appear in review pool."""
        populated_db.save_question(sample_question)
        # Show the question once
        populated_db.record_question_shown("test-q-001", correct=True)
        # Set review as due
        populated_db.upsert_review(
            "terse", 2.5, 1.0, 1, "2020-01-01T00:00:00+00:00", True
        )
        review_qs = populated_db.get_review_questions(limit=10)
        assert len(review_qs) == 1
        assert review_qs[0]["id"] == "test-q-001"

    def test_get_review_questions_not_due(self, populated_db, sample_question):
        """Questions for words not yet due don't appear in review pool."""
        populated_db.save_question(sample_question)
        populated_db.record_question_shown("test-q-001", correct=True)
        # Set review far in the future
        populated_db.upsert_review(
            "terse", 2.5, 1.0, 1, "2099-01-01T00:00:00+00:00", True
        )
        review_qs = populated_db.get_review_questions(limit=10)
        assert len(review_qs) == 0

    def test_get_new_questions(self, populated_db, sample_question):
        """Never-shown questions appear in new pool."""
        populated_db.save_question(sample_question)
        new_qs = populated_db.get_new_questions(limit=10)
        assert len(new_qs) == 1
        assert new_qs[0]["id"] == "test-q-001"

    def test_get_new_questions_excludes_shown(self, populated_db, sample_question):
        """Shown questions don't appear in new pool."""
        populated_db.save_question(sample_question)
        populated_db.record_question_shown("test-q-001", correct=True)
        new_qs = populated_db.get_new_questions(limit=10)
        assert len(new_qs) == 0

    def test_get_active_word_count(self, populated_db, sample_question):
        """Active word count is 0 before showing, 1 after."""
        populated_db.save_question(sample_question)
        assert populated_db.get_active_word_count() == 0
        populated_db.record_question_shown("test-q-001", correct=True)
        assert populated_db.get_active_word_count() == 1

    def test_ready_count_only_new(self, populated_db, sample_question):
        """Ready count only includes never-shown questions."""
        populated_db.save_question(sample_question)
        assert populated_db.get_ready_question_count() == 1
        populated_db.record_question_shown("test-q-001", correct=True)
        assert populated_db.get_ready_question_count() == 0


class TestReviews:
    def test_new_word_no_review(self, populated_db):
        r = populated_db.get_review("perspicacious")
        assert r is None

    def test_upsert_creates_review(self, populated_db):
        populated_db.upsert_review(
            word="concise",
            easiness_factor=2.5,
            interval_days=1.0,
            repetitions=1,
            next_review="2026-02-19T00:00:00+00:00",
            correct=True,
        )
        r = populated_db.get_review("concise")
        assert r is not None
        assert r["easiness_factor"] == 2.5
        assert r["total_correct"] == 1
        assert r["total_incorrect"] == 0

    def test_upsert_updates_review(self, populated_db):
        populated_db.upsert_review("concise", 2.5, 1.0, 1, "2026-02-19T00:00:00+00:00", True)
        populated_db.upsert_review("concise", 2.6, 6.0, 2, "2026-02-25T00:00:00+00:00", True)

        r = populated_db.get_review("concise")
        assert r["easiness_factor"] == 2.6
        assert r["interval_days"] == 6.0
        assert r["total_correct"] == 2

    def test_upsert_incorrect(self, populated_db):
        populated_db.upsert_review("concise", 2.5, 1.0, 0, "2026-02-19T00:00:00+00:00", False)
        r = populated_db.get_review("concise")
        assert r["total_incorrect"] == 1
        assert r["total_correct"] == 0

    def test_get_new_words(self, populated_db):
        # All words should be "new" initially
        new = populated_db.get_new_words(limit=100)
        total = populated_db.get_word_count()
        assert len(new) == total

    def test_get_new_words_excludes_reviewed(self, populated_db):
        populated_db.upsert_review("concise", 2.5, 1.0, 1, "2026-02-19T00:00:00+00:00", True)
        new = populated_db.get_new_words(limit=100)
        new_words = [w["word"] for w in new]
        assert "concise" not in new_words

    def test_get_due_words(self, populated_db):
        # Set a review in the past so it's due
        populated_db.upsert_review("concise", 2.5, 1.0, 1, "2020-01-01T00:00:00+00:00", True)
        due = populated_db.get_due_words()
        assert len(due) == 1
        assert due[0]["word"] == "concise"

    def test_future_review_not_due(self, populated_db):
        populated_db.upsert_review("concise", 2.5, 1.0, 1, "2099-01-01T00:00:00+00:00", True)
        due = populated_db.get_due_words()
        assert len(due) == 0


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
        assert stats["words_new"] == stats["total_words"]

    def test_stats_with_session(self, populated_db):
        sid = populated_db.start_session()
        populated_db.end_session(sid, 10, 7)
        # Accuracy now comes from reviews table, not sessions
        from datetime import datetime, timezone
        next_rev = datetime.now(timezone.utc).isoformat()
        for i in range(7):
            populated_db.upsert_review(f"word{i}", 2.5, 1.0, 1, next_rev, correct=True)
        for i in range(3):
            populated_db.upsert_review(f"wrong{i}", 2.5, 1.0, 0, next_rev, correct=False)
        stats = populated_db.get_stats()
        assert stats["total_sessions"] == 1
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
