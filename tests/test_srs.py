"""Tests for the SM-2 spaced repetition module."""
from __future__ import annotations

import pytest

from vocab_trainer.srs import (
    quality_from_answer,
    record_review,
    select_session_words,
    sm2_update,
)


class TestSM2Update:
    def test_perfect_first_review(self):
        ef, interval, reps = sm2_update(quality=5, easiness_factor=2.5, interval_days=1.0, repetitions=0)
        assert reps == 1
        assert interval == 1.0  # first rep always 1 day
        assert ef > 2.5  # perfect score raises EF

    def test_second_review(self):
        ef, interval, reps = sm2_update(quality=4, easiness_factor=2.5, interval_days=1.0, repetitions=1)
        assert reps == 2
        assert interval == 6.0  # second rep always 6 days

    def test_third_review_uses_ef(self):
        ef, interval, reps = sm2_update(quality=4, easiness_factor=2.5, interval_days=6.0, repetitions=2)
        assert reps == 3
        assert interval == 6.0 * ef  # interval * new EF

    def test_failure_resets(self):
        ef, interval, reps = sm2_update(quality=1, easiness_factor=2.5, interval_days=30.0, repetitions=5)
        assert reps == 0
        assert interval == 1.0  # reset to 1 day

    def test_quality_2_resets(self):
        ef, interval, reps = sm2_update(quality=2, easiness_factor=2.5, interval_days=10.0, repetitions=3)
        assert reps == 0
        assert interval == 1.0

    def test_quality_3_continues(self):
        ef, interval, reps = sm2_update(quality=3, easiness_factor=2.5, interval_days=1.0, repetitions=0)
        assert reps == 1

    def test_ef_never_below_1_3(self):
        ef, _, _ = sm2_update(quality=0, easiness_factor=1.3, interval_days=1.0, repetitions=0)
        assert ef >= 1.3

    def test_ef_decreases_on_low_quality(self):
        ef, _, _ = sm2_update(quality=3, easiness_factor=2.5, interval_days=1.0, repetitions=0)
        assert ef < 2.5

    def test_ef_increases_on_high_quality(self):
        ef, _, _ = sm2_update(quality=5, easiness_factor=2.5, interval_days=1.0, repetitions=0)
        assert ef > 2.5

    def test_quality_clamped_low(self):
        ef, interval, reps = sm2_update(quality=-5, easiness_factor=2.5, interval_days=1.0, repetitions=0)
        assert reps == 0  # treated as quality 0 — failure

    def test_quality_clamped_high(self):
        ef, _, _ = sm2_update(quality=10, easiness_factor=2.5, interval_days=1.0, repetitions=0)
        # Should not crash, treated as quality 5
        assert ef > 2.5

    def test_interval_grows_over_time(self):
        """Simulate a series of correct answers and verify interval growth."""
        ef = 2.5
        interval = 1.0
        reps = 0
        intervals = []

        for _ in range(5):
            ef, interval, reps = sm2_update(quality=4, easiness_factor=ef, interval_days=interval, repetitions=reps)
            intervals.append(interval)

        # Each interval should be >= the previous
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i - 1]


class TestQualityFromAnswer:
    def test_wrong(self):
        assert quality_from_answer(correct=False) == 1

    def test_wrong_ignores_time(self):
        assert quality_from_answer(correct=False, time_seconds=1.0) == 1

    def test_correct_no_time(self):
        assert quality_from_answer(correct=True) == 4

    def test_correct_instant(self):
        assert quality_from_answer(correct=True, time_seconds=1.5) == 5

    def test_correct_fast(self):
        assert quality_from_answer(correct=True, time_seconds=5.0) == 4

    def test_correct_slow(self):
        assert quality_from_answer(correct=True, time_seconds=15.0) == 3


class TestRecordReview:
    def test_first_review_correct(self, populated_db):
        record_review(populated_db, "concise", quality=4)
        r = populated_db.get_review("concise")
        assert r is not None
        assert r["repetitions"] == 1
        assert r["total_correct"] == 1

    def test_first_review_wrong(self, populated_db):
        record_review(populated_db, "concise", quality=1)
        r = populated_db.get_review("concise")
        assert r["repetitions"] == 0
        assert r["total_incorrect"] == 1

    def test_multiple_reviews(self, populated_db):
        record_review(populated_db, "concise", quality=4)
        record_review(populated_db, "concise", quality=5)
        r = populated_db.get_review("concise")
        assert r["repetitions"] == 2
        assert r["total_correct"] == 2

    def test_failure_after_success_resets_reps(self, populated_db):
        record_review(populated_db, "concise", quality=4)
        record_review(populated_db, "concise", quality=4)
        record_review(populated_db, "concise", quality=1)  # fail
        r = populated_db.get_review("concise")
        assert r["repetitions"] == 0
        assert r["interval_days"] == 1.0


class TestSelectSessionWords:
    def test_all_new_words(self, populated_db):
        words = select_session_words(populated_db, session_size=5)
        assert len(words) <= 5
        assert len(words) > 0

    def test_respects_session_size(self, populated_db):
        words = select_session_words(populated_db, session_size=3)
        assert len(words) <= 3

    def test_due_words_prioritized(self, populated_db):
        # Mark "concise" as due
        populated_db.upsert_review("concise", 2.5, 1.0, 1, "2020-01-01T00:00:00+00:00", True)
        words = select_session_words(populated_db, session_size=5)
        assert "concise" in words

    def test_fills_with_new_words(self, populated_db):
        words = select_session_words(populated_db, session_size=100)
        # Should get all available cluster words (no cap beyond session_size)
        assert len(words) > 0
