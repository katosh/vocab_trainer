"""Tests for the SM-2 spaced repetition module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vocab_trainer.srs import (
    OVERDUE_DAMPENING,
    quality_from_answer,
    record_review,
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
    """Tests for record_review with cluster_title only."""

    def test_first_review_correct(self, populated_db):
        result = record_review(populated_db, "Being Brief", quality=4)
        cp = populated_db.get_cluster_progress("Being Brief")
        assert cp is not None
        assert cp["repetitions"] == 1
        assert cp["total_correct"] == 1
        assert result["archived"] is False

    def test_first_review_wrong(self, populated_db):
        result = record_review(populated_db, "Being Brief", quality=1)
        cp = populated_db.get_cluster_progress("Being Brief")
        assert cp["repetitions"] == 0
        assert cp["total_incorrect"] == 1

    def test_multiple_reviews(self, populated_db):
        record_review(populated_db, "Being Brief", quality=4)
        record_review(populated_db, "Being Brief", quality=5)
        cp = populated_db.get_cluster_progress("Being Brief")
        assert cp["repetitions"] == 2
        assert cp["total_correct"] == 2

    def test_failure_after_success_resets_reps(self, populated_db):
        record_review(populated_db, "Being Brief", quality=4)
        record_review(populated_db, "Being Brief", quality=4)
        record_review(populated_db, "Being Brief", quality=1)  # fail
        cp = populated_db.get_cluster_progress("Being Brief")
        assert cp["repetitions"] == 0
        assert cp["interval_days"] == 1.0

    def test_archive_when_interval_reached(self, populated_db):
        """Cluster archives when SRS interval >= threshold after correct answer."""
        # Set up high interval
        populated_db.upsert_cluster_progress(
            "Being Brief", 2.6, 50.0, 5, "2020-01-01T00:00:00+00:00", True
        )
        result = record_review(
            populated_db, "Being Brief", quality=5,
            archive_interval_days=45,
        )
        assert result["archived"] is True
        assert "Mastered" in result["reason"]

        cp = populated_db.get_cluster_progress("Being Brief")
        assert cp["archived"] == 1

    def test_no_archive_below_threshold(self, populated_db):
        """Cluster with short interval stays in rotation."""
        # Use a recent next_review so overdue credit doesn't inflate past threshold
        now = datetime.now(timezone.utc).isoformat()
        populated_db.upsert_cluster_progress(
            "Being Brief", 2.5, 6.0, 2, now, True
        )
        result = record_review(
            populated_db, "Being Brief", quality=4,
            archive_interval_days=45,
        )
        assert result["archived"] is False

    def test_no_archive_on_wrong(self, populated_db):
        """Wrong answer never archives, even with high interval."""
        populated_db.upsert_cluster_progress(
            "Being Brief", 2.6, 60.0, 5, "2020-01-01T00:00:00+00:00", True
        )
        result = record_review(
            populated_db, "Being Brief", quality=1,
            archive_interval_days=45,
        )
        assert result["archived"] is False

    def test_overdue_correct_gets_credit(self, populated_db):
        """A correct answer on an overdue cluster gets a longer interval."""
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 6.0, 2, ten_days_ago, True)

        record_review(populated_db, "Being Brief", quality=4)
        overdue = populated_db.get_cluster_progress("Being Brief")

        # Set up a fresh review for another cluster, due right now, same interval
        now = datetime.now(timezone.utc).isoformat()
        populated_db.upsert_cluster_progress("Other Cluster", 2.5, 6.0, 2, now, True)
        record_review(populated_db, "Other Cluster", quality=4)
        ontime = populated_db.get_cluster_progress("Other Cluster")

        # Overdue cluster should get a longer interval
        assert overdue["interval_days"] > ontime["interval_days"]

    def test_overdue_wrong_no_credit(self, populated_db):
        """A wrong answer on an overdue cluster still resets to 1 day."""
        ten_days_ago = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        populated_db.upsert_cluster_progress("Being Brief", 2.5, 6.0, 2, ten_days_ago, True)

        record_review(populated_db, "Being Brief", quality=1)
        cp = populated_db.get_cluster_progress("Being Brief")
        assert cp["interval_days"] == 1.0
        assert cp["repetitions"] == 0
