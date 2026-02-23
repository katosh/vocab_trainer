"""SM-2 spaced repetition algorithm and word selection logic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocab_trainer.db import Database

# When a word is overdue but remembered, give half-credit for the overdue period.
# effective_interval = scheduled_interval + (overdue_days * OVERDUE_DAMPENING)
OVERDUE_DAMPENING = 0.5


def sm2_update(
    quality: int,
    easiness_factor: float = 2.5,
    interval_days: float = 1.0,
    repetitions: int = 0,
) -> tuple[float, float, int]:
    """Apply SM-2 algorithm.

    quality: 0-5 rating
      0-1: complete blackout / wrong
      2: wrong but recognized after reveal
      3: correct with significant difficulty
      4: correct with minor hesitation
      5: instant, perfect recall

    Returns (new_ef, new_interval, new_repetitions).
    """
    # Clamp quality
    quality = max(0, min(5, quality))

    # Update easiness factor
    new_ef = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)

    if quality < 3:
        # Failed — reset
        new_repetitions = 0
        new_interval = 1.0
    else:
        new_repetitions = repetitions + 1
        if new_repetitions == 1:
            new_interval = 1.0
        elif new_repetitions == 2:
            new_interval = 6.0
        else:
            new_interval = interval_days * new_ef

    return new_ef, new_interval, new_repetitions


def quality_from_answer(correct: bool, time_seconds: float | None = None) -> int:
    """Map answer correctness + response time to SM-2 quality score."""
    if not correct:
        return 1  # wrong
    if time_seconds is not None:
        if time_seconds < 3.0:
            return 5  # instant
        if time_seconds < 8.0:
            return 4  # correct, minor hesitation
        return 3  # correct, significant difficulty
    return 4  # correct, no timing info


def record_review(
    db: Database,
    word: str,
    cluster_title: str,
    quality: int,
    archive_interval_days: int = 21,
) -> dict:
    """Record a review of a (word, cluster) pair using SM-2.

    When a word is overdue and answered correctly, the scheduled interval is
    boosted by half the overdue period before feeding into SM-2.

    Returns {"archived": bool, "reason": str, "interval_days": float,
             "archive_threshold": int}.
    """
    progress = db.get_word_progress(word, cluster_title)

    if progress:
        ef = progress["easiness_factor"]
        interval = progress["interval_days"]
        reps = progress["repetitions"]

        # Compute effective interval with overdue credit for correct answers
        if quality >= 3 and progress["next_review"]:
            next_due = datetime.fromisoformat(progress["next_review"])
            if next_due.tzinfo is None:
                next_due = next_due.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            overdue_seconds = (now - next_due).total_seconds()
            if overdue_seconds > 0:
                overdue_days = overdue_seconds / 86400
                interval = interval + (overdue_days * OVERDUE_DAMPENING)
    else:
        ef = 2.5
        interval = 1.0
        reps = 0

    new_ef, new_interval, new_reps = sm2_update(quality, ef, interval, reps)
    next_review = datetime.now(timezone.utc) + timedelta(days=new_interval)
    correct = quality >= 3

    db.upsert_word_progress(
        word=word,
        cluster_title=cluster_title,
        easiness_factor=new_ef,
        interval_days=new_interval,
        repetitions=new_reps,
        next_review=next_review.isoformat(),
        correct=correct,
    )

    # Archive policy: archive when SRS interval proves sustained mastery
    should_archive = False
    reason = ""
    if correct and new_interval >= archive_interval_days:
        should_archive = True
        reason = f"Mastered (interval {new_interval:.0f} days)"
        db.set_word_archived(word, cluster_title, True)

    return {
        "archived": should_archive,
        "reason": reason,
        "interval_days": round(new_interval, 1),
        "archive_threshold": archive_interval_days,
        "easiness_factor": round(new_ef, 4),
        "repetitions": new_reps,
        "next_review": next_review.isoformat(),
    }
