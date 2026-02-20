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


def record_review(db: Database, word: str, quality: int) -> None:
    """Record a review of a word using SM-2.

    When a word is overdue and answered correctly, the scheduled interval is
    boosted by half the overdue period before feeding into SM-2.  This gives
    credit for remembering something longer than expected.  Wrong answers
    still reset to 1 day (SM-2 handles that internally).
    """
    review = db.get_review(word)

    if review:
        ef = review["easiness_factor"]
        interval = review["interval_days"]
        reps = review["repetitions"]

        # Compute effective interval with overdue credit for correct answers
        if quality >= 3 and review["next_review"]:
            next_due = datetime.fromisoformat(review["next_review"])
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

    db.upsert_review(
        word=word,
        easiness_factor=new_ef,
        interval_days=new_interval,
        repetitions=new_reps,
        next_review=next_review.isoformat(),
        correct=quality >= 3,
    )


def select_session_words(
    db: Database,
    session_size: int = 20,
) -> list[str]:
    """Select words for a training session (on-the-fly fallback).

    Only selects words that are in clusters (can generate questions).

    Priority:
    1. Overdue cluster words (sorted by staleness — most overdue first)
    2. New cluster words (never reviewed)
    """
    words: list[str] = []

    # 1. Due cluster words
    due = db.get_due_cluster_words(limit=session_size)
    for w in due:
        words.append(w)
        if len(words) >= session_size:
            return words

    # 2. New cluster words
    remaining = session_size - len(words)
    if remaining > 0:
        new = db.get_new_cluster_words(limit=remaining)
        for w in new:
            words.append(w)
            if len(words) >= session_size:
                break

    return words
