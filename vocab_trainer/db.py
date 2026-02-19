from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from vocab_trainer.models import (
    DistinctionCluster,
    Question,
    VocabWord,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
    word TEXT PRIMARY KEY,
    definition TEXT NOT NULL,
    section TEXT,
    source_file TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL UNIQUE,
    preamble TEXT,
    commentary TEXT
);

CREATE TABLE IF NOT EXISTS cluster_words (
    cluster_id INTEGER REFERENCES clusters(id),
    word TEXT,
    meaning TEXT,
    distinction TEXT,
    PRIMARY KEY (cluster_id, word)
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    question_type TEXT NOT NULL,
    target_word TEXT NOT NULL,
    stem TEXT NOT NULL,
    choices_json TEXT NOT NULL,
    correct_index INTEGER NOT NULL,
    explanation TEXT,
    context_sentence TEXT,
    cluster_title TEXT,
    llm_provider TEXT,
    generated_at TEXT NOT NULL,
    times_shown INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reviews (
    word TEXT PRIMARY KEY,
    easiness_factor REAL DEFAULT 2.5,
    interval_days REAL DEFAULT 1.0,
    repetitions INTEGER DEFAULT 0,
    next_review TEXT,
    last_review TEXT,
    total_correct INTEGER DEFAULT 0,
    total_incorrect INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    questions_total INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS audio_cache (
    sentence_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    tts_provider TEXT,
    created_at TEXT NOT NULL
);
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after initial schema."""
        cursor = self.conn.execute("PRAGMA table_info(questions)")
        cols = {row[1] for row in cursor.fetchall()}
        if "archived" not in cols:
            self.conn.execute(
                "ALTER TABLE questions ADD COLUMN archived INTEGER DEFAULT 0"
            )
        if "consecutive_correct" not in cols:
            self.conn.execute(
                "ALTER TABLE questions ADD COLUMN consecutive_correct INTEGER DEFAULT 0"
            )
        if "choice_explanations_json" not in cols:
            self.conn.execute(
                "ALTER TABLE questions ADD COLUMN choice_explanations_json TEXT DEFAULT '[]'"
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ── Import ────────────────────────────────────────────────────────────

    def import_words(self, words: list[VocabWord]) -> int:
        count = 0
        for w in words:
            self.conn.execute(
                "INSERT OR REPLACE INTO words (word, definition, section, source_file) "
                "VALUES (?, ?, ?, ?)",
                (w.word, w.definition, w.section, w.source_file),
            )
            count += 1
        self.conn.commit()
        return count

    def import_clusters(self, clusters: list[DistinctionCluster]) -> int:
        count = 0
        for c in clusters:
            cur = self.conn.execute(
                "INSERT OR REPLACE INTO clusters (title, preamble, commentary) "
                "VALUES (?, ?, ?)",
                (c.title, c.preamble, c.commentary),
            )
            cluster_id = cur.lastrowid
            # Clear old entries for this cluster
            self.conn.execute(
                "DELETE FROM cluster_words WHERE cluster_id = ?", (cluster_id,)
            )
            for e in c.entries:
                self.conn.execute(
                    "INSERT OR REPLACE INTO cluster_words (cluster_id, word, meaning, distinction) "
                    "VALUES (?, ?, ?, ?)",
                    (cluster_id, e.word, e.meaning, e.distinction),
                )
                # Also ensure the word exists in the words table
                self.conn.execute(
                    "INSERT OR IGNORE INTO words (word, definition, section, source_file) "
                    "VALUES (?, ?, ?, ?)",
                    (e.word, e.meaning, c.title, "vocabulary_distinctions.md"),
                )
            count += 1
        self.conn.commit()
        return count

    # ── Words ─────────────────────────────────────────────────────────────

    def get_word_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM words").fetchone()
        return row[0]

    def get_all_words(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM words").fetchall()
        return [dict(r) for r in rows]

    # ── Clusters ──────────────────────────────────────────────────────────

    def get_cluster_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM clusters").fetchone()
        return row[0]

    def get_all_clusters(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM clusters").fetchall()
        return [dict(r) for r in rows]

    def get_cluster_words(self, cluster_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM cluster_words WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_by_title(self, title: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM clusters WHERE title = ?", (title,)
        ).fetchone()
        return dict(row) if row else None

    def get_random_cluster(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM clusters ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ── Questions ─────────────────────────────────────────────────────────

    def save_question(self, q: Question) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO questions "
            "(id, question_type, target_word, stem, choices_json, correct_index, "
            "explanation, context_sentence, cluster_title, llm_provider, generated_at, "
            "choice_explanations_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                q.id,
                q.question_type,
                q.correct_word,
                q.stem,
                json.dumps(q.choices),
                q.correct_index,
                q.explanation,
                q.context_sentence,
                q.cluster_title,
                q.llm_provider,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(q.choice_explanations),
            ),
        )
        self.conn.commit()

    def get_question_bank_size(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM questions").fetchone()
        return row[0]

    def get_ready_question_count(self) -> int:
        """Non-archived questions available for sessions."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE archived = 0"
        ).fetchone()
        return row[0]

    def get_archived_question_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE archived = 1"
        ).fetchone()
        return row[0]

    def get_unused_questions(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM questions WHERE times_shown = 0 ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_questions_for_word(self, word: str, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM questions WHERE target_word = ? ORDER BY RANDOM() LIMIT ?",
            (word, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_questions(self, limit: int = 20) -> list[dict]:
        """Pull non-archived banked questions ordered by SRS priority.

        Priority:
        1. Due words the user previously got wrong (struggling)
        2. Other due words (spaced repetition schedule)
        3. Never-reviewed words (new material)
        Least-shown questions preferred within each tier.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""
            SELECT q.*,
                CASE
                    WHEN r.word IS NOT NULL AND r.next_review <= ?
                         AND r.total_incorrect > r.total_correct
                        THEN 0  -- struggling + due
                    WHEN r.word IS NOT NULL AND r.next_review <= ?
                        THEN 1  -- due
                    WHEN r.word IS NULL
                        THEN 2  -- new
                    ELSE 3      -- not yet due
                END AS priority
            FROM questions q
            LEFT JOIN reviews r ON q.target_word = r.word
            WHERE priority < 3 AND q.archived = 0
            ORDER BY priority ASC, q.times_shown ASC, RANDOM()
            LIMIT ?
        """, (now, now, limit)).fetchall()
        return [dict(r) for r in rows]

    def record_question_shown(self, question_id: str, correct: bool) -> dict:
        """Record answer, update streak, auto-archive if mastered.

        Returns {"archived": bool, "reason": str, "question_id": str}.
        """
        q = self.conn.execute(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
        if not q:
            return {"archived": False, "reason": "", "question_id": question_id}

        q = dict(q)
        if correct:
            new_consec = q.get("consecutive_correct", 0) + 1
            self.conn.execute(
                "UPDATE questions SET times_shown = times_shown + 1, "
                "times_correct = times_correct + 1, consecutive_correct = ? "
                "WHERE id = ?",
                (new_consec, question_id),
            )
        else:
            new_consec = 0
            self.conn.execute(
                "UPDATE questions SET times_shown = times_shown + 1, "
                "consecutive_correct = 0 WHERE id = ?",
                (question_id,),
            )

        # Archive policy:
        #   - First encounter + correct → mastered immediately
        #   - Two consecutive correct (after a prior wrong) → mastered
        should_archive = False
        reason = ""
        if correct:
            if q["times_shown"] == 0:
                should_archive = True
                reason = "Nailed it first try"
            elif new_consec >= 2:
                should_archive = True
                reason = "Correct twice in a row"

        if should_archive:
            self.conn.execute(
                "UPDATE questions SET archived = 1 WHERE id = ?",
                (question_id,),
            )

        self.conn.commit()
        return {
            "archived": should_archive,
            "reason": reason,
            "question_id": question_id,
        }

    def set_question_archived(self, question_id: str, archived: bool) -> None:
        self.conn.execute(
            "UPDATE questions SET archived = ? WHERE id = ?",
            (1 if archived else 0, question_id),
        )
        if not archived:
            # Reset streak when un-archiving so it can be earned again
            self.conn.execute(
                "UPDATE questions SET consecutive_correct = 0 WHERE id = ?",
                (question_id,),
            )
        self.conn.commit()

    # ── Reviews (SRS) ─────────────────────────────────────────────────────

    def get_review(self, word: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM reviews WHERE word = ?", (word,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_review(
        self,
        word: str,
        easiness_factor: float,
        interval_days: float,
        repetitions: int,
        next_review: str,
        correct: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_review(word)
        if existing:
            if correct:
                self.conn.execute(
                    "UPDATE reviews SET easiness_factor=?, interval_days=?, "
                    "repetitions=?, next_review=?, last_review=?, "
                    "total_correct=total_correct+1 WHERE word=?",
                    (easiness_factor, interval_days, repetitions, next_review, now, word),
                )
            else:
                self.conn.execute(
                    "UPDATE reviews SET easiness_factor=?, interval_days=?, "
                    "repetitions=?, next_review=?, last_review=?, "
                    "total_incorrect=total_incorrect+1 WHERE word=?",
                    (easiness_factor, interval_days, repetitions, next_review, now, word),
                )
        else:
            tc = 1 if correct else 0
            ti = 0 if correct else 1
            self.conn.execute(
                "INSERT INTO reviews (word, easiness_factor, interval_days, "
                "repetitions, next_review, last_review, total_correct, total_incorrect) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (word, easiness_factor, interval_days, repetitions, next_review, now, tc, ti),
            )
        self.conn.commit()

    def get_due_words(self, limit: int = 50) -> list[dict]:
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute(
            "SELECT * FROM reviews WHERE next_review <= ? "
            "ORDER BY next_review ASC LIMIT ?",
            (now, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_new_words(self, limit: int = 10) -> list[dict]:
        """Words that have never been reviewed."""
        rows = self.conn.execute(
            "SELECT w.* FROM words w LEFT JOIN reviews r ON w.word = r.word "
            "WHERE r.word IS NULL ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_new_cluster_words(self, limit: int = 10) -> list[str]:
        """Cluster words that have never been reviewed (can generate questions).

        Prioritizes words that already have banked questions (instant start),
        then fills with random unreviewed cluster words.
        """
        # First: unreviewed words WITH banked questions (no generation needed)
        banked = self.conn.execute(
            "SELECT DISTINCT q.target_word FROM questions q "
            "JOIN cluster_words cw ON q.target_word = cw.word "
            "LEFT JOIN reviews r ON q.target_word = r.word "
            "WHERE r.word IS NULL ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
        words = [r[0] for r in banked]
        if len(words) >= limit:
            return words[:limit]

        # Then: unreviewed cluster words without banked questions
        seen = set(words)
        remaining = limit - len(words)
        rows = self.conn.execute(
            "SELECT DISTINCT cw.word FROM cluster_words cw "
            "LEFT JOIN reviews r ON cw.word = r.word "
            "WHERE r.word IS NULL ORDER BY RANDOM() LIMIT ?",
            (remaining + len(seen),),  # fetch extra to account for overlap
        ).fetchall()
        for r in rows:
            if r[0] not in seen:
                words.append(r[0])
                seen.add(r[0])
                if len(words) >= limit:
                    break
        return words

    def get_due_cluster_words(self, limit: int = 50) -> list[str]:
        """Cluster words that are due for review."""
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute(
            "SELECT r.word FROM reviews r "
            "JOIN cluster_words cw ON r.word = cw.word "
            "WHERE r.next_review <= ? "
            "ORDER BY r.next_review ASC LIMIT ?",
            (now, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def get_reviewed_word_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM reviews").fetchone()
        return row[0]

    # ── Sessions ──────────────────────────────────────────────────────────

    def start_session(self) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (now,)
        )
        self.conn.commit()
        return cur.lastrowid

    def end_session(self, session_id: int, total: int, correct: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE sessions SET ended_at=?, questions_total=?, questions_correct=? "
            "WHERE id=?",
            (now, total, correct, session_id),
        )
        self.conn.commit()

    def get_session_history(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        word_count = self.get_word_count()
        cluster_count = self.get_cluster_count()
        reviewed = self.get_reviewed_word_count()
        due = len(self.get_due_words(limit=9999))
        bank = self.get_question_bank_size()
        ready = self.get_ready_question_count()
        archived = self.get_archived_question_count()

        sessions = self.conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(questions_total),0) as total_q, "
            "COALESCE(SUM(questions_correct),0) as correct_q FROM sessions "
            "WHERE ended_at IS NOT NULL"
        ).fetchone()

        return {
            "total_words": word_count,
            "total_clusters": cluster_count,
            "words_reviewed": reviewed,
            "words_due": due,
            "words_new": word_count - reviewed,
            "question_bank_size": bank,
            "questions_ready": ready,
            "questions_archived": archived,
            "total_sessions": sessions["cnt"],
            "total_questions_answered": sessions["total_q"],
            "total_correct": sessions["correct_q"],
            "accuracy": (
                round(sessions["correct_q"] / sessions["total_q"] * 100, 1)
                if sessions["total_q"] > 0
                else 0
            ),
        }

    # ── Audio cache ───────────────────────────────────────────────────────

    def get_audio_cache(self, sentence_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT file_path FROM audio_cache WHERE sentence_hash = ?",
            (sentence_hash,),
        ).fetchone()
        return row["file_path"] if row else None

    def set_audio_cache(
        self, sentence_hash: str, file_path: str, tts_provider: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO audio_cache "
            "(sentence_hash, file_path, tts_provider, created_at) VALUES (?, ?, ?, ?)",
            (sentence_hash, file_path, tts_provider, now),
        )
        self.conn.commit()
