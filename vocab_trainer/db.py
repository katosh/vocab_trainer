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
    commentary TEXT,
    source_file TEXT
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

CREATE TABLE IF NOT EXISTS file_mtimes (
    file_path TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL
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
        # clusters.source_file
        cursor = self.conn.execute("PRAGMA table_info(clusters)")
        cluster_cols = {row[1] for row in cursor.fetchall()}
        if "source_file" not in cluster_cols:
            self.conn.execute(
                "ALTER TABLE clusters ADD COLUMN source_file TEXT"
            )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ── Import ────────────────────────────────────────────────────────────

    def delete_words_by_source(self, source_file: str) -> int:
        """Remove all words originally imported from *source_file*."""
        cur = self.conn.execute(
            "DELETE FROM words WHERE source_file = ?", (source_file,)
        )
        self.conn.commit()
        return cur.rowcount

    def delete_clusters_by_source(self, source_file: str) -> int:
        """Remove clusters (and their entries) from *source_file*."""
        ids = [
            row[0]
            for row in self.conn.execute(
                "SELECT id FROM clusters WHERE source_file = ?", (source_file,)
            ).fetchall()
        ]
        for cid in ids:
            self.conn.execute("DELETE FROM cluster_words WHERE cluster_id = ?", (cid,))
        self.conn.execute("DELETE FROM clusters WHERE source_file = ?", (source_file,))
        self.conn.commit()
        return len(ids)

    def import_words(self, words: list[VocabWord]) -> int:
        count = 0
        for w in words:
            self.conn.execute(
                "INSERT OR IGNORE INTO words (word, definition, section, source_file) "
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
                "INSERT OR REPLACE INTO clusters (title, preamble, commentary, source_file) "
                "VALUES (?, ?, ?, ?)",
                (c.title, c.preamble, c.commentary, c.source_file),
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
                    (e.word, e.meaning, c.title, c.source_file),
                )
            count += 1
        self.conn.commit()
        return count

    # ── File mtimes ─────────────────────────────────────────────────────

    def get_file_mtime(self, file_path: str) -> int | None:
        row = self.conn.execute(
            "SELECT mtime_ns FROM file_mtimes WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def set_file_mtime(self, file_path: str, mtime_ns: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO file_mtimes (file_path, mtime_ns) VALUES (?, ?)",
            (file_path, mtime_ns),
        )
        self.conn.commit()

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
        """Never-shown, non-archived questions (new material buffer)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE archived = 0 AND times_shown = 0"
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
        0. Due words the user previously got wrong (struggling)
        1. Other due words (spaced repetition schedule)
        2. Never-reviewed words (new material)
        3. Reviewed but not yet due (fallback — better than on-the-fly generation)
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
            WHERE q.archived = 0
            ORDER BY priority ASC, q.times_shown ASC, RANDOM()
            LIMIT ?
        """, (now, now, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_review_questions(self, limit: int = 20) -> list[dict]:
        """Non-archived questions for words already in rotation and due for review.

        Includes never-shown questions for due words (e.g. freshly generated
        questions for a word that already has review history).
        Priority: struggling words first, then by staleness.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""
            SELECT q.*
            FROM questions q
            JOIN reviews r ON q.target_word = r.word
            WHERE q.archived = 0
              AND r.next_review <= ?
            ORDER BY
                CASE WHEN r.total_incorrect > r.total_correct THEN 0 ELSE 1 END ASC,
                r.next_review ASC
            LIMIT ?
        """, (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_new_questions(self, limit: int = 10) -> list[dict]:
        """Non-archived questions for words with no review history (truly new)."""
        rows = self.conn.execute(
            "SELECT * FROM questions WHERE archived = 0 AND times_shown = 0 "
            "AND target_word NOT IN (SELECT word FROM reviews) "
            "ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_active_word_new_questions(self, limit: int = 10, exclude_words: set[str] | None = None) -> list[dict]:
        """Unseen questions for words already in rotation (reinforcement, not new material)."""
        rows = self.conn.execute(
            "SELECT q.* FROM questions q "
            "WHERE q.archived = 0 AND q.times_shown = 0 "
            "AND q.target_word IN ("
            "  SELECT DISTINCT target_word FROM questions "
            "  WHERE archived = 0 AND times_shown > 0"
            ") "
            "ORDER BY RANDOM() LIMIT ?",
            (limit,),
        ).fetchall()
        results = [dict(r) for r in rows]
        if exclude_words:
            lc = {w.lower() for w in exclude_words}
            results = [r for r in results if r["target_word"].lower() not in lc]
        return results

    def get_active_word_count(self) -> int:
        """Count distinct words currently in rotation (shown but not archived)."""
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT target_word) FROM questions "
            "WHERE archived = 0 AND times_shown > 0"
        ).fetchone()
        return row[0]

    def record_question_shown(self, question_id: str, correct: bool, archive_interval_days: int = 21) -> dict:
        """Record answer and auto-archive when SRS interval reaches threshold.

        Archives when the word's review interval_days >= archive_interval_days
        after a correct answer (meaning sustained mastery over multiple reviews).

        Returns {"archived": bool, "reason": str, "question_id": str}.
        """
        q = self.conn.execute(
            "SELECT * FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
        if not q:
            return {"archived": False, "reason": "", "question_id": question_id}

        q = dict(q)
        if correct:
            self.conn.execute(
                "UPDATE questions SET times_shown = times_shown + 1, "
                "times_correct = times_correct + 1 WHERE id = ?",
                (question_id,),
            )
        else:
            self.conn.execute(
                "UPDATE questions SET times_shown = times_shown + 1 WHERE id = ?",
                (question_id,),
            )

        # Archive policy: archive when SRS interval proves sustained mastery
        should_archive = False
        reason = ""
        interval_days = 0.0
        review = self.get_review(q["target_word"])
        if review:
            interval_days = review["interval_days"]
        if correct and interval_days >= archive_interval_days:
            should_archive = True
            reason = f"Mastered (interval {interval_days:.0f} days)"

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
            "interval_days": round(interval_days, 1),
            "archive_threshold": archive_interval_days,
        }

    def set_question_archived(self, question_id: str, archived: bool) -> None:
        self.conn.execute(
            "UPDATE questions SET archived = ? WHERE id = ?",
            (1 if archived else 0, question_id),
        )
        self.conn.commit()

    def get_active_questions(self) -> list[dict]:
        """Active (non-archived) questions that have been shown at least once, with SRS info."""
        rows = self.conn.execute("""
            SELECT q.id, q.target_word, q.question_type, q.stem, q.cluster_title,
                   q.times_shown, q.times_correct,
                   r.interval_days, r.next_review, r.easiness_factor
            FROM questions q
            LEFT JOIN reviews r ON LOWER(q.target_word) = LOWER(r.word)
            WHERE q.archived = 0 AND q.times_shown > 0
            ORDER BY r.next_review ASC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_archived_questions(self) -> list[dict]:
        """Archived questions with SRS info, most-recently-reviewed first."""
        rows = self.conn.execute("""
            SELECT q.id, q.target_word, q.question_type, q.stem, q.cluster_title,
                   q.times_shown, q.times_correct,
                   r.interval_days, r.next_review, r.easiness_factor
            FROM questions q
            LEFT JOIN reviews r ON LOWER(q.target_word) = LOWER(r.word)
            WHERE q.archived = 1
            ORDER BY r.last_review DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def reset_word_due(self, word: str) -> None:
        """Reset SRS for a word so it becomes due immediately."""
        self.conn.execute(
            "UPDATE reviews SET next_review = datetime('now'), interval_days = 1.0, repetitions = 0 "
            "WHERE LOWER(word) = LOWER(?)",
            (word,),
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
        active = self.get_active_word_count()

        # Count accuracy from the reviews table (always recorded, unlike sessions)
        review_stats = self.conn.execute(
            "SELECT COALESCE(SUM(total_correct),0) as correct, "
            "COALESCE(SUM(total_correct + total_incorrect),0) as total "
            "FROM reviews"
        ).fetchone()

        sessions = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions"
        ).fetchone()

        total_answered = review_stats["total"]
        total_correct = review_stats["correct"]

        return {
            "total_words": word_count,
            "total_clusters": cluster_count,
            "words_reviewed": reviewed,
            "words_due": due,
            "words_new": word_count - reviewed,
            "question_bank_size": bank,
            "questions_ready": ready,
            "questions_archived": archived,
            "active_words": active,
            "new_questions": ready,
            "total_sessions": sessions["cnt"],
            "total_questions_answered": total_answered,
            "total_correct": total_correct,
            "accuracy": (
                round(total_correct / total_answered * 100, 1)
                if total_answered > 0
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
