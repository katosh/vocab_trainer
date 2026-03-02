from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
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
    choice_details_json TEXT DEFAULT '[]',
    answered_at TEXT,
    chosen_index INTEGER,
    was_correct INTEGER,
    response_time_ms INTEGER,
    session_id INTEGER
);

CREATE TABLE IF NOT EXISTS cluster_progress (
    cluster_title TEXT PRIMARY KEY,
    archived INTEGER DEFAULT 0,
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


def _lookup_cluster_word(cw_map: dict[str, dict], choice: str) -> dict | None:
    """Match a choice to a cluster word, trying stem stripping for inflected forms."""
    key = choice.lower()
    if key in cw_map:
        return cw_map[key]
    for suffix in ("ed", "d", "ing", "s", "es", "ly"):
        if key.endswith(suffix) and key[: -len(suffix)] in cw_map:
            return cw_map[key[: -len(suffix)]]
    return None


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        # Add quality_issue column (safe migration for existing DBs)
        try:
            self.conn.execute("ALTER TABLE questions ADD COLUMN quality_issue TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        # Migrate word_progress → cluster_progress (pessimistic merge)
        self._migrate_word_progress_to_cluster()
        self.conn.commit()

    def _migrate_word_progress_to_cluster(self) -> None:
        """One-time migration: merge word_progress rows into cluster_progress.

        Groups by cluster_title and takes the pessimistic (weakest) values:
        MIN easiness_factor, MIN interval_days, MIN repetitions, earliest next_review.
        The old word_progress table is left intact for rollback safety.
        """
        # Check if old table exists
        has_old = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='word_progress'"
        ).fetchone()
        if not has_old:
            return
        # Check if we already migrated (cluster_progress has data)
        cp_count = self.conn.execute("SELECT COUNT(*) FROM cluster_progress").fetchone()[0]
        if cp_count > 0:
            return
        # Check if there's actually data to migrate
        wp_count = self.conn.execute("SELECT COUNT(*) FROM word_progress").fetchone()[0]
        if wp_count == 0:
            return
        self.conn.execute("""
            INSERT INTO cluster_progress
                (cluster_title, archived, easiness_factor, interval_days, repetitions,
                 next_review, last_review, total_correct, total_incorrect)
            SELECT
                cluster_title,
                MAX(archived),
                MIN(easiness_factor),
                MIN(interval_days),
                MIN(repetitions),
                MIN(next_review),
                MAX(last_review),
                SUM(total_correct),
                SUM(total_incorrect)
            FROM word_progress
            GROUP BY cluster_title
        """)

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
            "choice_details_json, quality_issue) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                json.dumps(q.choice_details),
                q.quality_issue,
            ),
        )
        self.conn.commit()

    def get_question_bank_size(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM questions").fetchone()
        return row[0]

    def get_ready_question_count(self) -> int:
        """Unanswered questions (ready to serve)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE answered_at IS NULL AND quality_issue IS NULL"
        ).fetchone()
        return row[0]

    def get_archived_question_count(self) -> int:
        """Count archived clusters."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM cluster_progress WHERE archived = 1"
        ).fetchone()
        return row[0]

    def get_questions_for_word(self, word: str, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM questions WHERE target_word = ? ORDER BY RANDOM() LIMIT ?",
            (word, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_questions(self, limit: int = 20) -> list[dict]:
        """Pull ready (unanswered) questions for due reviews and new clusters only.

        Excludes questions for archived clusters and clusters
        that have been reviewed but are not yet due (SRS timing respected).

        Priority:
        0. Due clusters (freshly due first — optimal SRS timing)
        1. Never-reviewed clusters (new material)
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""
            SELECT q.*,
                CASE
                    WHEN cp.cluster_title IS NOT NULL AND cp.next_review <= ?
                        THEN 0
                    WHEN cp.cluster_title IS NULL
                        THEN 1
                END AS priority
            FROM questions q
            LEFT JOIN cluster_progress cp
                ON COALESCE(q.cluster_title, '') = cp.cluster_title
            WHERE q.answered_at IS NULL
              AND q.quality_issue IS NULL
              AND (cp.archived IS NULL OR cp.archived = 0)
              AND (cp.cluster_title IS NULL OR cp.next_review <= ?)
            ORDER BY priority ASC, cp.next_review DESC, RANDOM()
            LIMIT ?
        """, (now, now, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_review_questions(self, limit: int = 20) -> list[dict]:
        """Ready questions for due clusters.

        Freshly-due first (optimal SRS timing), long-overdue last.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""
            SELECT q.*
            FROM questions q
            JOIN cluster_progress cp
                ON COALESCE(q.cluster_title, '') = cp.cluster_title
            WHERE q.answered_at IS NULL
              AND q.quality_issue IS NULL
              AND cp.archived = 0
              AND cp.next_review <= ?
            ORDER BY cp.next_review DESC
            LIMIT ?
        """, (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_new_questions(self, limit: int = 10) -> list[dict]:
        """Ready questions for clusters with no progress entry (truly new)."""
        rows = self.conn.execute("""
            SELECT q.*
            FROM questions q
            LEFT JOIN cluster_progress cp
                ON COALESCE(q.cluster_title, '') = cp.cluster_title
            WHERE q.answered_at IS NULL
              AND q.quality_issue IS NULL
              AND cp.cluster_title IS NULL
            ORDER BY RANDOM() LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_active_cluster_count(self) -> int:
        """Count clusters currently in rotation."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM cluster_progress WHERE archived = 0"
        ).fetchone()
        return row[0]

    def mark_question_answered(
        self,
        question_id: str,
        chosen_index: int,
        was_correct: bool,
        response_time_ms: int | None = None,
        session_id: int | None = None,
    ) -> None:
        """Mark a question as answered (historical record)."""
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE questions SET answered_at=?, chosen_index=?, was_correct=?, "
            "response_time_ms=?, session_id=? WHERE id=?",
            (now, chosen_index, 1 if was_correct else 0,
             response_time_ms, session_id, question_id),
        )
        self.conn.commit()

    def get_active_clusters(self) -> list[dict]:
        """Active (non-archived) clusters with SRS info and per-word accuracy."""
        rows = self.conn.execute("""
            SELECT cp.cluster_title,
                   cp.total_correct + cp.total_incorrect AS times_shown,
                   cp.total_correct AS times_correct,
                   cp.interval_days, cp.next_review, cp.easiness_factor,
                   cp.last_review
            FROM cluster_progress cp
            WHERE cp.archived = 0
            ORDER BY cp.next_review ASC
        """).fetchall()
        result = []
        for r in rows:
            entry = dict(r)
            entry["words"] = self.get_cluster_word_accuracy(entry["cluster_title"])
            result.append(entry)
        return result

    def get_archived_clusters(self) -> list[dict]:
        """Archived clusters with SRS info and per-word accuracy."""
        rows = self.conn.execute("""
            SELECT cp.cluster_title,
                   cp.total_correct + cp.total_incorrect AS times_shown,
                   cp.total_correct AS times_correct,
                   cp.interval_days, cp.next_review, cp.easiness_factor,
                   cp.last_review
            FROM cluster_progress cp
            WHERE cp.archived = 1
            ORDER BY cp.last_review DESC
        """).fetchall()
        result = []
        for r in rows:
            entry = dict(r)
            entry["words"] = self.get_cluster_word_accuracy(entry["cluster_title"])
            result.append(entry)
        return result

    def reset_cluster_due(self, cluster_title: str) -> None:
        """Reset SRS for a cluster so it becomes due in 1 day (like a wrong answer)."""
        next_review = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        self.conn.execute(
            "UPDATE cluster_progress SET next_review = ?, "
            "interval_days = 1.0, repetitions = 0 "
            "WHERE cluster_title = ?",
            (next_review, cluster_title),
        )
        self.conn.commit()

    # ── Cluster Progress (SRS) ────────────────────────────────────────────

    def get_cluster_progress(self, cluster_title: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM cluster_progress WHERE cluster_title = ?",
            (cluster_title,),
        ).fetchone()
        return dict(row) if row else None

    def upsert_cluster_progress(
        self,
        cluster_title: str,
        easiness_factor: float,
        interval_days: float,
        repetitions: int,
        next_review: str,
        correct: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_cluster_progress(cluster_title)
        if existing:
            if correct:
                self.conn.execute(
                    "UPDATE cluster_progress SET easiness_factor=?, interval_days=?, "
                    "repetitions=?, next_review=?, last_review=?, "
                    "total_correct=total_correct+1 "
                    "WHERE cluster_title=?",
                    (easiness_factor, interval_days, repetitions, next_review, now,
                     cluster_title),
                )
            else:
                self.conn.execute(
                    "UPDATE cluster_progress SET easiness_factor=?, interval_days=?, "
                    "repetitions=?, next_review=?, last_review=?, "
                    "total_incorrect=total_incorrect+1 "
                    "WHERE cluster_title=?",
                    (easiness_factor, interval_days, repetitions, next_review, now,
                     cluster_title),
                )
        else:
            tc = 1 if correct else 0
            ti = 0 if correct else 1
            self.conn.execute(
                "INSERT INTO cluster_progress (cluster_title, easiness_factor, "
                "interval_days, repetitions, next_review, last_review, "
                "total_correct, total_incorrect) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (cluster_title, easiness_factor, interval_days, repetitions,
                 next_review, now, tc, ti),
            )
        self.conn.commit()

    def set_cluster_archived(self, cluster_title: str, archived: bool) -> None:
        """Archive or restore a cluster."""
        self.conn.execute(
            "UPDATE cluster_progress SET archived = ? "
            "WHERE cluster_title = ?",
            (1 if archived else 0, cluster_title),
        )
        self.conn.commit()

    def get_clusters_needing_questions(self) -> list[dict]:
        """Active clusters that have no ready (unanswered) question.

        Ordered by next_review so due clusters generate first.
        """
        rows = self.conn.execute("""
            SELECT cp.cluster_title
            FROM cluster_progress cp
            LEFT JOIN questions q
                ON COALESCE(q.cluster_title, '') = cp.cluster_title
                AND q.answered_at IS NULL
                AND q.quality_issue IS NULL
            WHERE cp.archived = 0
              AND q.id IS NULL
            ORDER BY cp.next_review ASC
        """).fetchall()
        return [dict(r) for r in rows]


    def get_new_clusters_with_ready_count(self) -> int:
        """Count distinct new (no cluster_progress) clusters that have a ready question."""
        row = self.conn.execute("""
            SELECT COUNT(DISTINCT COALESCE(q.cluster_title, ''))
            FROM questions q
            LEFT JOIN cluster_progress cp
                ON COALESCE(q.cluster_title, '') = cp.cluster_title
            WHERE q.answered_at IS NULL
              AND q.quality_issue IS NULL
              AND cp.cluster_title IS NULL
              AND q.cluster_title IS NOT NULL
              AND q.cluster_title != ''
        """).fetchone()
        return row[0]

    def get_new_clusters_without_questions(self, limit: int = 20) -> list[dict]:
        """New clusters (no cluster_progress, not archived) that have no ready question.

        Only includes clusters with >= 4 words.
        Returns cluster_title values for generation targeting.
        """
        rows = self.conn.execute("""
            SELECT DISTINCT c.title AS cluster_title
            FROM clusters c
            LEFT JOIN cluster_progress cp
                ON c.title = cp.cluster_title
            LEFT JOIN questions q
                ON q.cluster_title = c.title
                AND q.answered_at IS NULL
                AND q.quality_issue IS NULL
            WHERE cp.cluster_title IS NULL
              AND q.id IS NULL
              AND c.id IN (
                  SELECT cluster_id FROM cluster_words
                  GROUP BY cluster_id HAVING COUNT(*) >= 4
              )
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_question_history(self, word: str, cluster_title: str) -> list[dict]:
        """Answered questions for a word-cluster pair, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM questions "
            "WHERE target_word = ? AND COALESCE(cluster_title, '') = ? "
            "AND answered_at IS NOT NULL "
            "ORDER BY answered_at DESC",
            (word, cluster_title),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_random_words(
        self, limit: int = 10, exclude: list[str] | None = None,
    ) -> list[dict]:
        if exclude:
            placeholders = ",".join("?" * len(exclude))
            rows = self.conn.execute(
                f"SELECT * FROM words WHERE word NOT IN ({placeholders}) "
                "ORDER BY RANDOM() LIMIT ?",
                (*exclude, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM words ORDER BY RANDOM() LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_word_accuracy(self, cluster_title: str) -> list[dict]:
        """Per-word accuracy from answered questions for a cluster.

        Returns list of {word, total, correct} for target word weighting
        and Library display.
        """
        rows = self.conn.execute("""
            SELECT q.target_word AS word,
                   COUNT(*) AS total,
                   SUM(CASE WHEN q.was_correct = 1 THEN 1 ELSE 0 END) AS correct
            FROM questions q
            WHERE q.cluster_title = ?
              AND q.answered_at IS NOT NULL
            GROUP BY q.target_word
        """, (cluster_title,)).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_question_counts(self) -> list[dict]:
        """All clusters with their ready (unanswered) question count.

        Only includes clusters with >= 4 words (enough for 4 choices).
        Excludes archived clusters.
        """
        rows = self.conn.execute("""
            SELECT c.id AS cluster_id, c.title AS cluster_title,
                   COUNT(q.id) AS question_count
            FROM clusters c
            LEFT JOIN questions q
                ON q.cluster_title = c.title
                AND q.answered_at IS NULL
                AND q.quality_issue IS NULL
            LEFT JOIN cluster_progress cp
                ON cp.cluster_title = c.title
            WHERE c.id IN (
                SELECT cluster_id FROM cluster_words
                GROUP BY cluster_id HAVING COUNT(*) >= 4
            )
            AND (cp.archived IS NULL OR cp.archived = 0)
            GROUP BY c.id
        """).fetchall()
        return [dict(r) for r in rows]

    def get_reviewed_cluster_count(self) -> int:
        """Count of clusters with any review history."""
        row = self.conn.execute("SELECT COUNT(*) FROM cluster_progress").fetchone()
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
        reviewed = self.get_reviewed_cluster_count()
        bank = self.get_question_bank_size()
        ready = self.get_ready_question_count()
        archived = self.get_archived_question_count()
        active = self.get_active_cluster_count()

        # Due clusters
        now = datetime.now(timezone.utc).isoformat()
        due_row = self.conn.execute(
            "SELECT COUNT(*) FROM cluster_progress "
            "WHERE archived = 0 AND next_review <= ?",
            (now,),
        ).fetchone()
        due = due_row[0]

        # Count accuracy from cluster_progress
        review_stats = self.conn.execute(
            "SELECT COALESCE(SUM(total_correct),0) as correct, "
            "COALESCE(SUM(total_correct + total_incorrect),0) as total "
            "FROM cluster_progress"
        ).fetchone()

        sessions = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE ended_at IS NOT NULL"
        ).fetchone()

        total_answered = review_stats["total"]
        total_correct = review_stats["correct"]

        return {
            "total_words": word_count,
            "total_clusters": cluster_count,
            "clusters_reviewed": reviewed,
            "clusters_due": due,
            "clusters_new": cluster_count - reviewed,
            "question_bank_size": bank,
            "questions_ready": ready,
            "clusters_archived": archived,
            "clusters_active": active,
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

    # ── Question content update ────────────────────────────────────────────

    def update_question_content(
        self,
        question_id: str,
        stem: str,
        choices: list[str],
        correct_index: int,
        explanation: str,
        context_sentence: str,
        choice_details: list[dict],
    ) -> None:
        """Update only the content columns of a question, preserving answer history.

        Unlike save_question() (INSERT OR REPLACE), this keeps answered_at,
        chosen_index, was_correct, response_time_ms, and session_id intact.
        """
        self.conn.execute(
            "UPDATE questions SET stem=?, choices_json=?, correct_index=?, "
            "explanation=?, context_sentence=?, choice_details_json=?, "
            "generated_at=? WHERE id=?",
            (
                stem,
                json.dumps(choices),
                correct_index,
                explanation,
                context_sentence,
                json.dumps(choice_details),
                datetime.now(timezone.utc).isoformat(),
                question_id,
            ),
        )
        self.conn.commit()

    def get_all_questions_ordered(self) -> list[dict]:
        """All questions ordered by cluster_title, target_word for batch processing."""
        rows = self.conn.execute(
            "SELECT * FROM questions ORDER BY cluster_title, target_word"
        ).fetchall()
        return [dict(r) for r in rows]

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

    def get_question_audio_texts(self) -> list[str]:
        """Return all explanation and context_sentence texts from unanswered questions."""
        rows = self.conn.execute(
            "SELECT explanation, context_sentence FROM questions "
            "WHERE answered_at IS NULL AND quality_issue IS NULL"
        ).fetchall()
        texts = []
        for r in rows:
            if r["explanation"]:
                texts.append(r["explanation"])
            if r["context_sentence"]:
                texts.append(r["context_sentence"])
        return texts

    def delete_audio_cache(self, sentence_hash: str) -> None:
        self.conn.execute(
            "DELETE FROM audio_cache WHERE sentence_hash = ?",
            (sentence_hash,),
        )
        self.conn.commit()
