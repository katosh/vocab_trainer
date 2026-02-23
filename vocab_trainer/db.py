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

CREATE TABLE IF NOT EXISTS word_progress (
    word TEXT NOT NULL,
    cluster_title TEXT NOT NULL,
    archived INTEGER DEFAULT 0,
    easiness_factor REAL DEFAULT 2.5,
    interval_days REAL DEFAULT 1.0,
    repetitions INTEGER DEFAULT 0,
    next_review TEXT,
    last_review TEXT,
    total_correct INTEGER DEFAULT 0,
    total_incorrect INTEGER DEFAULT 0,
    PRIMARY KEY (word, cluster_title)
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
            "choice_details_json) "
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
                json.dumps(q.choice_details),
            ),
        )
        self.conn.commit()

    def get_question_bank_size(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM questions").fetchone()
        return row[0]

    def get_ready_question_count(self) -> int:
        """Unanswered questions (ready to serve)."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM questions WHERE answered_at IS NULL"
        ).fetchone()
        return row[0]

    def get_archived_question_count(self) -> int:
        """Count archived word-cluster pairs."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM word_progress WHERE archived = 1"
        ).fetchone()
        return row[0]

    def get_questions_for_word(self, word: str, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM questions WHERE target_word = ? ORDER BY RANDOM() LIMIT ?",
            (word, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_session_questions(self, limit: int = 20) -> list[dict]:
        """Pull ready (unanswered) questions for due reviews and new words only.

        Excludes questions for archived word-clusters and word-clusters
        that have been reviewed but are not yet due (SRS timing respected).

        Priority:
        0. Due word-clusters (freshly due first — optimal SRS timing)
        1. Never-reviewed word-clusters (new material)
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""
            SELECT q.*,
                CASE
                    WHEN wp.word IS NOT NULL AND wp.next_review <= ?
                        THEN 0
                    WHEN wp.word IS NULL
                        THEN 1
                END AS priority
            FROM questions q
            LEFT JOIN word_progress wp
                ON q.target_word = wp.word
                AND COALESCE(q.cluster_title, '') = wp.cluster_title
            WHERE q.answered_at IS NULL
              AND (wp.archived IS NULL OR wp.archived = 0)
              AND (wp.word IS NULL OR wp.next_review <= ?)
            ORDER BY priority ASC, wp.next_review DESC, RANDOM()
            LIMIT ?
        """, (now, now, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_review_questions(self, limit: int = 20) -> list[dict]:
        """Ready questions for due word-clusters.

        Freshly-due first (optimal SRS timing), long-overdue last.
        """
        now = datetime.now(timezone.utc).isoformat()
        rows = self.conn.execute("""
            SELECT q.*
            FROM questions q
            JOIN word_progress wp
                ON q.target_word = wp.word
                AND COALESCE(q.cluster_title, '') = wp.cluster_title
            WHERE q.answered_at IS NULL
              AND wp.archived = 0
              AND wp.next_review <= ?
            ORDER BY wp.next_review DESC
            LIMIT ?
        """, (now, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_new_questions(self, limit: int = 10) -> list[dict]:
        """Ready questions for word-clusters with no progress entry (truly new)."""
        rows = self.conn.execute("""
            SELECT q.*
            FROM questions q
            LEFT JOIN word_progress wp
                ON q.target_word = wp.word
                AND COALESCE(q.cluster_title, '') = wp.cluster_title
            WHERE q.answered_at IS NULL
              AND wp.word IS NULL
            ORDER BY RANDOM() LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_active_word_count(self) -> int:
        """Count distinct word-cluster pairs currently in rotation."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM word_progress WHERE archived = 0"
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

    def get_active_questions(self) -> list[dict]:
        """Active (non-archived) word-cluster pairs with SRS info."""
        rows = self.conn.execute("""
            SELECT wp.word AS target_word, wp.cluster_title,
                   wp.total_correct + wp.total_incorrect AS times_shown,
                   wp.total_correct AS times_correct,
                   wp.interval_days, wp.next_review, wp.easiness_factor,
                   wp.last_review
            FROM word_progress wp
            WHERE wp.archived = 0
            ORDER BY wp.next_review ASC
        """).fetchall()
        return [dict(r) for r in rows]

    def get_archived_questions(self) -> list[dict]:
        """Archived word-cluster pairs with SRS info."""
        rows = self.conn.execute("""
            SELECT wp.word AS target_word, wp.cluster_title,
                   wp.total_correct + wp.total_incorrect AS times_shown,
                   wp.total_correct AS times_correct,
                   wp.interval_days, wp.next_review, wp.easiness_factor,
                   wp.last_review
            FROM word_progress wp
            WHERE wp.archived = 1
            ORDER BY wp.last_review DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def reset_word_due(self, word: str, cluster_title: str | None = None) -> None:
        """Reset SRS for a word(-cluster) so it becomes due in 1 day (like a wrong answer)."""
        next_review = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        if cluster_title is not None:
            self.conn.execute(
                "UPDATE word_progress SET next_review = ?, "
                "interval_days = 1.0, repetitions = 0 "
                "WHERE LOWER(word) = LOWER(?) AND cluster_title = ?",
                (next_review, word, cluster_title),
            )
        else:
            self.conn.execute(
                "UPDATE word_progress SET next_review = ?, "
                "interval_days = 1.0, repetitions = 0 "
                "WHERE LOWER(word) = LOWER(?)",
                (next_review, word),
            )
        self.conn.commit()

    # ── Word Progress (SRS) ───────────────────────────────────────────────

    def get_word_progress(self, word: str, cluster_title: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM word_progress WHERE word = ? AND cluster_title = ?",
            (word, cluster_title),
        ).fetchone()
        return dict(row) if row else None

    def upsert_word_progress(
        self,
        word: str,
        cluster_title: str,
        easiness_factor: float,
        interval_days: float,
        repetitions: int,
        next_review: str,
        correct: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self.get_word_progress(word, cluster_title)
        if existing:
            if correct:
                self.conn.execute(
                    "UPDATE word_progress SET easiness_factor=?, interval_days=?, "
                    "repetitions=?, next_review=?, last_review=?, "
                    "total_correct=total_correct+1 "
                    "WHERE word=? AND cluster_title=?",
                    (easiness_factor, interval_days, repetitions, next_review, now,
                     word, cluster_title),
                )
            else:
                self.conn.execute(
                    "UPDATE word_progress SET easiness_factor=?, interval_days=?, "
                    "repetitions=?, next_review=?, last_review=?, "
                    "total_incorrect=total_incorrect+1 "
                    "WHERE word=? AND cluster_title=?",
                    (easiness_factor, interval_days, repetitions, next_review, now,
                     word, cluster_title),
                )
        else:
            tc = 1 if correct else 0
            ti = 0 if correct else 1
            self.conn.execute(
                "INSERT INTO word_progress (word, cluster_title, easiness_factor, "
                "interval_days, repetitions, next_review, last_review, "
                "total_correct, total_incorrect) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (word, cluster_title, easiness_factor, interval_days, repetitions,
                 next_review, now, tc, ti),
            )
        self.conn.commit()

    def set_word_archived(self, word: str, cluster_title: str, archived: bool) -> None:
        """Archive or restore a word-cluster pair."""
        self.conn.execute(
            "UPDATE word_progress SET archived = ? "
            "WHERE word = ? AND cluster_title = ?",
            (1 if archived else 0, word, cluster_title),
        )
        self.conn.commit()

    def get_word_clusters_needing_questions(self) -> list[dict]:
        """Active word-clusters that have no ready (unanswered) question.

        Ordered by next_review so due words generate first.
        """
        rows = self.conn.execute("""
            SELECT wp.word, wp.cluster_title
            FROM word_progress wp
            LEFT JOIN questions q
                ON q.target_word = wp.word
                AND COALESCE(q.cluster_title, '') = wp.cluster_title
                AND q.answered_at IS NULL
            WHERE wp.archived = 0
              AND q.id IS NULL
            ORDER BY wp.next_review ASC
        """).fetchall()
        return [dict(r) for r in rows]


    def get_new_word_clusters_with_ready_count(self) -> int:
        """Count distinct new (no word_progress) word-clusters that have a ready question."""
        row = self.conn.execute("""
            SELECT COUNT(DISTINCT q.target_word || '|' || COALESCE(q.cluster_title, ''))
            FROM questions q
            LEFT JOIN word_progress wp
                ON q.target_word = wp.word
                AND COALESCE(q.cluster_title, '') = wp.cluster_title
            WHERE q.answered_at IS NULL
              AND wp.word IS NULL
              AND q.cluster_title IS NOT NULL
              AND q.cluster_title != ''
        """).fetchone()
        return row[0]

    def get_new_word_clusters_without_questions(self, limit: int = 20) -> list[dict]:
        """New word-clusters (no word_progress, not archived) that have no ready question.

        Only includes pairs from clusters with >= 4 words.
        Returns (word, cluster_title) pairs for generation targeting.
        """
        rows = self.conn.execute("""
            SELECT cw.word, c.title AS cluster_title
            FROM cluster_words cw
            JOIN clusters c ON cw.cluster_id = c.id
            LEFT JOIN word_progress wp
                ON cw.word = wp.word AND c.title = wp.cluster_title
            LEFT JOIN questions q
                ON q.target_word = cw.word
                AND q.cluster_title = c.title
                AND q.answered_at IS NULL
            WHERE wp.word IS NULL
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

    def get_random_words(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM words ORDER BY RANDOM() LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_word_cluster_question_counts(self) -> list[dict]:
        """All (word, cluster) pairs with their ready (unanswered) question count.

        Only includes pairs from clusters with >= 4 words (enough for 4 choices).
        Excludes archived word-clusters.
        """
        rows = self.conn.execute("""
            SELECT cw.word, cw.meaning, cw.distinction,
                   c.id AS cluster_id, c.title AS cluster_title,
                   COUNT(q.id) AS question_count
            FROM cluster_words cw
            JOIN clusters c ON cw.cluster_id = c.id
            LEFT JOIN questions q
                ON q.target_word = cw.word
                AND q.cluster_title = c.title
                AND q.answered_at IS NULL
            LEFT JOIN word_progress wp
                ON wp.word = cw.word
                AND wp.cluster_title = c.title
            WHERE c.id IN (
                SELECT cluster_id FROM cluster_words
                GROUP BY cluster_id HAVING COUNT(*) >= 4
            )
            AND (wp.archived IS NULL OR wp.archived = 0)
            GROUP BY cw.word, c.id
        """).fetchall()
        return [dict(r) for r in rows]

    def get_reviewed_word_count(self) -> int:
        """Count of word-cluster pairs with any review history."""
        row = self.conn.execute("SELECT COUNT(*) FROM word_progress").fetchone()
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
        bank = self.get_question_bank_size()
        ready = self.get_ready_question_count()
        archived = self.get_archived_question_count()
        active = self.get_active_word_count()

        # Due word-clusters
        now = datetime.now(timezone.utc).isoformat()
        due_row = self.conn.execute(
            "SELECT COUNT(*) FROM word_progress "
            "WHERE archived = 0 AND next_review <= ?",
            (now,),
        ).fetchone()
        due = due_row[0]

        # Count accuracy from word_progress
        review_stats = self.conn.execute(
            "SELECT COALESCE(SUM(total_correct),0) as correct, "
            "COALESCE(SUM(total_correct + total_incorrect),0) as total "
            "FROM word_progress"
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
            "WHERE answered_at IS NULL"
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
