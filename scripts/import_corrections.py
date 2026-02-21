"""Import corrected questions from questions_export.json back into progress.db.

Only updates content fields (stem, choices, explanations, context_sentence, archived).
Preserves review stats (times_shown, times_correct, consecutive_correct, generated_at).

Usage:
    uv run python scripts/import_corrections.py [--dry-run]
"""
import json
import sqlite3
import sys
from pathlib import Path

EXPORT_PATH = Path("questions_export.json")
DB_PATH = Path("progress.db")

# Fields we update (content only, not review stats)
CONTENT_FIELDS = {
    "stem":                     "stem",
    "choices":                  "choices_json",       # array → JSON string
    "correct_index":            "correct_index",
    "explanation":              "explanation",
    "context_sentence":         "context_sentence",
    "choice_explanations":      "choice_explanations_json",  # array → JSON string
    "archived":                 "archived",
}


def main():
    dry_run = "--dry-run" in sys.argv

    data = json.loads(EXPORT_PATH.read_text())
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    updated = 0
    skipped = 0

    for q in data:
        qid = q["id"]
        row = conn.execute("SELECT * FROM questions WHERE id = ?", (qid,)).fetchone()
        if row is None:
            print(f"  SKIP {q['target_word']} ({qid[:8]}…) — not in DB")
            skipped += 1
            continue

        changes = {}
        for export_key, db_col in CONTENT_FIELDS.items():
            new_val = q[export_key]
            # Convert arrays to JSON strings for DB storage
            if isinstance(new_val, list):
                new_val = json.dumps(new_val)
            old_val = row[db_col]
            if str(new_val) != str(old_val):
                changes[db_col] = new_val

        if not changes:
            continue

        word = q["target_word"]
        for col, val in changes.items():
            trunc = str(val)[:80]
            print(f"  [{word}] {col}: {trunc}")

        if not dry_run:
            set_clause = ", ".join(f"{col} = ?" for col in changes)
            values = list(changes.values()) + [qid]
            conn.execute(
                f"UPDATE questions SET {set_clause} WHERE id = ?",
                values,
            )
        updated += 1

    if not dry_run:
        conn.commit()
    conn.close()

    print(f"\nUpdated {updated} questions, skipped {skipped}")
    if dry_run:
        print("(dry run — no changes written)")


if __name__ == "__main__":
    main()
