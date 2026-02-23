"""CLI entry point for vocab-trainer.

Usage:
  uv run python -m vocab_trainer serve [--port PORT] [--no-auto-import]
  uv run python -m vocab_trainer stop
  uv run python -m vocab_trainer restart [--port PORT]
  uv run python -m vocab_trainer status
  uv run python -m vocab_trainer import
  uv run python -m vocab_trainer generate [--count N]
  uv run python -m vocab_trainer regenerate [--batch N] [--dry-run]
  uv run python -m vocab_trainer stats
"""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

PID_FILE = Path(__file__).resolve().parent.parent / ".server.pid"


def main():
    args = sys.argv[1:]
    command = args[0] if args else "serve"

    if command == "serve":
        _serve(args[1:])
    elif command == "stop":
        _stop()
    elif command == "restart":
        _restart(args[1:])
    elif command == "status":
        _status()
    elif command == "import":
        _import_vocab()
    elif command == "generate":
        _generate(args[1:])
    elif command == "regenerate":
        _regenerate(args[1:])
    elif command == "stats":
        _stats()
    else:
        print(f"Unknown command: {command}")
        print("Commands: serve, stop, restart, status, import, generate, regenerate, stats")
        sys.exit(1)


def _parse_flag(args: list[str], name: str, default: str) -> str:
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
    return default


def _read_pid() -> int | None:
    """Read PID from file, return None if stale or missing."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check if process is actually running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


def _write_pid() -> None:
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    PID_FILE.unlink(missing_ok=True)


def _stop() -> bool:
    """Stop a running server. Returns True if a server was stopped."""
    pid = _read_pid()
    if pid is None:
        print("Server is not running.")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped server (PID {pid}).")
        PID_FILE.unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        print("Server was not running (stale PID file removed).")
        PID_FILE.unlink(missing_ok=True)
        return False


def _status():
    pid = _read_pid()
    if pid is None:
        print("Server is not running.")
    else:
        print(f"Server is running (PID {pid}).")


def _restart(args: list[str]):
    import time
    _stop()
    time.sleep(1)
    _serve(args)


def _serve(args: list[str]):
    import uvicorn

    existing = _read_pid()
    if existing is not None:
        print(f"Server already running (PID {existing}). Use 'restart' or 'stop' first.")
        sys.exit(1)

    if "--no-auto-import" in args:
        os.environ["VOCAB_TRAINER_NO_AUTO_IMPORT"] = "1"

    port = int(_parse_flag(args, "--port", "8765"))
    host = _parse_flag(args, "--host", "127.0.0.1")
    _write_pid()

    if host == "0.0.0.0":
        import socket
        local_ip = socket.gethostbyname(socket.gethostname())
        print(f"Starting Vocab Trainer on http://{local_ip}:{port}")
    else:
        print(f"Starting Vocab Trainer on http://{host}:{port}")
    print("Press Ctrl+C to stop\n")
    try:
        uvicorn.run(
            "vocab_trainer.app:app",
            host=host,
            port=port,
            reload=False,
            timeout_graceful_shutdown=5,
        )
    finally:
        _remove_pid()
        os.environ.pop("VOCAB_TRAINER_NO_AUTO_IMPORT", None)


def _import_vocab():
    from vocab_trainer.config import load_settings
    from vocab_trainer.db import Database
    from vocab_trainer.parsers.distinctions_parser import parse_distinctions_file
    from vocab_trainer.parsers.vocabulary_parser import parse_vocabulary_file

    settings = load_settings()
    db = Database(settings.db_full_path)

    total_words = 0
    total_clusters = 0

    for vf in settings.resolved_vocab_files():
        if not vf.exists():
            print(f"  Skipping (not found): {vf}")
            continue
        print(f"  Parsing: {vf.name}")
        if "distinctions" in vf.name:
            db.delete_clusters_by_source(vf.name)
            clusters = parse_distinctions_file(vf)
            n = db.import_clusters(clusters)
            total_clusters += n
            entries = sum(len(c.entries) for c in clusters)
            print(f"    {n} clusters, {entries} entries")
        else:
            db.delete_words_by_source(vf.name)
            words = parse_vocabulary_file(vf)
            n = db.import_words(words)
            total_words += n
            print(f"    {n} words")
        db.set_file_mtime(str(vf), vf.stat().st_mtime_ns)

    print(f"\nTotal in DB: {db.get_word_count()} words, {db.get_cluster_count()} clusters")
    db.close()


def _generate(args: list[str]):
    count = 10
    for i, a in enumerate(args):
        if a == "--count" and i + 1 < len(args):
            count = int(args[i + 1])

    from vocab_trainer.config import load_settings
    from vocab_trainer.db import Database

    settings = load_settings()
    db = Database(settings.db_full_path)

    # Check we have data
    if db.get_word_count() == 0:
        print("No words in database. Run 'import' first.")
        db.close()
        sys.exit(1)

    print(f"Generating {count} questions using {settings.llm_provider}...")

    if settings.llm_provider == "ollama":
        from vocab_trainer.providers.llm_ollama import OllamaProvider
        llm = OllamaProvider(base_url=settings.ollama_url, model=settings.llm_model)
    elif settings.llm_provider == "anthropic":
        from vocab_trainer.providers.llm_anthropic import AnthropicProvider
        llm = AnthropicProvider()
    elif settings.llm_provider == "openai":
        from vocab_trainer.providers.llm_openai import OpenAIProvider
        llm = OpenAIProvider()
    else:
        print(f"Unknown LLM provider: {settings.llm_provider}")
        sys.exit(1)

    from vocab_trainer.question_generator import generate_batch
    questions = asyncio.run(generate_batch(llm, db, count=count))

    print(f"\nGenerated {len(questions)} questions")
    print(f"Question bank size: {db.get_question_bank_size()}")
    db.close()


def _regenerate(args: list[str]):
    import json as _json
    from datetime import datetime, timezone

    batch_size = int(_parse_flag(args, "--batch", "10"))
    dry_run = "--dry-run" in args

    from vocab_trainer.config import load_settings
    from vocab_trainer.db import Database

    settings = load_settings()
    db = Database(settings.db_full_path)

    # Load all questions ordered for stable iteration
    all_questions = db.get_all_questions_ordered()
    total = len(all_questions)
    if total == 0:
        print("No questions in database.")
        db.close()
        sys.exit(1)

    # Load log of already-regenerated questions
    log_path = Path(__file__).resolve().parent.parent / "regenerated.json"
    log_entries: list[dict] = []
    done_ids: set[str] = set()
    if log_path.exists():
        log_entries = _json.loads(log_path.read_text())
        done_ids = {e["id"] for e in log_entries}

    # Filter to pending questions
    pending = [q for q in all_questions if q["id"] not in done_ids]
    batch = pending[:batch_size]

    if not batch:
        print(f"All {total} questions already regenerated.")
        db.close()
        return

    # Set up LLM provider (same pattern as _generate)
    if settings.llm_provider == "ollama":
        from vocab_trainer.providers.llm_ollama import OllamaProvider
        llm = OllamaProvider(base_url=settings.ollama_url, model=settings.llm_model)
    elif settings.llm_provider == "anthropic":
        from vocab_trainer.providers.llm_anthropic import AnthropicProvider
        llm = AnthropicProvider()
    elif settings.llm_provider == "openai":
        from vocab_trainer.providers.llm_openai import OpenAIProvider
        llm = OpenAIProvider()
    else:
        print(f"Unknown LLM provider: {settings.llm_provider}")
        sys.exit(1)

    from vocab_trainer.question_generator import generate_question

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"Regenerating batch of {len(batch)}/{total} questions ({mode})")
    print(f"Using {settings.llm_provider}, skipping {len(done_ids)} already done")
    print()

    succeeded = 0
    failed = 0

    for i, q in enumerate(batch, 1):
        cluster_title = q.get("cluster_title", "")
        target_word = q["target_word"]
        question_type = q["question_type"]
        old_stem = q["stem"]

        # Look up cluster + word info from DB
        cluster = db.get_cluster_by_title(cluster_title) if cluster_title else None
        if cluster is None:
            print(f"  [{i}/{len(batch)}] SKIP {target_word} — cluster '{cluster_title}' not found")
            failed += 1
            continue

        cluster_words = db.get_cluster_words(cluster["id"])
        word_info = next(
            (w for w in cluster_words if w["word"].lower() == target_word.lower()),
            None,
        )
        if word_info is None:
            print(f"  [{i}/{len(batch)}] SKIP {target_word} — not found in cluster '{cluster_title}'")
            failed += 1
            continue

        # Generate new question
        new_q = asyncio.run(
            generate_question(llm, db, cluster=cluster, target_word_info=word_info, question_type=question_type)
        )

        if new_q is None:
            print(f"  [{i}/{len(batch)}] FAIL {question_type:12s} {cluster_title}: {target_word}")
            failed += 1
            continue

        # Truncate stems for display
        old_short = old_stem[:60].replace("\n", " ")
        new_short = new_q.stem[:60].replace("\n", " ")
        print(f"  [{i}/{len(batch)}] OK   {question_type:12s} {cluster_title}: {target_word}")
        print(f"         old: {old_short}")
        print(f"         new: {new_short}")

        # Update DB (unless dry run)
        if not dry_run:
            db.update_question_content(
                question_id=q["id"],
                stem=new_q.stem,
                choices=new_q.choices,
                correct_index=new_q.correct_index,
                explanation=new_q.explanation,
                context_sentence=new_q.context_sentence,
                choice_details=new_q.choice_details,
            )

        # Append to log
        log_entries.append({
            "id": q["id"],
            "target_word": target_word,
            "cluster_title": cluster_title,
            "question_type": question_type,
            "old_stem": old_stem,
            "new_stem": new_q.stem,
            "old_explanation": q.get("explanation", ""),
            "new_explanation": new_q.explanation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
        })
        succeeded += 1

    # Write log
    log_path.write_text(_json.dumps(log_entries, indent=2, ensure_ascii=False))

    print()
    done_total = len(done_ids) + succeeded
    print(f"Batch: {succeeded} succeeded, {failed} failed")
    print(f"Progress: {done_total}/{total} done")
    if dry_run:
        print("(dry run — no DB changes made)")
    db.close()


def _stats():
    from vocab_trainer.config import load_settings
    from vocab_trainer.db import Database

    settings = load_settings()
    db = Database(settings.db_full_path)
    stats = db.get_stats()

    print("Vocab Trainer Stats")
    print("=" * 40)
    print(f"Total words:        {stats['total_words']}")
    print(f"Total clusters:     {stats['total_clusters']}")
    print(f"Words reviewed:     {stats['words_reviewed']}")
    print(f"Words due:          {stats['words_due']}")
    print(f"Words new:          {stats['words_new']}")
    print(f"Question bank:      {stats['question_bank_size']}")
    print(f"Sessions completed: {stats['total_sessions']}")
    print(f"Questions answered: {stats['total_questions_answered']}")
    print(f"Overall accuracy:   {stats['accuracy']}%")
    db.close()


if __name__ == "__main__":
    main()
