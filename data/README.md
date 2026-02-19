# Vocabulary Data

These baseline vocabulary files were collected by Dominik ([@katosh](https://github.com/katosh)) for personal use while learning English.

They are bundled as starter content — feel free to edit, replace, or remove them. The tool can import from any markdown files that follow the expected format.

## Files

- **vocabulary.md** — word/definition tables
- **vocabulary_upper_intermediate.md** — upper-intermediate level words
- **vocabulary_distinctions.md** — groups of easily confused words with meanings and distinctions

## How updates work

On server startup, the trainer checks each file's modification time against what was last imported. If a file has changed, it is automatically re-imported into the SQLite database. This means you can edit any file here and the changes take effect on the next restart (or `uv run python -m vocab_trainer import`).

To skip the startup check, use `--no-auto-import`:

```bash
uv run python -m vocab_trainer serve --no-auto-import
```

To use your own files instead, set `vocab_files` in `config.json` to a list of paths (relative to the project root). When `vocab_files` is empty (the default), all `*.md` files in this directory are imported.
