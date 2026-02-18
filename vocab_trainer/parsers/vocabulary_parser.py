"""Parse vocabulary.md and vocabulary_upper_intermediate.md into VocabWord objects.

Handles two table schemas:
  | Word | Definition | Example |   (starter words — 3 columns)
  | Word | Definition |              (all other sections — 2 columns)

Section headers: ## Starter Words, ## 1. Cognition..., etc.
"""
from __future__ import annotations

import re
from pathlib import Path

from vocab_trainer.models import VocabWord


def parse_vocabulary_file(path: Path) -> list[VocabWord]:
    text = path.read_text()
    source = path.name
    words: list[VocabWord] = []
    current_section = "Unknown"

    for line in text.splitlines():
        # Match section headers like "## Starter Words" or "## 1. Cognition..."
        m = re.match(r"^## (.+)", line)
        if m:
            current_section = m.group(1).strip()
            continue

        # Skip subsection headers and non-table lines
        if not line.startswith("|"):
            continue

        # Match table rows with bold word: | **word** | definition | ...
        m = re.match(r"\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|", line)
        if m:
            word = m.group(1).strip()
            definition = m.group(2).strip()
            # Remove trailing columns (example column)
            definition = definition.rstrip("|").strip()
            words.append(VocabWord(
                word=word,
                definition=definition,
                section=current_section,
                source_file=source,
            ))

    return words
