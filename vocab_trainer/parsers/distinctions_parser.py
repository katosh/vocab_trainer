"""Parse vocabulary_distinctions.md into DistinctionCluster objects.

All clusters use a 3-column table:
  | Word | What it really means | Key distinction |

But column headers vary slightly across sections, so we detect the header
row dynamically and always read: bold-word, col2, col3.
"""
from __future__ import annotations

import re
from pathlib import Path

from vocab_trainer.models import DistinctionCluster, DistinctionEntry


def parse_distinctions_file(path: Path) -> list[DistinctionCluster]:
    text = path.read_text()
    source = path.name
    clusters: list[DistinctionCluster] = []

    # Split on horizontal rules to get per-cluster blocks
    blocks = re.split(r"\n---\n", text)

    for block in blocks:
        cluster = _parse_block(block.strip())
        if cluster and cluster.entries:
            cluster.source_file = source
            clusters.append(cluster)

    return clusters


def _parse_block(block: str) -> DistinctionCluster | None:
    lines = block.splitlines()
    title = ""
    preamble = ""
    entries: list[DistinctionEntry] = []
    commentary_lines: list[str] = []
    in_table = False
    header_seen = False

    for line in lines:
        stripped = line.strip()

        # Cluster title
        if stripped.startswith("## "):
            title = stripped.lstrip("#").strip()
            continue

        # Top-level header (# Rhetoric, ...) — skip
        if stripped.startswith("# "):
            continue

        # Preamble: italic line before the table
        if not in_table and not header_seen and stripped.startswith("*") and not stripped.startswith("**"):
            preamble = stripped.strip("*").strip()
            continue

        # Also capture "All mean..." style preambles (non-italic)
        if not in_table and not header_seen and not stripped.startswith("|") and stripped and not stripped.startswith(">"):
            if not title:
                continue
            preamble = stripped
            continue

        # Table header row (contains Word and separator-like dashes)
        if stripped.startswith("|") and not header_seen:
            # Check if this is the header
            if "Word" in stripped or "word" in stripped:
                header_seen = True
                in_table = True
                continue
            # Separator row
            if re.match(r"\|[-|\s]+\|", stripped):
                continue

        # Separator row
        if stripped.startswith("|") and re.match(r"\|[-|\s]+\|", stripped):
            in_table = True
            continue

        # Table data row
        if stripped.startswith("|") and in_table:
            m = re.match(r"\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", stripped)
            if m:
                word = m.group(1).strip()
                meaning = m.group(2).strip()
                distinction = m.group(3).strip()
                entries.append(DistinctionEntry(
                    word=word,
                    meaning=meaning,
                    distinction=distinction,
                ))
            continue

        # Blockquote commentary (after table)
        if stripped.startswith(">"):
            in_table = False
            commentary_lines.append(stripped.lstrip(">").strip())
            continue

    if not title:
        return None

    return DistinctionCluster(
        title=title,
        preamble=preamble,
        entries=entries,
        commentary="\n".join(commentary_lines),
    )
