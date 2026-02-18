"""Tests for vocabulary and distinctions parsers."""
from __future__ import annotations

from pathlib import Path

import pytest

from vocab_trainer.parsers.distinctions_parser import parse_distinctions_file
from vocab_trainer.parsers.vocabulary_parser import parse_vocabulary_file


class TestVocabularyParser:
    def test_parse_basic(self, tmp_path, vocab_md_content):
        f = tmp_path / "vocabulary.md"
        f.write_text(vocab_md_content)
        words = parse_vocabulary_file(f)

        assert len(words) == 5
        assert words[0].word == "unpalatable"
        assert words[0].source_file == "vocabulary.md"

    def test_sections_detected(self, tmp_path, vocab_md_content):
        f = tmp_path / "vocabulary.md"
        f.write_text(vocab_md_content)
        words = parse_vocabulary_file(f)

        starter = [w for w in words if w.section == "Starter Words"]
        cognition = [w for w in words if "Cognition" in w.section]
        assert len(starter) == 2
        assert len(cognition) == 3

    def test_definition_extracted(self, tmp_path, vocab_md_content):
        f = tmp_path / "vocabulary.md"
        f.write_text(vocab_md_content)
        words = parse_vocabulary_file(f)

        persp = next(w for w in words if w.word == "perspicacious")
        assert "keen insight" in persp.definition

    def test_three_column_table(self, tmp_path, vocab_md_content):
        """Starter Words table has 3 columns — definition should not include example."""
        f = tmp_path / "vocabulary.md"
        f.write_text(vocab_md_content)
        words = parse_vocabulary_file(f)

        unpal = next(w for w in words if w.word == "unpalatable")
        assert "difficult to accept" in unpal.definition
        # Example column content should not leak into definition
        assert "The truth" not in unpal.definition

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("# Empty\n\nNo tables here.\n")
        words = parse_vocabulary_file(f)
        assert words == []

    def test_real_vocabulary_file(self):
        """Parse the actual vocabulary.md and verify reasonable counts."""
        path = Path(__file__).resolve().parent.parent.parent / "vocabulary.md"
        if not path.exists():
            pytest.skip("vocabulary.md not found")
        words = parse_vocabulary_file(path)
        assert len(words) > 1000
        # All words should have non-empty fields
        for w in words:
            assert w.word
            assert w.definition
            assert w.source_file


class TestDistinctionsParser:
    def test_parse_basic(self, tmp_path, distinctions_md_content):
        f = tmp_path / "distinctions.md"
        f.write_text(distinctions_md_content)
        clusters = parse_distinctions_file(f)

        assert len(clusters) == 2
        assert clusters[0].title == "Being Brief"
        assert clusters[1].title == "Being Wordy"

    def test_entries_parsed(self, tmp_path, distinctions_md_content):
        f = tmp_path / "distinctions.md"
        f.write_text(distinctions_md_content)
        clusters = parse_distinctions_file(f)

        brief = clusters[0]
        assert len(brief.entries) == 4
        assert brief.entries[0].word == "concise"
        assert "few words" in brief.entries[0].meaning
        assert "compression" in brief.entries[0].distinction

    def test_preamble_parsed(self, tmp_path, distinctions_md_content):
        f = tmp_path / "distinctions.md"
        f.write_text(distinctions_md_content)
        clusters = parse_distinctions_file(f)

        assert "few words" in clusters[0].preamble

    def test_commentary_parsed(self, tmp_path, distinctions_md_content):
        f = tmp_path / "distinctions.md"
        f.write_text(distinctions_md_content)
        clusters = parse_distinctions_file(f)

        assert "concise" in clusters[0].commentary
        assert "terse" in clusters[0].commentary

    def test_wordy_cluster(self, tmp_path, distinctions_md_content):
        f = tmp_path / "distinctions.md"
        f.write_text(distinctions_md_content)
        clusters = parse_distinctions_file(f)

        wordy = clusters[1]
        assert len(wordy.entries) == 5
        words = [e.word for e in wordy.entries]
        assert "verbose" in words
        assert "garrulous" in words

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("# Nothing\n\nNo clusters.\n")
        clusters = parse_distinctions_file(f)
        assert clusters == []

    def test_real_distinctions_file(self):
        """Parse the actual distinctions file and verify reasonable counts."""
        path = Path(__file__).resolve().parent.parent.parent / "vocabulary_distinctions.md"
        if not path.exists():
            pytest.skip("vocabulary_distinctions.md not found")
        clusters = parse_distinctions_file(path)
        assert len(clusters) >= 50
        total_entries = sum(len(c.entries) for c in clusters)
        assert total_entries > 500
        # Every cluster should have at least 2 entries
        for c in clusters:
            assert len(c.entries) >= 2, f"Cluster '{c.title}' has only {len(c.entries)} entries"
            assert c.title
