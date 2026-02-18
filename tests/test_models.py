"""Tests for data models."""
from __future__ import annotations

from vocab_trainer.models import (
    DistinctionCluster,
    DistinctionEntry,
    Question,
    VocabWord,
)


class TestVocabWord:
    def test_create(self):
        w = VocabWord("perspicacious", "having keen insight", "Cognition", "vocabulary.md")
        assert w.word == "perspicacious"
        assert w.definition == "having keen insight"
        assert w.section == "Cognition"
        assert w.source_file == "vocabulary.md"


class TestDistinctionEntry:
    def test_create(self):
        e = DistinctionEntry("terse", "brief to the point of rudeness", "brevity as personality")
        assert e.word == "terse"
        assert e.meaning == "brief to the point of rudeness"
        assert e.distinction == "brevity as personality"


class TestDistinctionCluster:
    def test_create(self):
        entries = [
            DistinctionEntry("concise", "expressing much in few words", "compression"),
            DistinctionEntry("terse", "brief rudely", "personality trait"),
        ]
        c = DistinctionCluster(
            title="Being Brief",
            preamble="All mean using few words.",
            entries=entries,
            commentary="A concise report, a terse reply.",
        )
        assert c.title == "Being Brief"
        assert len(c.entries) == 2

    def test_empty_entries(self):
        c = DistinctionCluster("Empty", "", [], "")
        assert len(c.entries) == 0


class TestQuestion:
    def test_create(self):
        q = Question(
            id="q1",
            question_type="fill_blank",
            stem="Her ___ reply surprised everyone.",
            choices=["terse", "concise", "pithy", "laconic"],
            correct_index=0,
            correct_word="terse",
            explanation="Terse implies rudeness.",
            context_sentence="Her terse reply surprised everyone.",
            cluster_title="Being Brief",
            llm_provider="test",
        )
        assert q.correct_word == "terse"
        assert q.choices[q.correct_index] == "terse"
        assert len(q.choices) == 4
