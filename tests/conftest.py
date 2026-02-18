"""Shared test fixtures."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vocab_trainer.db import Database
from vocab_trainer.models import (
    DistinctionCluster,
    DistinctionEntry,
    Question,
    VocabWord,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a fresh temporary database."""
    db = Database(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture
def sample_words():
    """A small set of VocabWord objects for testing."""
    return [
        VocabWord("perspicacious", "having keen insight", "Cognition", "vocabulary.md"),
        VocabWord("sagacious", "having practical wisdom", "Cognition", "vocabulary.md"),
        VocabWord("astute", "shrewd; quick to assess", "Cognition", "vocabulary.md"),
        VocabWord("ebullient", "full of enthusiasm", "Energy", "vocabulary.md"),
        VocabWord("sanguine", "optimistic", "Energy", "vocabulary.md"),
    ]


@pytest.fixture
def sample_cluster():
    """A DistinctionCluster with enough entries for question generation."""
    return DistinctionCluster(
        title="Being Brief",
        preamble="All mean using few words, but the quality of that brevity differs.",
        entries=[
            DistinctionEntry("concise", "expressing much in few words", "implies skillful compression"),
            DistinctionEntry("succinct", "briefly and clearly expressed", "emphasizes clarity"),
            DistinctionEntry("terse", "brief to the point of seeming rude", "brevity as personality trait"),
            DistinctionEntry("laconic", "using very few words habitually", "temperamental preference for silence"),
            DistinctionEntry("pithy", "concise AND forcefully meaningful", "the best kind of brief"),
            DistinctionEntry("curt", "rudely brief", "the worst kind of brief"),
        ],
        commentary="A concise report, a succinct summary, a terse reply.",
    )


@pytest.fixture
def sample_question():
    """A valid Question object."""
    return Question(
        id="test-q-001",
        question_type="fill_blank",
        stem="Her ___ reply left no room for pleasantries.",
        choices=["terse", "concise", "pithy", "laconic"],
        correct_index=0,
        correct_word="terse",
        explanation="Terse implies brevity that borders on rudeness.",
        context_sentence="Her terse reply left no room for pleasantries.",
        cluster_title="Being Brief",
        llm_provider="test",
    )


@pytest.fixture
def populated_db(tmp_db, sample_words, sample_cluster):
    """A database pre-loaded with sample data."""
    tmp_db.import_words(sample_words)
    tmp_db.import_clusters([sample_cluster])
    return tmp_db


@pytest.fixture
def vocab_md_content():
    """Minimal vocabulary.md content for parser testing."""
    return """\
# Vocabulary Building

---

## Starter Words

| Word | Definition | Example |
|------|------------|---------|
| **unpalatable** | difficult to accept | *The truth was unpalatable.* |
| **sprightly** | lively, energetic | *She remained sprightly.* |

---

## 1. Cognition, Attention, and Insight

*How clearly someone perceives.*

| Word | Definition |
|------|------------|
| **perspicacious** | having keen insight; perceptive |
| **sagacious** | having keen practical wisdom |
| **astute** | shrewd; able to assess situations quickly |
"""


@pytest.fixture
def distinctions_md_content():
    """Minimal vocabulary_distinctions.md content for parser testing."""
    return """\
# Vocabulary Distinctions

---

## Being Brief

All mean "using few words," but the *quality* of that brevity differs.

| Word | What it really means | Key distinction |
|------|---------------------|-----------------|
| **concise** | expressing much in few words | implies skillful *compression* |
| **succinct** | briefly and clearly expressed | emphasizes *clarity* |
| **terse** | brief to the point of seeming rude | brevity as a *personality trait* |
| **laconic** | using very few words habitually | *temperamental* preference for silence |

> A *concise* report, a *succinct* summary, a *terse* reply, a *laconic* cowboy.

---

## Being Wordy

All mean "using too many words," but the *nature* of the excess differs.

| Word | What it really means | Key distinction |
|------|---------------------|-----------------|
| **verbose** | using more words than needed | the general term |
| **prolix** | tediously lengthy | implies *boring* as well as long |
| **diffuse** | spread out, lacking focus | a structural problem |
| **loquacious** | naturally talkative | about *personality*, not quality |
| **garrulous** | talking too much about trivial things | implies *age* and *triviality* |

> *Loquacious* people talk a lot; *garrulous* people talk a lot about nothing.
"""
