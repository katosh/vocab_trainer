from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VocabWord:
    word: str
    definition: str
    section: str
    source_file: str


@dataclass
class DistinctionEntry:
    word: str
    meaning: str
    distinction: str


@dataclass
class DistinctionCluster:
    title: str
    preamble: str
    entries: list[DistinctionEntry]
    commentary: str
    source_file: str = ""


@dataclass
class Question:
    id: str
    question_type: str  # fill_blank | best_fit | distinction
    stem: str
    choices: list[str]
    correct_index: int
    correct_word: str
    explanation: str
    context_sentence: str  # full sentence for TTS
    cluster_title: str | None
    llm_provider: str
    choice_explanations: list[str] = field(default_factory=list)
