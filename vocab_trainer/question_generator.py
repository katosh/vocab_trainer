"""Orchestrate LLM to generate quiz questions from vocabulary clusters."""
from __future__ import annotations

import json
import random
import re
import uuid
from typing import TYPE_CHECKING

from vocab_trainer.models import Question
from vocab_trainer.prompts import (
    BEST_FIT_PROMPT,
    DISTINCTION_PROMPT,
    FILL_BLANK_PROMPT,
    format_cluster_info,
    format_enrichment,
)

if TYPE_CHECKING:
    from vocab_trainer.db import Database
    from vocab_trainer.providers.base import LLMProvider

MAX_RETRIES = 3

# Question type weights
TYPE_WEIGHTS = {
    "fill_blank": 0.60,
    "best_fit": 0.25,
    "distinction": 0.15,
}

PROMPTS = {
    "fill_blank": FILL_BLANK_PROMPT,
    "best_fit": BEST_FIT_PROMPT,
    "distinction": DISTINCTION_PROMPT,
}


def _pick_question_type() -> str:
    r = random.random()
    cumulative = 0.0
    for qtype, weight in TYPE_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return qtype
    return "fill_blank"


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response, handling markdown code fences."""
    # Try to find JSON in code fence
    m = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _validate_question(data: dict, target_word: str) -> bool:
    """Validate that the generated question meets requirements."""
    required = {"stem", "choices", "correct_index", "explanation", "context_sentence"}
    if not required.issubset(data.keys()):
        return False
    if not isinstance(data["choices"], list) or len(data["choices"]) != 4:
        return False
    if not isinstance(data["correct_index"], int):
        return False
    if data["correct_index"] < 0 or data["correct_index"] >= 4:
        return False
    # Check the correct choice matches the target word
    correct_choice = data["choices"][data["correct_index"]].lower().strip()
    if correct_choice != target_word.lower().strip():
        return False
    # No duplicate choices
    lower_choices = [c.lower().strip() for c in data["choices"]]
    if len(set(lower_choices)) != 4:
        return False
    return True


async def generate_question(
    llm: LLMProvider,
    db: Database,
    cluster: dict | None = None,
    target_word_info: dict | None = None,
    question_type: str | None = None,
) -> Question | None:
    """Generate a single question using the LLM.

    If cluster/target_word_info not provided, picks a random cluster and word.
    """
    # Pick a random cluster if not provided
    if cluster is None:
        cluster = db.get_random_cluster()
        if cluster is None:
            return None

    cluster_words = db.get_cluster_words(cluster["id"])
    if len(cluster_words) < 4:
        return None

    # Pick target word
    if target_word_info is None:
        target_word_info = random.choice(cluster_words)

    # Pick question type
    if question_type is None:
        question_type = _pick_question_type()

    # Get enrichment words
    enrichment = db.get_new_words(limit=random.randint(5, 10))

    # Format prompt
    prompt_template = PROMPTS[question_type]
    prompt = prompt_template.format(
        cluster_title=cluster["title"],
        cluster_info=format_cluster_info(cluster_words),
        target_word=target_word_info["word"],
        target_meaning=target_word_info["meaning"],
        target_distinction=target_word_info["distinction"],
        enrichment_section=format_enrichment(enrichment),
    )

    # Try generation with retries
    for attempt in range(MAX_RETRIES):
        try:
            response = await llm.generate(prompt, temperature=0.7)
            data = _extract_json(response)
            if data is None:
                continue
            if not _validate_question(data, target_word_info["word"]):
                continue

            return Question(
                id=str(uuid.uuid4()),
                question_type=question_type,
                stem=data["stem"],
                choices=data["choices"],
                correct_index=data["correct_index"],
                correct_word=target_word_info["word"],
                explanation=data["explanation"],
                context_sentence=data["context_sentence"],
                cluster_title=cluster["title"],
                llm_provider=llm.name(),
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}")
    return None


async def generate_batch(
    llm: LLMProvider,
    db: Database,
    count: int = 10,
    target_words: list[str] | None = None,
) -> list[Question]:
    """Generate a batch of questions and save to database."""
    questions: list[Question] = []

    if target_words:
        # Generate questions for specific words
        for word in target_words:
            # Find the cluster containing this word
            all_clusters = db.get_all_clusters()
            for cl in all_clusters:
                cw = db.get_cluster_words(cl["id"])
                word_info = next((w for w in cw if w["word"].lower() == word.lower()), None)
                if word_info:
                    q = await generate_question(llm, db, cluster=cl, target_word_info=word_info)
                    if q:
                        db.save_question(q)
                        questions.append(q)
                        print(f"  [{len(questions)}/{count}] {q.question_type}: {q.correct_word}")
                    break
            if len(questions) >= count:
                break
    else:
        # Generate random questions
        for i in range(count):
            q = await generate_question(llm, db)
            if q:
                db.save_question(q)
                questions.append(q)
                print(f"  [{len(questions)}/{count}] {q.question_type}: {q.correct_word}")

    return questions
