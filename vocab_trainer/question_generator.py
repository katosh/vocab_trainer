"""Orchestrate LLM to generate quiz questions from vocabulary clusters."""
from __future__ import annotations

import json
import logging
import random
import re
import uuid

_log = logging.getLogger("vocab_trainer.qgen")
from typing import TYPE_CHECKING

from vocab_trainer.models import Question
from vocab_trainer.prompts import (
    BEST_FIT_PROMPT,
    CHOICE_ENRICHMENT_PROMPT,
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


def _pick_word_cluster(db: Database) -> tuple[dict, dict] | None:
    """Pick a (cluster, word_info) pair weighted by coverage deficit.

    Builds a categorical distribution over all (word, cluster) pairs where
    each pair's probability is proportional to 1 / (1 + question_count).
    Words that already have many questions are unlikely to be chosen;
    words with zero questions are most likely.

    Returns (cluster_dict, word_info_dict) or None if no valid pairs exist.
    """
    pairs = db.get_word_cluster_question_counts()
    if not pairs:
        return None

    weights = [1.0 / (1 + p["question_count"]) for p in pairs]
    chosen = random.choices(pairs, weights=weights, k=1)[0]

    cluster = db.get_cluster_by_title(chosen["cluster_title"])
    if cluster is None:
        return None

    word_info = {
        "word": chosen["word"],
        "meaning": chosen["meaning"],
        "distinction": chosen["distinction"],
    }
    return cluster, word_info


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


def _fix_article_before_blank(stem: str, choices: list[str]) -> str:
    """Replace 'a ___' or 'an ___' with 'a(n) ___' when choices have mixed initial letters."""
    has_vowel = any(c[0].lower() in "aeiou" for c in choices if c)
    has_consonant = any(c[0].lower() not in "aeiou" for c in choices if c)
    if has_vowel and has_consonant:
        stem = re.sub(r"\b[Aa]n ___", lambda m: m.group(0)[0] + "(n) ___", stem)
        stem = re.sub(r"\b([Aa]) ___", lambda m: m.group(1) + "(n) ___", stem)
    return stem


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
    # Stem must have a blank
    if "___" not in data["stem"] and "___" not in data.get("stem", ""):
        return False
    # Context sentence should contain the target word (or a form of it)
    ctx = data.get("context_sentence", "").lower()
    if target_word.lower() not in ctx:
        return False
    # Auto-fix article before blank
    data["stem"] = _fix_article_before_blank(data["stem"], data["choices"])
    return True


ENRICHMENT_RETRIES = 2


async def _enrich_choices(
    llm: LLMProvider,
    cluster: dict,
    cluster_words: list[dict],
    data: dict,
) -> list[dict]:
    """Step 2: enrich choices via a second LLM call.

    Returns a list of dicts with word, base_word, meaning, distinction, why.
    Falls back to stem-based lookup on failure.
    """
    prompt = CHOICE_ENRICHMENT_PROMPT.format(
        cluster_title=cluster["title"],
        cluster_info=format_cluster_info(cluster_words),
        stem=data["stem"],
        choices_formatted=", ".join(data["choices"]),
        correct_word=data["choices"][data["correct_index"]],
        correct_index=data["correct_index"],
    )

    for attempt in range(ENRICHMENT_RETRIES):
        try:
            _log.info("  Enrich choices (attempt %d/%d)", attempt + 1, ENRICHMENT_RETRIES)
            response = await llm.generate(prompt, temperature=0.3)
            parsed = _extract_json(response)
            if parsed is None:
                _log.info("  Enrich: no valid JSON in response")
                continue
            details = parsed.get("choice_details", [])
            if not isinstance(details, list) or len(details) != len(data["choices"]):
                _log.info("  Enrich: wrong number of details (%s)", len(details) if isinstance(details, list) else "n/a")
                continue
            # Validate each detail has required fields
            required = {"word", "base_word", "meaning", "distinction", "why"}
            if all(required.issubset(d.keys()) for d in details):
                _log.info("  Enrich: OK")
                return details
            _log.info("  Enrich: missing fields in details")
        except Exception as e:
            _log.info("  Enrich: error — %s", e)

    _log.info("  Enrich: falling back to DB lookup")
    # Fallback: stem-based lookup (no why field from LLM)
    from vocab_trainer.db import _lookup_cluster_word

    cw_map = {w["word"].lower(): w for w in cluster_words}
    fallback = []
    for c in data["choices"]:
        info = _lookup_cluster_word(cw_map, c) or {}
        fallback.append({
            "word": c,
            "base_word": info.get("word", c) if info else c,
            "meaning": info.get("meaning", ""),
            "distinction": info.get("distinction", ""),
            "why": "",
        })
    return fallback


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
    # Pick cluster + word using coverage-weighted selection if not provided
    if cluster is None or target_word_info is None:
        picked = _pick_word_cluster(db)
        if picked is None:
            return None
        cluster, target_word_info = picked

    cluster_words = db.get_cluster_words(cluster["id"])
    if len(cluster_words) < 4:
        return None

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
    target = target_word_info["word"]
    for attempt in range(MAX_RETRIES):
        try:
            _log.info("Generate %s for '%s' (attempt %d/%d)",
                       question_type, target, attempt + 1, MAX_RETRIES)
            response = await llm.generate(prompt, temperature=0.7)
            data = _extract_json(response)
            if data is None:
                _log.info("  Step 1 failed: no valid JSON in response")
                continue
            if not _validate_question(data, target):
                _log.info("  Step 1 failed: validation rejected question")
                continue

            _log.info("  Step 1 OK — question generated")

            # Step 2: enrich choices via second LLM call (falls back to
            # stem-based lookup if the enrichment call fails)
            choice_details = await _enrich_choices(
                llm, cluster, cluster_words, data,
            )

            _log.info("  Done: %s '%s'", question_type, target)
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
                choice_details=choice_details,
            )
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                _log.warning("Failed after %d attempts: %s", MAX_RETRIES, e)
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
                    else:
                        _log.warning("Batch [%d/%d]: failed to generate for '%s'", len(questions), count, word)
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
            else:
                _log.warning("Batch [%d/%d]: generation attempt %d returned nothing", len(questions), count, i + 1)

    return questions
