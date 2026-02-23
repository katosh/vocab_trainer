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
    """Extract JSON object from LLM response, handling markdown code fences.

    Strips ``<think>`` blocks first (Qwen3 reasoning can contain JSON-like
    fragments that confuse the greedy regex).  Tries code-fenced JSON first,
    then falls back to matching balanced ``{…}`` blocks — preferring the
    *last* match, since the LLM often drafts partial JSON before the final
    answer.
    """
    # Strip <think>…</think> blocks that may contain draft JSON
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1. Try JSON inside a code fence (most reliable)
    m = re.search(r"```(?:json)?\s*\n?({.*?})\s*\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Find all top-level {…} blocks and try each, last-first
    candidates = _find_json_objects(text)
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _find_json_objects(text: str) -> list[str]:
    """Find balanced top-level ``{…}`` substrings in *text*."""
    results: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            in_str = False
            escape = False
            start = i
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        results.append(text[start : j + 1])
                        i = j + 1
                        break
            else:
                # Unbalanced — skip this opening brace
                i += 1
        else:
            i += 1
    return results


def _fix_article_before_blank(stem: str, choices: list[str]) -> str:
    """Replace 'a ___' or 'an ___' with 'a(n) ___' when choices have mixed initial letters."""
    has_vowel = any(c[0].lower() in "aeiou" for c in choices if c)
    has_consonant = any(c[0].lower() not in "aeiou" for c in choices if c)
    if has_vowel and has_consonant:
        stem = re.sub(r"\b[Aa]n ___", lambda m: m.group(0)[0] + "(n) ___", stem)
        stem = re.sub(r"\b([Aa]) ___", lambda m: m.group(1) + "(n) ___", stem)
    return stem


def _is_inflection(candidate: str, base: str) -> bool:
    """Check if *candidate* is a plausible English inflection of *base*.

    Covers regular suffixes: -s, -es, -ed, -d, -ing, -ly, -er, -est, -tion,
    and the e-dropping pattern (e.g. "explicate" → "explicating").
    """
    if candidate == base:
        return True
    # candidate must start with the base (or base minus trailing 'e')
    if candidate.startswith(base):
        suffix = candidate[len(base):]
        return suffix in ("s", "es", "ed", "d", "ing", "ly", "er", "est",
                          "tion", "ment", "ness", "ous", "ive", "al")
    # e-dropping: "explicate" → "explicating", "cajole" → "cajoled"
    if base.endswith("e") and candidate.startswith(base[:-1]):
        suffix = candidate[len(base) - 1:]
        return suffix in ("ing", "ed", "er", "est", "ation", "ive",
                          "ous", "able", "ible")
    # y→i: "carry" → "carried"
    if base.endswith("y") and candidate.startswith(base[:-1] + "i"):
        suffix = candidate[len(base):]
        return suffix in ("ed", "es", "er", "est", "ness", "ly")
    return False


def _validate_question(data: dict, target_word: str, question_type: str = "fill_blank") -> str | None:
    """Validate a generated question and auto-fix minor issues.

    Returns ``None`` on success (data is valid and possibly patched in-place),
    or a human-readable reason string on failure.
    """
    required = {"stem", "choices", "correct_index", "explanation", "context_sentence"}
    missing = required - data.keys()
    if missing:
        return f"missing fields: {', '.join(sorted(missing))}"

    if not isinstance(data["choices"], list) or len(data["choices"]) != 4:
        n = len(data["choices"]) if isinstance(data["choices"], list) else type(data["choices"]).__name__
        return f"choices must be list of 4 (got {n})"

    # Coerce correct_index from string to int (common LLM mistake)
    ci = data["correct_index"]
    if isinstance(ci, str) and ci.strip().isdigit():
        data["correct_index"] = int(ci.strip())
        ci = data["correct_index"]
    if not isinstance(ci, int):
        return f"correct_index not an int (got {type(ci).__name__}: {ci!r})"
    if ci < 0 or ci >= 4:
        return f"correct_index out of range: {ci}"

    # Check the correct choice matches the target word (accept inflected forms)
    correct_choice = data["choices"][ci].lower().strip()
    target_lower = target_word.lower().strip()
    if correct_choice != target_lower:
        # Try exact match first, then inflection match
        for i, ch in enumerate(data["choices"]):
            if ch.lower().strip() == target_lower:
                data["correct_index"] = i
                ci = i
                break
        else:
            # Accept inflected forms (e.g. "cajoled" for "cajole")
            for i, ch in enumerate(data["choices"]):
                if _is_inflection(ch.lower().strip(), target_lower):
                    data["correct_index"] = i
                    ci = i
                    # Store the base word as correct_word (caller uses target)
                    break
            else:
                choices_str = ", ".join(data["choices"])
                return (
                    f"The target word '{target_word}' MUST be one of the 4 choices "
                    f"and correct_index must point to it. "
                    f"Your choices were: [{choices_str}] — '{target_word}' is missing."
                )

    # No duplicate choices
    lower_choices = [c.lower().strip() for c in data["choices"]]
    if len(set(lower_choices)) != 4:
        dupes = [c for c in lower_choices if lower_choices.count(c) > 1]
        return f"duplicate choices: {set(dupes)}"

    # Stem must have a blank (fill_blank only — best_fit uses "Which word…?" format)
    if question_type == "fill_blank":
        stem = data["stem"]
        # Normalize various blank markers to ___
        if "___" not in stem:
            normalized = re.sub(r"_{4,}", "___", stem)   # ____ → ___
            normalized = re.sub(r"\[blank\]", "___", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"\(blank\)", "___", normalized, flags=re.IGNORECASE)
            if "___" in normalized:
                data["stem"] = normalized
            else:
                return f"stem has no blank marker: {stem[:80]!r}"

    # Context sentence should contain the target word (or a morphological form)
    ctx = data.get("context_sentence", "").lower()
    if target_lower not in ctx:
        # Accept common morphological suffixes
        stem_root = target_lower.rstrip("edsying")
        if len(stem_root) < 3 or not any(stem_root in w for w in ctx.split()):
            return f"context_sentence missing target word '{target_word}'"

    # Auto-fix article before blank (only relevant for fill_blank)
    if question_type == "fill_blank":
        data["stem"] = _fix_article_before_blank(data["stem"], data["choices"])
    return None


ENRICHMENT_RETRIES = 3


def _validate_enrichment(details: list, expected_count: int) -> str | None:
    """Return None if valid, or an error message describing what's wrong."""
    if not isinstance(details, list):
        return f"choice_details must be a list, got {type(details).__name__}"
    if len(details) != expected_count:
        return f"expected {expected_count} choice_details, got {len(details)}"
    required = {"word", "base_word", "meaning", "distinction", "why"}
    problems = []
    for i, d in enumerate(details):
        if not isinstance(d, dict):
            problems.append(f"  detail[{i}]: expected object, got {type(d).__name__}")
            continue
        missing = required - d.keys()
        if missing:
            problems.append(f"  detail[{i}] ({d.get('word', '?')}): missing {', '.join(sorted(missing))}")
    if problems:
        return "field errors:\n" + "\n".join(problems)
    return None


async def _enrich_choices(
    llm: LLMProvider,
    cluster: dict,
    cluster_words: list[dict],
    data: dict,
) -> list[dict]:
    """Step 2: enrich choices via a second LLM call.

    Returns a list of dicts with word, base_word, meaning, distinction, why.
    On validation errors, feeds the specific error back to the LLM so it can
    correct its response.  Falls back to stem-based lookup on total failure.
    """
    base_prompt = CHOICE_ENRICHMENT_PROMPT.format(
        cluster_title=cluster["title"],
        cluster_info=format_cluster_info(cluster_words),
        stem=data["stem"],
        choices_formatted=", ".join(data["choices"]),
        correct_word=data["choices"][data["correct_index"]],
        correct_index=data["correct_index"],
    )

    prompt = base_prompt
    for attempt in range(ENRICHMENT_RETRIES):
        try:
            _log.info("  Enrich choices (attempt %d/%d)", attempt + 1, ENRICHMENT_RETRIES)
            response = await llm.generate(prompt, temperature=0.3)
            parsed = _extract_json(response)
            if parsed is None:
                feedback = "Your response did not contain valid JSON. Respond with ONLY a JSON object, no other text."
                prompt = base_prompt + "\n\n" + feedback
                _log.info("  Enrich: no valid JSON — feeding back")
                continue

            details = parsed.get("choice_details", [])
            error = _validate_enrichment(details, len(data["choices"]))
            if error:
                prompt = base_prompt + f"\n\nYour previous response had errors:\n{error}\n\nPlease fix and respond with corrected JSON only."
                _log.info("  Enrich: validation failed — feeding back: %s", error.split('\n')[0])
                continue

            _log.info("  Enrich: OK")
            return details
        except Exception as e:
            _log.info("  Enrich: error — %s", e)

    _log.info("  Enrich: falling back to DB lookup")
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
    enrichment = db.get_random_words(limit=random.randint(5, 10))

    # Format prompt
    prompt_template = PROMPTS[question_type]
    base_prompt = prompt_template.format(
        cluster_title=cluster["title"],
        cluster_info=format_cluster_info(cluster_words),
        target_word=target_word_info["word"],
        target_meaning=target_word_info["meaning"],
        target_distinction=target_word_info["distinction"],
        enrichment_section=format_enrichment(enrichment),
    )

    # Try generation with retries — feed validation errors back to the LLM
    target = target_word_info["word"]
    prompt = base_prompt
    for attempt in range(MAX_RETRIES):
        try:
            _log.info("Generate %s (attempt %d/%d)",
                       question_type, attempt + 1, MAX_RETRIES)
            response = await llm.generate(prompt, temperature=0.7)
            data = _extract_json(response)
            if data is None:
                feedback = "Your response did not contain valid JSON. Respond with ONLY a JSON object, no other text."
                prompt = base_prompt + "\n\n" + feedback
                _log.info("  Step 1 failed: no valid JSON — feeding back")
                _log.debug("  Raw response: %.300s", response)
                continue
            reason = _validate_question(data, target, question_type)
            if reason:
                prompt = base_prompt + f"\n\nYour previous response had errors: {reason}\nPlease fix and respond with corrected JSON only."
                _log.info("  Step 1 failed: %s — feeding back", reason)
                continue

            _log.info("  Step 1 OK — question generated")

            # Step 2: enrich choices via second LLM call (falls back to
            # stem-based lookup if the enrichment call fails)
            choice_details = await _enrich_choices(
                llm, cluster, cluster_words, data,
            )

            _log.info("  Saved (%s)", cluster["title"])
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
    target_pairs: list[tuple[str, str]] | None = None,
) -> list[Question]:
    """Generate a batch of questions and save to database.

    target_pairs: list of (word, cluster_title) to generate for specific
    word-cluster combinations (e.g. refilling after a question is answered).
    """
    questions: list[Question] = []

    # Generate for specific (word, cluster) pairs first
    if target_pairs:
        total_pairs = len(target_pairs)
        for pair_idx, (word, cluster_title) in enumerate(target_pairs, 1):
            cluster = db.get_cluster_by_title(cluster_title)
            if not cluster:
                continue
            cw = db.get_cluster_words(cluster["id"])
            word_info = next((w for w in cw if w["word"].lower() == word.lower()), None)
            if not word_info:
                continue
            _log.info("[%d/%d] Generating for '%s'", pair_idx, total_pairs, cluster_title)
            q = await generate_question(llm, db, cluster=cluster, target_word_info=word_info)
            if q:
                db.save_question(q)
                questions.append(q)
            else:
                _log.warning("[%d/%d] Failed for '%s' in '%s'", pair_idx, total_pairs, word, cluster_title)

    if target_words:
        # Generate questions for specific words (legacy)
        for word in target_words:
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
    elif not target_pairs:
        # Generate random questions (only if no targeted generation)
        for i in range(count):
            q = await generate_question(llm, db)
            if q:
                db.save_question(q)
                questions.append(q)
            else:
                _log.warning("Batch [%d/%d]: generation attempt %d returned nothing", len(questions), count, i + 1)

    return questions
