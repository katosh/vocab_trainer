"""One-off script: send each question to the LLM to fix choice conjugation.

Usage:
    uv run --extra all python scripts/fix_conjugation.py [--dry-run]
"""
from __future__ import annotations

import asyncio
import json
import sys

from vocab_trainer.config import get_settings
from vocab_trainer.db import Database
from vocab_trainer.question_generator import _extract_json

FIX_PROMPT = """\
Below is a vocabulary quiz question. Your ONLY job is to ensure that ALL four \
choices are conjugated/inflected identically so they fit the blank's grammatical \
slot. The student should choose based on MEANING, never grammar.

Rules:
- Look at the stem and determine what form the blank requires (e.g. adjective, \
past-tense verb, noun, adverb).
- If all four choices already match that form, return them unchanged.
- If any choice has a different inflection, fix it so all four match the blank.
- Update the context_sentence if you changed the correct answer's form.
- Do NOT change the stem, explanation, or correct_index.
- Do NOT change which word is correct — only fix inflection.

Current question:
{{
  "stem": {stem},
  "choices": {choices},
  "correct_index": {correct_index},
  "explanation": {explanation},
  "context_sentence": {context_sentence},
  "choice_explanations": {choice_explanations}
}}

Respond with the corrected JSON only (same keys), no other text.\
"""


def _get_llm():
    s = get_settings()
    if s.llm_provider == "ollama":
        from vocab_trainer.providers.llm_ollama import OllamaProvider
        return OllamaProvider(base_url=s.ollama_url, model=s.llm_model)
    elif s.llm_provider == "anthropic":
        from vocab_trainer.providers.llm_anthropic import AnthropicProvider
        return AnthropicProvider()
    elif s.llm_provider == "openai":
        from vocab_trainer.providers.llm_openai import OpenAIProvider
        return OpenAIProvider()
    raise ValueError(f"Unknown LLM provider: {s.llm_provider}")


async def fix_question(llm, q: dict) -> dict | None:
    """Ask LLM to fix conjugation. Returns corrected fields or None."""
    choices = json.loads(q["choices_json"])
    choice_explanations = json.loads(q.get("choice_explanations_json", "[]"))

    prompt = FIX_PROMPT.format(
        stem=json.dumps(q["stem"]),
        choices=json.dumps(choices),
        correct_index=q["correct_index"],
        explanation=json.dumps(q["explanation"]),
        context_sentence=json.dumps(q["context_sentence"]),
        choice_explanations=json.dumps(choice_explanations),
    )

    response = await llm.generate(prompt, temperature=0.0)
    data = _extract_json(response)
    if data is None:
        return None

    # Sanity checks
    if not isinstance(data.get("choices"), list) or len(data["choices"]) != 4:
        return None
    if data.get("correct_index") != q["correct_index"]:
        return None  # LLM should not change the answer

    return data


async def main():
    dry_run = "--dry-run" in sys.argv

    db = Database()
    llm = _get_llm()
    print(f"LLM provider: {llm.name()}")

    # Fetch ALL questions (active + archived)
    rows = db.conn.execute(
        "SELECT * FROM questions ORDER BY target_word"
    ).fetchall()
    questions = [dict(r) for r in rows]
    print(f"Found {len(questions)} questions total\n")

    changed = 0
    failed = 0
    unchanged = 0

    for i, q in enumerate(questions, 1):
        old_choices = json.loads(q["choices_json"])
        label = f"[{i}/{len(questions)}] {q['target_word']}: "

        data = await fix_question(llm, q)
        if data is None:
            print(f"{label}FAILED (bad LLM response)")
            failed += 1
            continue

        new_choices = data["choices"]
        new_ctx = data.get("context_sentence", q["context_sentence"])
        new_expl = data.get("explanation", q["explanation"])
        new_choice_expl = data.get("choice_explanations", json.loads(q.get("choice_explanations_json", "[]")))

        # Check if anything changed
        if (new_choices == old_choices
                and new_ctx == q["context_sentence"]
                and new_expl == q["explanation"]):
            print(f"{label}OK (no changes needed)")
            unchanged += 1
            continue

        print(f"{label}FIXED")
        if new_choices != old_choices:
            print(f"  choices: {old_choices} → {new_choices}")
        if new_ctx != q["context_sentence"]:
            print(f"  context: {q['context_sentence']!r} → {new_ctx!r}")

        if not dry_run:
            db.conn.execute(
                "UPDATE questions SET choices_json = ?, context_sentence = ?, "
                "explanation = ?, choice_explanations_json = ? WHERE id = ?",
                (
                    json.dumps(new_choices),
                    new_ctx,
                    new_expl,
                    json.dumps(new_choice_expl),
                    q["id"],
                ),
            )
            db.conn.commit()

        changed += 1

    print(f"\nDone: {changed} fixed, {unchanged} already correct, {failed} failed")
    if dry_run:
        print("(dry run — no changes written)")


if __name__ == "__main__":
    asyncio.run(main())
