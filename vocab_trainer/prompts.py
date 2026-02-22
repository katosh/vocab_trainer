"""Prompt templates for question generation."""
from __future__ import annotations

FILL_BLANK_PROMPT = """\
You are generating a vocabulary quiz question. Create a fill-in-the-blank \
question that tests precise word choice among near-synonyms.

Cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}**
Meaning: {target_meaning}
Key distinction: {target_distinction}

{enrichment_section}

Instructions:
1. Write a rich, natural English sentence (15-30 words) where ONLY the target word \
fits perfectly. The sentence should create a context where the target word's \
specific nuance is clearly required — a near-synonym would feel wrong to a native ear.
2. Replace the target word with "___" in the stem.
3. Provide exactly 4 choices: "{target_word}" and 3 distractors from the SAME cluster. \
The distractors should be plausible but specifically wrong based on the distinctions.
4. Set correct_index to the position of "{target_word}" in the choices array (0-based).
5. Write an explanation (2-3 sentences) of why the correct word fits AND why the \
closest distractor specifically doesn't.
6. Provide the full sentence with the answer filled in (for text-to-speech).

ALL four choices MUST be conjugated/inflected identically to fit the blank's grammatical \
slot. If the sentence needs a past tense verb, make every choice past tense. If it needs \
an adjective, make every choice an adjective. The student chooses based on MEANING, \
not grammar.

Respond in this exact JSON format only, with no other text:
{{
  "stem": "sentence with ___ blank",
  "choices": ["word1", "word2", "word3", "word4"],
  "correct_index": 0,
  "explanation": "Why this word fits best and why the closest distractor doesn't",
  "context_sentence": "Full sentence with answer filled in"
}}
"""

BEST_FIT_PROMPT = """\
You are generating a vocabulary quiz question. Create a "best fit" \
question that tests understanding of subtle word distinctions.

Cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}**
Meaning: {target_meaning}
Key distinction: {target_distinction}

{enrichment_section}

Instructions:
1. Write a short scenario or description (2-3 sentences) where a specific quality, \
action, or characteristic is described WITHOUT using any of the cluster words.
2. Ask: "Which word best describes [the key element]?"
3. The scenario must make the target word clearly the best fit based on its specific \
shade of meaning.
4. Provide exactly 4 choices: "{target_word}" and 3 distractors from the SAME cluster. \
The target word MUST appear in the choices array.
5. Set correct_index to the position of "{target_word}" in the choices array (0-based).
6. Write an explanation (2-3 sentences) that teaches the distinction.
7. Provide a sentence using the correct word that captures the scenario.

Respond in this exact JSON format only, with no other text:
{{
  "stem": "Scenario description. Which word best describes...?",
  "choices": ["word1", "word2", "word3", "word4"],
  "correct_index": 0,
  "explanation": "Why this word fits best",
  "context_sentence": "A sentence using the correct word"
}}
"""

DISTINCTION_PROMPT = """\
You are generating a vocabulary quiz question testing the distinction between near-synonyms.

Cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}**
Meaning: {target_meaning}
Key distinction: {target_distinction}

{enrichment_section}

Instructions:
1. Write a question that tests whether the student truly *understands* the nuance of \
"{target_word}" versus its near-synonyms. VARY your question format — use one of these:
   - A scenario + "Which word applies here?" (e.g. "A diplomat who rejected a treaty on \
behalf of their government would be said to ___ it. Which word fits?")
   - A distinction question (e.g. "Which of these words implies X rather than Y?")
   - A usage question (e.g. "You would use which word when describing X?")
   Do NOT always use "Which word specifically implies..." — vary the phrasing.
2. The question must have only ONE defensible answer: "{target_word}".
3. Provide exactly 4 choices: "{target_word}" and 3 distractors from the SAME cluster. \
The target word MUST appear in the choices array.
4. Set correct_index to the position of "{target_word}" in the choices array (0-based).
5. Write an explanation (2-3 sentences) that teaches the distinction — reference how \
the distractors differ.
6. Provide a sentence illustrating the target word's distinctive meaning.

Respond in this exact JSON format only, with no other text:
{{
  "stem": "Your question here",
  "choices": ["word1", "word2", "word3", "word4"],
  "correct_index": 0,
  "explanation": "Why this word is correct based on its distinction",
  "context_sentence": "A sentence illustrating the word's specific meaning"
}}
"""


CHOICE_ENRICHMENT_PROMPT = """\
Given this quiz question and the vocabulary cluster it draws from, analyze each choice.

Cluster "{cluster_title}":
{cluster_info}

Question stem: {stem}
Choices: {choices_formatted}
Correct answer: {correct_word} (index {correct_index})

For each choice, provide:
- "word": the choice exactly as shown
- "base_word": the cluster word it derives from (e.g. "heralded" → "herald")
- "meaning": what this word means (from the cluster definition)
- "distinction": how it differs from the other cluster words
- "why": 1 sentence explaining why it does or doesn't fit THIS specific sentence/scenario

JSON only:
{{
  "choice_details": [
    {{"word": "...", "base_word": "...", "meaning": "...", "distinction": "...", "why": "..."}},
    ...
  ]
}}
"""


def format_cluster_info(entries: list[dict]) -> str:
    lines = []
    for e in entries:
        lines.append(f"- **{e['word']}**: {e['meaning']} — {e['distinction']}")
    return "\n".join(lines)


def format_enrichment(extra_words: list[dict]) -> str:
    if not extra_words:
        return ""
    words_str = ", ".join(f"**{w['word']}** ({w['definition'][:60]})" for w in extra_words)
    return (
        f"For richer context, you may weave in these vocabulary words: {words_str}\n"
        "But only if they fit naturally — do not force them."
    )
