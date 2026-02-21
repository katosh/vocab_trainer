"""Prompt templates for question generation."""
from __future__ import annotations

FILL_BLANK_PROMPT = """\
You are generating a vocabulary quiz question. You will create a fill-in-the-blank \
question that tests precise word choice among near-synonyms.

Here is the vocabulary cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}**
Meaning: {target_meaning}
Key distinction: {target_distinction}

{enrichment_section}

Instructions:
1. Write a rich, natural English sentence (15-30 words) where ONLY the target word \
fits perfectly. The sentence should make the target word's specific shade of meaning \
clearly the best choice.
2. Replace the target word with "___" in the stem.
3. Provide exactly 4 choices: the correct word and 3 distractors from the SAME cluster. \
The distractors should be plausible but specifically wrong based on the distinctions.
4. Write a brief explanation (1-2 sentences) of why the correct word fits and the \
best distractor doesn't.
5. For EACH choice, write a short explanation (1 sentence) of why it is correct or \
why it doesn't fit this specific sentence. Reference the sentence context, not just \
the generic definition.
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
  "explanation": "Why this word fits best",
  "choice_explanations": ["why word1 fits/doesn't", "why word2 fits/doesn't", "why word3 fits/doesn't", "why word4 fits/doesn't"],
  "context_sentence": "Full sentence with answer filled in"
}}
"""

BEST_FIT_PROMPT = """\
You are generating a vocabulary quiz question. You will create a "best fit" \
question that tests understanding of subtle word distinctions.

Here is the vocabulary cluster "{cluster_title}":
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
4. Provide exactly 4 choices from the same cluster.
5. Write a brief explanation.
6. For EACH choice, write a short explanation (1 sentence) of why it is correct or \
why it doesn't fit this specific scenario. Reference the scenario, not just the definition.
7. Provide a sentence using the correct word that captures the scenario.

Respond in this exact JSON format only, with no other text:
{{
  "stem": "Scenario description. Which word best describes...?",
  "choices": ["word1", "word2", "word3", "word4"],
  "correct_index": 0,
  "explanation": "Why this word fits best",
  "choice_explanations": ["why word1 fits/doesn't", "why word2 fits/doesn't", "why word3 fits/doesn't", "why word4 fits/doesn't"],
  "context_sentence": "A sentence using the correct word"
}}
"""

DISTINCTION_PROMPT = """\
You are generating a vocabulary quiz question. You will create a question that \
tests explicit knowledge of the distinction between near-synonyms.

Here is the vocabulary cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}**
Meaning: {target_meaning}
Key distinction: {target_distinction}

{enrichment_section}

Instructions:
1. Ask a question about the key distinction of the target word, e.g., \
"Which word specifically implies [distinctive quality]?" or \
"What is the key difference between {target_word} and [another word]?"
2. The question should test whether the student knows the PRECISE shade of meaning.
3. Provide exactly 4 choices from the same cluster.
4. Write a brief explanation referencing the key distinction.
5. For EACH choice, write a short explanation (1 sentence) of why it is correct or \
why it doesn't match the distinction being tested.
6. Provide a sentence illustrating the correct word's distinctive meaning.

Respond in this exact JSON format only, with no other text:
{{
  "stem": "Question about the distinction",
  "choices": ["word1", "word2", "word3", "word4"],
  "correct_index": 0,
  "explanation": "Why this word is correct based on its distinction",
  "choice_explanations": ["why word1 fits/doesn't", "why word2 fits/doesn't", "why word3 fits/doesn't", "why word4 fits/doesn't"],
  "context_sentence": "A sentence illustrating the word's specific meaning"
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
