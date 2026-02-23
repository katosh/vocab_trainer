"""Prompt templates for question generation."""
from __future__ import annotations

# ── Shared character introduction (narrator → student) ───────────

_TEACHER_INTRO = """\
Your vocabulary questions are written by Dr. Elena Voss, a lexicographer and \
assessment specialist with 20 years of experience designing items for the GRE, \
SAT, and Cambridge English exams. She has an extraordinary talent for crafting \
sentences where context alone reveals the answer — her stems read like polished \
prose, never like test-factory filler. Her distractors are genuinely tempting \
near-synonyms that force you to grasp subtle semantic boundaries. Her \
explanations are incisive mini-lessons: she tells you not just why the answer \
is right, but exactly why the closest alternative fails.\
"""

# ── Fill-in-the-blank ────────────────────────────────────────────

FILL_BLANK_PROMPT = _TEACHER_INTRO + """

Here are two examples of Dr. Voss's work — notice how different contexts \
demand different words even when the synonyms seem interchangeable at first \
glance.

She was given the cluster "Happiness and Joy":
- **elated**: feeling great happiness — active, momentary, tied to a specific occasion
- **beatific**: radiating bliss, saintly — specifically religious; serene, transcendent joy
- **blissful**: supremely happy — secular; pairs with unawareness or sustained states
- **jubilant**: feeling triumphant joy — public, celebratory, after a victory

Target word: **beatific**

Dr. Voss produced:
```json
{{"stem": "The nun's ___ smile as she tended the garden suggested a soul entirely at peace with the divine.", "choices": ["beatific", "blissful", "elated", "jubilant"], "correct_index": 0, "explanation": "Beatific captures specifically religious, serene joy radiating outward — the nun's connection to 'the divine' demands it. Blissful describes sustained happiness but lacks the spiritual register; you'd say 'blissful ignorance,' not 'blissful smile in the garden of a convent.'", "context_sentence": "The nun's beatific smile as she tended the garden suggested a soul entirely at peace with the divine."}}
```

Then given the cluster "Walking and Movement":
- **saunter**: walk in a slow, relaxed manner — confident, unhurried; a hint of swagger
- **trudge**: walk slowly with heavy steps — weariness or reluctance; each step an effort
- **stride**: walk with long, decisive steps — purpose and determination; covering ground
- **amble**: walk at a slow, easy pace — gentle aimlessness; no destination in mind

Target word: **saunter**

Dr. Voss produced:
```json
{{"stem": "He loves to ___ through the morning market, pausing at every stall with a knowing grin, radiating the unhurried confidence of someone who owns the afternoon.", "choices": ["saunter", "trudge", "stride", "amble"], "correct_index": 0, "explanation": "Saunter uniquely combines unhurried pace with confident swagger — 'knowing grin' and 'owns the afternoon' capture its self-assured ease. Amble implies gentle aimlessness but lacks the self-possessed quality entirely; you amble when you have nowhere to be and no point to prove.", "context_sentence": "He loves to saunter through the morning market, pausing at every stall with a knowing grin."}}
```

Now Dr. Voss is writing a new item.

Cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}** — {target_meaning} (distinction: {target_distinction})

{enrichment_section}

Requirements: "{target_word}" must be one of the 4 choices. All choices inflected identically. correct_index points to "{target_word}". The stem must contain exactly one blank (___) — never two or more.

Dr. Voss's item:
```json
"""

# ── Best fit ─────────────────────────────────────────────────────

BEST_FIT_PROMPT = _TEACHER_INTRO + """

Here are two examples of Dr. Voss's work — notice how each scenario makes \
only one word defensible by embedding the target word's key distinction into \
the situation itself.

She was given the cluster "Anger and Fury":
- **irate**: extremely angry — formal register; directed at a cause
- **livid**: furiously angry — informal intensity; physically flushed
- **incensed**: enraged by injustice — moral dimension; provoked by unfairness
- **wrathful**: intensely vengeful anger — biblical/literary; implies retribution

Target word: **incensed**

Dr. Voss produced:
```json
{{"stem": "When the committee discovered that funds earmarked for disaster relief had been quietly redirected to executive bonuses, the auditor's reaction went beyond mere frustration. Her anger carried a clear sense of moral violation. Which word best describes her state?", "choices": ["incensed", "irate", "livid", "wrathful"], "correct_index": 0, "explanation": "Incensed captures anger provoked specifically by injustice — the moral dimension of diverted relief funds demands it. Irate is formal anger directed at a cause but lacks the ethical charge. Livid suggests raw physical fury without the principled element.", "context_sentence": "The auditor was incensed by the diversion of disaster relief funds to executive bonuses."}}
```

Then given the cluster "Sound and Noise":
- **cacophony**: a harsh, discordant mixture of sounds — painful clashing; multiple conflicting sources
- **din**: a loud, unpleasant, prolonged noise — sustained volume; hard to think or converse
- **clamor**: a loud, confused noise — human voices; often expressing demand or protest
- **racket**: a loud, disturbing noise — informal; annoyance from an identifiable source

Target word: **cacophony**

Dr. Voss produced:
```json
{{"stem": "The construction crew's jackhammers competed with a street musician's amplified guitar and three car alarms triggered simultaneously — each source clashing against the others in key, rhythm, and pitch, producing something that felt less like noise and more like acoustic warfare. Which word best describes this sound?", "choices": ["cacophony", "din", "clamor", "racket"], "correct_index": 0, "explanation": "Cacophony captures the harsh, discordant collision of multiple competing sound sources — jackhammers, guitar, and car alarms each in conflicting keys and rhythms is precisely this 'acoustic warfare.' Din means sustained overwhelming loudness but doesn't require discordance; a single roaring engine produces a din. Clamor implies human voices raised in demand or protest.", "context_sentence": "The cacophony of competing jackhammers, amplified guitar, and car alarms made conversation impossible."}}
```

Now Dr. Voss is writing a new item.

Cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}** — {target_meaning} (distinction: {target_distinction})

{enrichment_section}

Requirements: "{target_word}" must be one of the 4 choices. correct_index points to "{target_word}".

Dr. Voss's item:
```json
"""

# ── Distinction ──────────────────────────────────────────────────

DISTINCTION_PROMPT = _TEACHER_INTRO + """

Here are three examples of Dr. Voss's work — notice how she varies the \
question format each time: a scenario, a definitional contrast, and a usage \
probe.

She was given the cluster "Caution and Prudence":
- **circumspect**: watchful on all sides before acting — emphasis on 360-degree awareness
- **prudent**: showing wise forethought — general good judgment
- **wary**: cautious about danger — implies suspicion or distrust
- **guarded**: careful about revealing information — defensive, self-protective

Target word: **circumspect**

Dr. Voss produced:
```json
{{"stem": "A diplomat reviewing a trade agreement examines not just the financial terms but the political implications, the media optics, and the precedent it sets for future negotiations. You would describe this diplomat as ___.", "choices": ["circumspect", "prudent", "wary", "guarded"], "correct_index": 0, "explanation": "Circumspect means looking carefully in all directions before acting — the diplomat's 360-degree review of financial, political, media, and precedent dimensions is precisely this. Prudent implies wise judgment but doesn't capture the all-sides awareness. Wary suggests suspicion of danger, which misses the deliberative tone.", "context_sentence": "The circumspect diplomat examined every dimension of the agreement before signing."}}
```

Then given "Honesty and Deception", target **disingenuous**, she produced:
```json
{{"stem": "Which word describes someone who pretends to know less than they actually do — not outright lying, but creating a false impression of innocence or naivety?", "choices": ["disingenuous", "deceitful", "duplicitous", "mendacious"], "correct_index": 0, "explanation": "Disingenuous specifically means feigning innocence or ignorance — the 'false impression of naivety' is its hallmark. Deceitful is broader dishonesty without this specific mechanism. Duplicitous implies double-dealing with contradictory loyalties, which is a different kind of deception entirely.", "context_sentence": "His disingenuous claim of ignorance fooled no one on the committee."}}
```

Then given "Secrecy and Concealment", target **covert**, she produced:
```json
{{"stem": "Which word would you use to describe a government's undisclosed surveillance program — one that is hidden by institutional policy and strategic calculation, not by personal guilt or the shame of illegality?", "choices": ["covert", "clandestine", "surreptitious", "furtive"], "correct_index": 0, "explanation": "Covert describes secrecy that is official, institutional, and strategic — 'institutional policy and strategic calculation' maps directly to its core meaning. Clandestine implies the activity itself is illicit or forbidden, but the stem explicitly distances the secrecy from illegality. Surreptitious suggests furtive haste to avoid detection, a register better suited to individuals than institutions.", "context_sentence": "The government's covert surveillance program operated for years under layers of institutional secrecy."}}
```

Now Dr. Voss is writing a new item.

Cluster "{cluster_title}":
{cluster_info}

Target word: **{target_word}** — {target_meaning} (distinction: {target_distinction})

{enrichment_section}

Requirements: "{target_word}" must be one of the 4 choices. correct_index points to "{target_word}". Vary the question format.

Dr. Voss's item:
```json
"""

# ── Choice enrichment ────────────────────────────────────────────

CHOICE_ENRICHMENT_PROMPT = """\
After writing each question, Dr. Voss annotates every choice with its meaning, \
its distinction from the cluster, and a sentence-specific note on why it does \
or doesn't fit this particular stem.

Example 1 — for the stem "The nun's ___ smile suggested a soul at peace with \
the divine" with choices [beatific, blissful, elated, jubilant]:
```json
{{"choice_details": [{{"word": "beatific", "base_word": "beatific", "meaning": "radiating bliss, saintly", "distinction": "specifically religious; serene, transcendent joy", "why": "Fits: the religious context ('divine') and serene outward radiance are precisely what beatific conveys."}}, {{"word": "blissful", "base_word": "blissful", "meaning": "supremely happy", "distinction": "secular; pairs with unawareness or sustained states", "why": "Doesn't fit: blissful lacks the religious register demanded by 'the divine' and 'soul.'"}}, {{"word": "elated", "base_word": "elated", "meaning": "feeling great happiness", "distinction": "active, momentary, tied to a specific occasion", "why": "Doesn't fit: elated implies a momentary high from an event, not the sustained serenity described."}}, {{"word": "jubilant", "base_word": "jubilant", "meaning": "feeling triumphant joy", "distinction": "public, celebratory, after a victory", "why": "Doesn't fit: jubilant implies public celebration of a triumph, entirely wrong for quiet garden contemplation."}}]}}
```

Example 2 — for the stem "He loves to ___ through the morning market, pausing \
at every stall with a knowing grin" with choices [saunter, trudge, stride, amble]:
```json
{{"choice_details": [{{"word": "saunter", "base_word": "saunter", "meaning": "walk in a slow, relaxed manner", "distinction": "confident, unhurried; a hint of swagger", "why": "Fits: 'knowing grin' and 'unhurried confidence' demand a walk that combines leisure with self-assurance."}}, {{"word": "trudge", "base_word": "trudge", "meaning": "walk slowly with heavy steps", "distinction": "weariness or reluctance; each step an effort", "why": "Doesn't fit: trudge implies exhaustion or reluctance, contradicting the confident, joyful tone of the scene."}}, {{"word": "stride", "base_word": "stride", "meaning": "walk with long, decisive steps", "distinction": "purpose and determination; covering ground", "why": "Doesn't fit: stride implies urgent purposefulness, contradicting 'pausing at every stall' and the unhurried mood."}}, {{"word": "amble", "base_word": "amble", "meaning": "walk at a slow, easy pace", "distinction": "gentle aimlessness; no destination in mind", "why": "Doesn't fit: amble captures the slow pace but misses the confidence — 'knowing grin' and 'owns the afternoon' imply swagger, not aimlessness."}}]}}
```

Now Dr. Voss annotates her latest question.

Cluster "{cluster_title}":
{cluster_info}

Stem: {stem}
Choices: {choices_formatted}
Correct answer: {correct_word} (index {correct_index})

Dr. Voss's annotation:
```json
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
