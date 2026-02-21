"""Apply comprehensive corrections to questions_export.json.

Fixes: conjugation mismatches, part-of-speech errors, grammar issues,
context sentence errors, explanation typos.

Usage:
    python scripts/apply_corrections.py
"""
import json
from pathlib import Path

EXPORT_PATH = Path("questions_export.json")

# Each correction: id → dict of fields to overwrite
CORRECTIONS = {

    # ── Past-tense conjugation ──────────────────────────────────────────

    # check: "The researcher meticulously ___ the data"
    "096481ff-5da0-41ab-b20b-52a5dedab081": {
        "choices": ["checked", "stemmed", "tabled", "flagged"],
    },

    # claim: "The teacher ___ that the answer was correct"
    "cebaf9c6-0ddb-4bdb-ab6d-4e5eda2e0712": {
        "choices": ["claimed", "asserted", "declared", "professed"],
    },

    # cleave: "The meticulous archaeologist ___ the stone"
    "00311f93-c3aa-4594-a105-14ccd0a6432c": {
        "choices": ["cleaved", "mooted", "sanctioned", "tempered"],
    },

    # confirm: "The scientist's data ___ the long-held theory"
    # (also: context sentence didn't match stem)
    "acc61afc-e272-425e-99b4-8aa09ffea1e3": {
        "choices": ["confirmed", "established", "verified", "authenticated"],
        "context_sentence": "The scientist's data confirmed the long-held theory, sealing its acceptance among peers.",
    },

    # depreciate: "The company's stock ___ sharply"
    "35fd370f-d1df-477e-8728-73dff23f4ce6": {
        "choices": ["depreciated", "deprecated", "flaunted", "flouted"],
    },

    # desecrate: "The vandals ___ the ancient temple"
    "b662f5c1-2af6-48a8-94cd-5f7624632f4f": {
        "choices": ["desecrated", "violated", "transgressed", "profaned"],
    },

    # deviate: "The researcher ___ from protocol"
    "7d245685-c7e3-4ea1-be76-1cca602bf30d": {
        "choices": ["deviated", "digressed", "diverged", "meandered"],
    },

    # dictate: "The leader ___ terms" (+ context had wrong tense)
    "3a95adeb-487c-4947-b3c6-0e4ff8716311": {
        "choices": ["dictated", "decreed", "pronounced", "editorialized"],
        "context_sentence": "The leader dictated terms without consulting the team, asserting unilateral control",
    },

    # excoriate: "The critic ___ the film's plot"
    "b67b875a-81ad-4409-9e46-744af784b63d": {
        "choices": ["excoriated", "castigated", "berated", "rebuked"],
    },

    # herald #1: "The leader's speech ___ the arrival"
    "3fbf00f9-6f63-45e1-8a33-259d0da5d802": {
        "choices": ["heralded", "declared", "proclaimed", "trumpeted"],
    },

    # herald #2: "The artist's debut album ___ the arrival"
    "d242a5e8-4119-407d-aa08-5a7f0a7a47d0": {
        "choices": ["heralded", "declared", "proclaimed", "trumpeted"],
    },

    # impugn: "The journalist ___ the witness's credibility"
    "d5c66d3c-33a1-466b-80ea-502bad3bfe0a": {
        "choices": ["impugned", "denied", "dismissed", "rebutted"],
    },

    # prostrate: "she ___ to the floor"
    "3594b654-a821-4d49-8b7b-e05f769937f5": {
        "choices": ["prostrated", "capitulated", "conceded", "submitted"],
    },

    # reprove: "The mentor ___ the student's careless mistakes"
    "d898e972-6d34-496c-bfdd-6b13d8826998": {
        "choices": ["reproved", "rebuked", "reproached", "chided"],
    },

    # undermine: "The manager's constant criticism ___ the team's confidence"
    "768b506b-d256-4080-b0ea-1b8a0c8f62cc": {
        "choices": ["undermined", "rebutted", "vitiated", "invalidated"],
        "context_sentence": "The manager's constant criticism undermined the team's confidence over time, leaving them paralyzed by self-doubt.",
    },

    # usurp: "The dictator ___ the throne"
    "4883f929-9cae-4719-94ae-d9129a8189a2": {
        "choices": ["usurped", "supplanted", "arrogated", "annexed"],
    },

    # ── Part-of-speech mismatches ───────────────────────────────────────

    # acquitted: "not guilty" is a phrase/adjective, not a verb
    "d3c43b2b-b8c1-4868-b088-4e687816c1a1": {
        "choices": ["acquitted", "vindicated", "exonerated", "absolved"],
        "choice_explanations": [
            "Acquitted fits perfectly as it describes the court's formal action ending prosecution and triggering double jeopardy protections.",
            "Vindicated implies proving innocence or correctness, but the jury's action is a procedural outcome, not a declaration of actual innocence.",
            "Exonerated implies actual innocence and post-trial vindication, which isn't the procedural endpoint described here.",
            "Absolved carries religious/moral overtones and doesn't align with the legal procedural context of ending prosecution.",
        ],
    },

    # expurgate: "censorious" is adjective, can't fill "decided to ___"
    "99ec4c44-5f21-4f84-9d26-547e41cb8760": {
        "choices": ["expurgate", "censor", "censure", "bowdlerize"],
        "choice_explanations": [
            "Correct: 'Expurgate' directly means removing disreputable content, matching the sentence's need for precise editing.",
            "Incorrect: 'Censor' implies suppression but lacks the specificity of removing particular objectionable material.",
            "Incorrect: 'Censure' refers to official disapproval, not the act of editing text.",
            "Incorrect: 'Bowdlerize' means to remove content prudishly, but implies excessive or unnecessary cuts rather than the targeted editing described.",
        ],
    },

    # federal: "republic" is a noun; blank needs adjective ("The ___ system")
    "a78c8498-a7d0-4f31-abad-1677c5b0b970": {
        "choices": ["federal", "unitary", "autonomous", "democratic"],
        "choice_explanations": [
            "Federal correctly describes a system where power is divided between central and regional governments",
            "Unitary implies all power is centralized, contradicting the shared sovereignty required for distinct roles",
            "Autonomous refers to partial independence, but the sentence describes structured division of power between two levels",
            "Democratic describes governance by the people, but doesn't address the structural division of power between levels of government",
        ],
    },

    # hedge: "qualification", "reservation", "condition" are nouns in a verb slot
    "e756cfb7-e465-4c66-85e4-6ab2c5ffa0a0": {
        "choices": ["hedged", "qualified", "tempered", "softened"],
        "context_sentence": "The author hedged their claim by using 'arguably' to avoid overcommitment to a controversial stance.",
        "explanation": "Hedge perfectly captures the rhetorical softening of commitment, while qualification (an intellectual modifier) and reservation (an emotional hesitation) misalign with the sentence's focus on verbal strategy.",
        "choice_explanations": [
            "Hedged fits perfectly: it captures the deliberate rhetorical weakening of a claim, matching the strategic use of 'arguably'.",
            "Qualified implies adding conditions or limits, but the sentence describes softening rather than restricting the claim.",
            "Tempered suggests moderating the force of the claim, but lacks the specific rhetorical evasiveness that 'hedged' conveys.",
            "Softened is too general and doesn't capture the strategic, deliberate nature of the rhetorical move described.",
        ],
    },

    # hiraeth: "nostalgic" and "wistful" are adjectives; blank needs nouns
    "221905dd-88ee-41cd-9280-81268aead332": {
        "choices": ["hiraeth", "nostalgia", "saudade", "wistfulness"],
        "choice_explanations": [
            "Hiraeth fits perfectly as it specifically refers to longing for a lost homeland that may no longer exist, matching the vanished hills metaphor.",
            "Nostalgia is too generic and lacks the specific cultural/territorial focus of hiraeth, which the vanished hills imply.",
            "Saudade emphasizes bittersweet acceptance of absence, but the sentence frames the longing as unresolved and tied to a specific place.",
            "Wistfulness suggests gentle yearning, but the sentence's tone is more profound and tied to a specific, vanished homeland.",
        ],
    },

    # indict: verbs in noun slot ("The grand jury's ___ of the mayor")
    # → rewrite stem to use verb form
    "baa512c6-1318-421e-bb34-6003500b2f11": {
        "stem": "The grand jury ___ the mayor for embezzlement, sparking public outrage as the evidence was deemed abysmal.",
        "choices": ["indicted", "charged", "accused", "arraigned"],
        "context_sentence": "The grand jury indicted the mayor for embezzlement, sparking public outrage as the evidence was deemed abysmal.",
        "explanation": "Indict specifically requires a grand jury's review and probable cause, fitting the context of a formal felony charge.",
        "choice_explanations": [
            "Indicted correctly captures the grand jury's role in formally charging a felony, aligning with the sentence's legal context.",
            "Charged is too generic and lacks the grand jury procedural specificity required for the sentence's context.",
            "Accused is informal and lacks the legal procedural nuance of a grand jury's formal action.",
            "Arraigned refers to a court appearance and plea, not the grand jury's role in initiating charges.",
        ],
    },

    # ought: "obligation" is noun among modals; also fix "ought [to]" grammar
    "d22dafb8-0162-4487-bddc-ee486e786ac3": {
        "choices": ["ought", "should", "must", "shall"],
        "context_sentence": "You ought to apologize for the bêtise, but it's not obligatory.",
        "choice_explanations": [
            "Ought fits perfectly: it's a moral recommendation with room for judgment, matching the context of a discretionary apology.",
            "Should is too weak; it implies advisability but lacks the moral weight of 'ought' in this context.",
            "Must implies absolute necessity, conflicting with the sentence's 'not obligatory' nuance.",
            "Shall implies formal command or legal obligation, which is too strong for the discretionary apology described.",
        ],
    },

    # proliferate: verbs in noun slot ("The rapid ___ of bacteria")
    # → rewrite stem to use verb form
    "3e26184d-088e-449d-98dc-5ab5ae55f965": {
        "stem": "Antibiotic-resistant bacteria ___ rapidly, creating a global health crisis.",
        "choices": ["proliferate", "propagate", "disseminate", "pervade"],
        "context_sentence": "Antibiotic-resistant bacteria proliferate rapidly, creating a global health crisis.",
        "explanation": "Proliferate emphasizes uncontrolled multiplication, matching the crisis caused by rapidly increasing bacteria.",
        "choice_explanations": [
            "Proliferate fits perfectly as it conveys uncontrolled, rapid growth of bacteria.",
            "Propagate suggests intentional spreading, but the crisis implies uncontrolled growth, not deliberate action.",
            "Disseminate implies deliberate distribution, which doesn't align with the chaotic spread of bacteria.",
            "Pervade focuses on saturation, not the rapid numerical increase described in the sentence.",
        ],
    },

    # propitious: "harbinger" is a noun; blank needs adjective ("The ___ climate")
    "f9d87c07-89e4-4bc3-a087-182e529c674a": {
        "choices": ["propitious", "auspicious", "portentous", "ominous"],
        "choice_explanations": [
            "Propitious fits because it directly refers to favorable conditions enabling success.",
            "Auspicious implies a promising beginning, but the sentence emphasizes existing favorable conditions, not an initial sign.",
            "Portentous suggests something momentous or ominously significant, which doesn't match the straightforwardly positive context.",
            "Ominous suggests negativity, contradicting the sentence's positive context.",
        ],
    },

    # sentence: "convict", "punish", "condemn" are verbs; blank needs noun
    "864631a2-423c-4273-92f0-5cbe6a759412": {
        "choices": ["sentence", "conviction", "punishment", "condemnation"],
        "choice_explanations": [
            "Correct: 'sentence' directly refers to the judicial punishment imposed by a court after a verdict, matching the context of sentencing.",
            "Incorrect: 'conviction' is the finding of guilt, which occurs before the sentence is handed down, not the punishment itself.",
            "Incorrect: 'punishment' is a general term for imposing consequences, lacking the specific judicial context of a court-ordered sentence.",
            "Incorrect: 'condemnation' carries moral overtones and can apply to non-judicial contexts, unlike the formal judicial process of sentencing.",
        ],
    },

    # steadfast: "integrity" is noun; blank needs adjective ("remained ___")
    "94e349db-67b6-497b-9a79-26d3d0a572ca": {
        "choices": ["steadfast", "resolute", "stalwart", "unwavering"],
        "choice_explanations": [
            "Steadfast perfectly fits the context of refusing to yield despite fierce contention, emphasizing immovability",
            "Resolute suggests determination but doesn't capture the refusal to be swayed by external pressures in the same way",
            "Stalwart implies reliability but focuses more on consistent support rather than refusal to change stance",
            "Unwavering describes firmness without shaking, but lacks the deeper connotation of loyalty and commitment under pressure",
        ],
    },

    # ── Grammar / context fixes ─────────────────────────────────────────

    # anachronistic: context "a anachronistic" → "an anachronistic"
    "9c4d8ba4-ec1b-4332-a66e-f41203133cd5": {
        "context_sentence": "The museum's exhibit featured an anachronistic device from the 22nd century, which seemed out of place among ancient artifacts.",
    },

    # probable: "The study's findings are probable that..." is ungrammatical
    "2bf196cc-3da3-4310-9bed-a201fa5530c5": {
        "stem": "Based on the study's findings, it is ___ that the new treatment will reduce symptoms by 30% within six months.",
        "context_sentence": "Based on the study's findings, it is probable that the new treatment will reduce symptoms by 30% within six months.",
    },

    # ── Explanation errors ──────────────────────────────────────────────

    # gullible: explanation says "Ingenious" but choice is "ingenuous"
    "1f80be9d-cebe-4f00-8215-47d178c164a3": {
        "choice_explanations": [
            "Gullible fits perfectly as it emphasizes the negative, direct pejorative connotation of being easily tricked by persuasive claims.",
            "Credulous is too formal and intellectualized, lacking the direct pejorative edge needed for the context of a scam victim.",
            "Naive is neutral and epistemic, missing the moral judgment implied by the scam's exploitation.",
            "Ingenuous implies positive innocence linked to trustworthiness, contradicting the negative context of being deceived.",
        ],
    },

    # ── Round 2 fixes ─────────────────────────────────────────────────

    # undercut: "was ___" needs past participles; undermine/refute/rebut are base form
    "63cf945b-d807-43d3-8750-6434695b8ed1": {
        "choices": ["undercut", "undermined", "refuted", "rebutted"],
    },

    # unitary: "autocracy" is a noun; "___ nations" needs adjective
    "a546c258-835e-4330-89ac-002ac21eba5e": {
        "choices": ["unitary", "federal", "autocratic", "sovereign"],
    },

    # amnesty: duplicate of eb6b18b3 (identical stem) → archive
    "11034fb2-7c0f-48b0-9a35-b2ebe2ebf0f1": {
        "archived": 1,
    },

    # prostrate: "she prostrated to the floor" → more natural phrasing
    "3594b654-a821-4d49-8b7b-e05f769937f5": {
        "stem": "After the harsh criticism, she ___ herself on the floor, her body mirroring her mental defeat",
        "context_sentence": "After the harsh criticism, she prostrated herself on the floor, her body mirroring her mental defeat",
    },
}


def main():
    data = json.loads(EXPORT_PATH.read_text())
    by_id = {q["id"]: q for q in data}

    applied = 0
    missing = []

    for qid, fixes in CORRECTIONS.items():
        if qid not in by_id:
            missing.append(qid)
            continue

        q = by_id[qid]
        for field, value in fixes.items():
            old = q.get(field)
            q[field] = value
            if old != value:
                print(f"  [{q['target_word']}] {field} updated")
        applied += 1

    if missing:
        print(f"\nWARNING: {len(missing)} IDs not found: {missing}")

    EXPORT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"\nApplied corrections to {applied} questions, wrote {EXPORT_PATH}")


if __name__ == "__main__":
    main()
