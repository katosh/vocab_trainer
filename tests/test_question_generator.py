"""Tests for question generation (JSON extraction, validation, LLM orchestration)."""
from __future__ import annotations

import json

import pytest

from vocab_trainer.question_generator import (
    _enrich_choices,
    _extract_json,
    _fix_article_before_blank,
    _pick_question_type,
    _pick_word_cluster,
    _validate_question,
    generate_question,
)


class FakeLLM:
    """Simple fake LLM that avoids AsyncMock's `name` attribute issue."""

    def __init__(self, responses=None):
        self._responses = responses or []
        self._call_count = 0

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def name(self) -> str:
        return "fake-llm"

    @property
    def call_count(self):
        return self._call_count


class TestExtractJson:
    def test_bare_json(self):
        text = '{"stem": "test", "choices": ["a","b","c","d"], "correct_index": 0}'
        result = _extract_json(text)
        assert result is not None
        assert result["stem"] == "test"

    def test_code_fence_json(self):
        text = 'Here is the question:\n```json\n{"stem": "test"}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["stem"] == "test"

    def test_code_fence_no_lang(self):
        text = '```\n{"stem": "test"}\n```'
        result = _extract_json(text)
        assert result is not None

    def test_json_with_surrounding_text(self):
        text = 'Sure, here it is:\n\n{"stem": "test", "value": 42}\n\nHope that helps!'
        result = _extract_json(text)
        assert result is not None
        assert result["value"] == 42

    def test_invalid_json(self):
        text = "This is not JSON at all."
        result = _extract_json(text)
        assert result is None

    def test_malformed_json(self):
        text = '{"stem": "missing closing brace"'
        result = _extract_json(text)
        assert result is None

    def test_multiline_json(self):
        text = """{
  "stem": "The professor's ___ analysis cut through the confusion.",
  "choices": ["incisive", "acute", "astute", "shrewd"],
  "correct_index": 0,
  "explanation": "Incisive implies cutting through complexity.",
  "context_sentence": "The professor's incisive analysis cut through the confusion."
}"""
        result = _extract_json(text)
        assert result is not None
        assert len(result["choices"]) == 4


class TestValidateQuestion:
    def _valid_data(self):
        return {
            "stem": "Her ___ reply left no room for pleasantries.",
            "choices": ["terse", "concise", "pithy", "laconic"],
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply left no room for pleasantries.",
        }

    def test_valid(self):
        assert _validate_question(self._valid_data(), "terse") is None

    def test_missing_field(self):
        d = self._valid_data()
        del d["explanation"]
        reason = _validate_question(d, "terse")
        assert reason is not None
        assert "missing fields" in reason

    def test_wrong_choice_count(self):
        d = self._valid_data()
        d["choices"] = ["terse", "concise", "pithy"]
        reason = _validate_question(d, "terse")
        assert reason is not None
        assert "choices" in reason

    def test_index_out_of_range(self):
        d = self._valid_data()
        d["correct_index"] = 5
        reason = _validate_question(d, "terse")
        assert reason is not None
        assert "out of range" in reason

    def test_negative_index(self):
        d = self._valid_data()
        d["correct_index"] = -1
        reason = _validate_question(d, "terse")
        assert reason is not None
        assert "out of range" in reason

    def test_correct_word_mismatch(self):
        d = self._valid_data()
        reason = _validate_question(d, "concise")
        assert reason is not None

    def test_duplicate_choices(self):
        d = self._valid_data()
        d["choices"] = ["terse", "terse", "pithy", "laconic"]
        reason = _validate_question(d, "terse")
        assert reason is not None
        assert "duplicate" in reason

    def test_case_insensitive_word_match(self):
        d = self._valid_data()
        d["choices"][0] = "Terse"
        assert _validate_question(d, "terse") is None

    def test_non_integer_index(self):
        d = self._valid_data()
        d["correct_index"] = "zero"
        reason = _validate_question(d, "terse")
        assert reason is not None
        assert "not an int" in reason

    def test_string_index_coerced(self):
        """correct_index as string "0" should be coerced to int."""
        d = self._valid_data()
        d["correct_index"] = "0"
        assert _validate_question(d, "terse") is None
        assert d["correct_index"] == 0

    def test_wrong_index_auto_fixed(self):
        """If correct_index points to wrong word but target exists in choices, fix it."""
        d = self._valid_data()
        d["correct_index"] = 2  # points to "pithy" instead of "terse"
        assert _validate_question(d, "terse") is None
        assert d["correct_index"] == 0  # auto-fixed to correct position

    def test_blank_normalization(self):
        """Various blank markers should be normalized to ___."""
        d = self._valid_data()
        d["stem"] = "Her ________ reply left no room for pleasantries."
        assert _validate_question(d, "terse") is None
        assert "___" in d["stem"]


class TestFixArticleBeforeBlank:
    def test_a_with_mixed_choices(self):
        stem = "She gave a ___ answer."
        choices = ["elaborate", "terse", "concise", "pithy"]
        assert _fix_article_before_blank(stem, choices) == "She gave a(n) ___ answer."

    def test_an_with_mixed_choices(self):
        stem = "He made an ___ observation."
        choices = ["acute", "sharp", "keen", "incisive"]
        assert _fix_article_before_blank(stem, choices) == "He made a(n) ___ observation."

    def test_no_change_when_all_consonant(self):
        stem = "She gave a ___ reply."
        choices = ["terse", "curt", "blunt", "pithy"]
        assert _fix_article_before_blank(stem, choices) == "She gave a ___ reply."

    def test_no_change_when_all_vowel(self):
        stem = "It was an ___ event."
        choices = ["unusual", "odd", "eerie", "anomalous"]
        assert _fix_article_before_blank(stem, choices) == "It was an ___ event."

    def test_no_article_before_blank(self):
        stem = "The ___ report was praised."
        choices = ["exhaustive", "thorough", "comprehensive", "meticulous"]
        assert _fix_article_before_blank(stem, choices) == "The ___ report was praised."

    def test_preserves_capitalized_a(self):
        stem = "A ___ response echoed."
        choices = ["eloquent", "terse", "pithy", "curt"]
        assert _fix_article_before_blank(stem, choices) == "A(n) ___ response echoed."

    def test_already_fixed(self):
        """Stem with a(n) should not be double-fixed."""
        stem = "She gave a(n) ___ answer."
        choices = ["elaborate", "terse", "concise", "pithy"]
        result = _fix_article_before_blank(stem, choices)
        assert result == "She gave a(n) ___ answer."


class TestPickQuestionType:
    def test_returns_valid_type(self):
        for _ in range(100):
            qtype = _pick_question_type()
            assert qtype in ("fill_blank", "best_fit", "distinction")


class TestPickWordCluster:
    def test_returns_valid_pair(self, populated_db):
        result = _pick_word_cluster(populated_db)
        assert result is not None
        cluster, word_info = result
        assert "title" in cluster
        assert "word" in word_info
        assert "meaning" in word_info

    def test_returns_none_with_no_clusters(self, tmp_db):
        assert _pick_word_cluster(tmp_db) is None

    def test_favors_uncovered_words(self, populated_db):
        """Words with no existing questions should be chosen much more often
        than words that already have several questions."""
        from vocab_trainer.models import Question
        import uuid

        # Give "terse" 10 banked questions in the "Being Brief" cluster
        for _ in range(10):
            populated_db.save_question(Question(
                id=str(uuid.uuid4()),
                question_type="fill_blank",
                stem="A ___ reply.",
                choices=["terse", "concise", "pithy", "laconic"],
                correct_index=0,
                correct_word="terse",
                explanation="test",
                context_sentence="A terse reply.",
                cluster_title="Being Brief",
                llm_provider="test",
            ))

        # Sample many times and count how often "terse" is picked
        terse_count = 0
        n = 200
        for _ in range(n):
            result = _pick_word_cluster(populated_db)
            assert result is not None
            _, word_info = result
            if word_info["word"] == "terse":
                terse_count += 1

        # "terse" has weight 1/(1+10) = 0.091, other 5 words have 1/(1+0) = 1.0 each
        # Expected terse fraction ≈ 0.091 / (5*1.0 + 0.091) ≈ 1.8%
        # It should be picked much less than uniform (1/6 ≈ 16.7%)
        assert terse_count < n * 0.08, (
            f"terse picked {terse_count}/{n} times — weighting not working"
        )


def _make_enrichment_response(choices, cluster_words_map=None):
    """Build a valid enrichment JSON response for the given choices."""
    details = []
    for c in choices:
        base = c.lower()
        meaning, distinction = "", ""
        if cluster_words_map:
            # Try exact then stem stripping
            for suffix in ("", "ly", "ed", "d", "ing", "s", "es"):
                key = base[: -len(suffix)] if suffix and base.endswith(suffix) else base
                if key in cluster_words_map:
                    meaning = cluster_words_map[key]["meaning"]
                    distinction = cluster_words_map[key]["distinction"]
                    base = key
                    break
        details.append({
            "word": c,
            "base_word": base,
            "meaning": meaning or f"meaning of {c}",
            "distinction": distinction or f"distinction of {c}",
            "why": f"Why {c} does or doesn't fit this sentence.",
        })
    return json.dumps({"choice_details": details})


class TestGenerateQuestion:
    @pytest.mark.asyncio
    async def test_with_mock_llm(self, populated_db):
        choices = ["terse", "concise", "pithy", "laconic"]
        step1 = json.dumps({
            "stem": "Her ___ reply surprised everyone.",
            "choices": choices,
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply surprised everyone.",
        })
        step2 = _make_enrichment_response(choices)

        llm = FakeLLM(responses=[step1, step2])

        cluster = populated_db.get_random_cluster()
        cw = populated_db.get_cluster_words(cluster["id"])
        target = next(w for w in cw if w["word"] == "terse")

        q = await generate_question(
            llm, populated_db,
            cluster=cluster,
            target_word_info=target,
            question_type="fill_blank",
        )

        assert q is not None
        assert q.correct_word == "terse"
        assert q.question_type == "fill_blank"
        assert q.llm_provider == "fake-llm"
        assert len(q.choices) == 4
        # Step 1 + Step 2 enrichment
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_choice_details_populated(self, populated_db):
        """Generated questions include enriched choice_details with all fields."""
        choices = ["terse", "concise", "pithy", "laconic"]
        step1 = json.dumps({
            "stem": "Her ___ reply surprised everyone.",
            "choices": choices,
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply surprised everyone.",
        })
        cluster = populated_db.get_random_cluster()
        cw = populated_db.get_cluster_words(cluster["id"])
        cw_map = {w["word"].lower(): w for w in cw}
        step2 = _make_enrichment_response(choices, cw_map)

        llm = FakeLLM(responses=[step1, step2])
        target = next(w for w in cw if w["word"] == "terse")

        q = await generate_question(
            llm, populated_db,
            cluster=cluster,
            target_word_info=target,
            question_type="fill_blank",
        )

        assert q is not None
        assert len(q.choice_details) == 4
        # Each detail should have all 5 fields from enrichment
        for detail in q.choice_details:
            assert "word" in detail
            assert "base_word" in detail
            assert "meaning" in detail
            assert "distinction" in detail
            assert "why" in detail
            assert detail["why"] != ""
        # Verify the details match cluster_words
        terse_detail = next(d for d in q.choice_details if d["word"] == "terse")
        assert "rude" in terse_detail["meaning"].lower() or "brief" in terse_detail["meaning"].lower()

    @pytest.mark.asyncio
    async def test_choice_details_with_conjugated_distractors(self, populated_db):
        """Conjugated distractors (e.g. 'concisely') still resolve via enrichment or fallback."""
        choices = ["terse", "concisely", "pithily", "laconically"]
        step1 = json.dumps({
            "stem": "Her ___ reply surprised everyone.",
            "choices": choices,
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply surprised everyone.",
        })
        cluster = populated_db.get_random_cluster()
        cw = populated_db.get_cluster_words(cluster["id"])
        cw_map = {w["word"].lower(): w for w in cw}
        step2 = _make_enrichment_response(choices, cw_map)

        llm = FakeLLM(responses=[step1, step2])
        target = next(w for w in cw if w["word"] == "terse")

        q = await generate_question(
            llm, populated_db,
            cluster=cluster,
            target_word_info=target,
            question_type="fill_blank",
        )

        assert q is not None
        assert len(q.choice_details) == 4
        # "concisely" should resolve to base_word "concise"
        concise_detail = next(d for d in q.choice_details if d["word"] == "concisely")
        assert concise_detail["base_word"] == "concise"
        assert concise_detail["meaning"] != "", (
            "Conjugated distractor 'concisely' should resolve to 'concise' meaning"
        )

    @pytest.mark.asyncio
    async def test_enrichment_fallback_on_failure(self, populated_db):
        """When enrichment LLM call fails, falls back to stem-based lookup."""
        choices = ["terse", "concise", "pithy", "laconic"]
        step1 = json.dumps({
            "stem": "Her ___ reply surprised everyone.",
            "choices": choices,
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply surprised everyone.",
        })
        # Only provide step 1 response — enrichment will get same JSON
        # which lacks choice_details, causing fallback after 2 retries
        llm = FakeLLM(responses=[step1])

        cluster = populated_db.get_random_cluster()
        cw = populated_db.get_cluster_words(cluster["id"])
        target = next(w for w in cw if w["word"] == "terse")

        q = await generate_question(
            llm, populated_db,
            cluster=cluster,
            target_word_info=target,
            question_type="fill_blank",
        )

        assert q is not None
        assert len(q.choice_details) == 4
        # Fallback should still populate from cluster_words
        for detail in q.choice_details:
            assert "word" in detail
            assert "base_word" in detail
            assert "meaning" in detail
            assert "distinction" in detail
            assert "why" in detail
        # But why will be empty (no LLM enrichment)
        for detail in q.choice_details:
            assert detail["why"] == ""
        # 1 step1 + 3 enrichment retries = 4 calls
        assert llm.call_count == 4

    @pytest.mark.asyncio
    async def test_invalid_llm_response_retries(self, populated_db):
        valid_step1 = json.dumps({
            "stem": "A ___ report wastes no words.",
            "choices": ["concise", "terse", "pithy", "laconic"],
            "correct_index": 0,
            "explanation": "Concise means compressed.",
            "context_sentence": "A concise report wastes no words.",
        })
        enrichment = _make_enrichment_response(
            ["concise", "terse", "pithy", "laconic"],
        )
        llm = FakeLLM(responses=[
            "Not valid JSON",
            "Still not valid",
            valid_step1,
            enrichment,
        ])

        cluster = populated_db.get_random_cluster()
        cw = populated_db.get_cluster_words(cluster["id"])
        target = next(w for w in cw if w["word"] == "concise")

        q = await generate_question(
            llm, populated_db,
            cluster=cluster,
            target_word_info=target,
        )

        assert q is not None
        assert q.correct_word == "concise"
        # 2 failed step1 + 1 successful step1 + 1 enrichment = 4
        assert llm.call_count == 4

    @pytest.mark.asyncio
    async def test_all_retries_fail(self, populated_db):
        llm = FakeLLM(responses=["garbage"])

        cluster = populated_db.get_random_cluster()
        cw = populated_db.get_cluster_words(cluster["id"])
        target = cw[0]

        q = await generate_question(
            llm, populated_db,
            cluster=cluster,
            target_word_info=target,
        )

        assert q is None

    @pytest.mark.asyncio
    async def test_no_clusters_returns_none(self, tmp_db):
        llm = FakeLLM(responses=[])

        q = await generate_question(llm, tmp_db)
        assert q is None
