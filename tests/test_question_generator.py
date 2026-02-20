"""Tests for question generation (JSON extraction, validation, LLM orchestration)."""
from __future__ import annotations

import json

import pytest

from vocab_trainer.question_generator import (
    _extract_json,
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
        assert _validate_question(self._valid_data(), "terse") is True

    def test_missing_field(self):
        d = self._valid_data()
        del d["explanation"]
        assert _validate_question(d, "terse") is False

    def test_wrong_choice_count(self):
        d = self._valid_data()
        d["choices"] = ["terse", "concise", "pithy"]
        assert _validate_question(d, "terse") is False

    def test_index_out_of_range(self):
        d = self._valid_data()
        d["correct_index"] = 5
        assert _validate_question(d, "terse") is False

    def test_negative_index(self):
        d = self._valid_data()
        d["correct_index"] = -1
        assert _validate_question(d, "terse") is False

    def test_correct_word_mismatch(self):
        d = self._valid_data()
        assert _validate_question(d, "concise") is False

    def test_duplicate_choices(self):
        d = self._valid_data()
        d["choices"] = ["terse", "terse", "pithy", "laconic"]
        assert _validate_question(d, "terse") is False

    def test_case_insensitive_word_match(self):
        d = self._valid_data()
        d["choices"][0] = "Terse"
        assert _validate_question(d, "terse") is True

    def test_non_integer_index(self):
        d = self._valid_data()
        d["correct_index"] = "zero"
        assert _validate_question(d, "terse") is False


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


class TestGenerateQuestion:
    @pytest.mark.asyncio
    async def test_with_mock_llm(self, populated_db):
        llm_response = json.dumps({
            "stem": "Her ___ reply surprised everyone.",
            "choices": ["terse", "concise", "pithy", "laconic"],
            "correct_index": 0,
            "explanation": "Terse implies rudeness.",
            "context_sentence": "Her terse reply surprised everyone.",
        })

        llm = FakeLLM(responses=[llm_response])

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
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_llm_response_retries(self, populated_db):
        llm = FakeLLM(responses=[
            "Not valid JSON",
            "Still not valid",
            json.dumps({
                "stem": "A ___ report wastes no words.",
                "choices": ["concise", "terse", "pithy", "laconic"],
                "correct_index": 0,
                "explanation": "Concise means compressed.",
                "context_sentence": "A concise report wastes no words.",
            }),
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
        assert llm.call_count == 3

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
