"""Tests for prompt templates and formatting."""
from __future__ import annotations

from vocab_trainer.prompts import (
    BEST_FIT_PROMPT,
    DISTINCTION_PROMPT,
    FILL_BLANK_PROMPT,
    format_cluster_info,
    format_enrichment,
)


class TestFormatClusterInfo:
    def test_basic(self):
        entries = [
            {"word": "terse", "meaning": "brief rudely", "distinction": "personality trait"},
            {"word": "concise", "meaning": "few words well", "distinction": "compression"},
        ]
        result = format_cluster_info(entries)
        assert "**terse**" in result
        assert "**concise**" in result
        assert "personality trait" in result
        assert "compression" in result

    def test_empty(self):
        assert format_cluster_info([]) == ""


class TestFormatEnrichment:
    def test_with_words(self):
        extra = [
            {"word": "ebullient", "definition": "full of enthusiasm"},
            {"word": "sanguine", "definition": "optimistic"},
        ]
        result = format_enrichment(extra)
        assert "ebullient" in result
        assert "sanguine" in result
        assert "palette" in result.lower()

    def test_empty(self):
        assert format_enrichment([]) == ""

    def test_no_definitions_in_output(self):
        extra = [{"word": "test", "definition": "some long definition here"}]
        result = format_enrichment(extra)
        # Ambient style: only word names, no definitions
        assert "definition" not in result
        assert "a" * 61 not in result


class TestPromptTemplates:
    def _format_kwargs(self):
        return {
            "cluster_title": "Being Brief",
            "cluster_info": "- **terse**: brief — personality",
            "target_word": "terse",
            "target_meaning": "brief rudely",
            "target_distinction": "personality trait",
            "enrichment_section": "",
        }

    def test_fill_blank_prompt_formats(self):
        result = FILL_BLANK_PROMPT.format(**self._format_kwargs())
        assert "Being Brief" in result
        assert "terse" in result
        assert "Dr. Voss" in result

    def test_best_fit_prompt_formats(self):
        result = BEST_FIT_PROMPT.format(**self._format_kwargs())
        assert "Dr. Voss" in result
        assert "terse" in result

    def test_distinction_prompt_formats(self):
        result = DISTINCTION_PROMPT.format(**self._format_kwargs())
        assert "distinction" in result.lower()
        assert "terse" in result

    def test_all_placeholders_filled(self):
        """No unfilled {placeholders} should remain."""
        for template in [FILL_BLANK_PROMPT, BEST_FIT_PROMPT, DISTINCTION_PROMPT]:
            result = template.format(**self._format_kwargs())
            # Check no single-brace placeholders remain (double braces {{ }} are literal)
            import re
            # After formatting, only literal braces from JSON example should exist
            assert "{cluster_title}" not in result
            assert "{target_word}" not in result
