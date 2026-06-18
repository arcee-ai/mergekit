# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Tests for ConditionalParameter filter matching in evaluate_setting."""

import pytest

from mergekit.config import _filter_matches, evaluate_setting, ConditionalParameter


class TestFilterMatches:
    """Unit tests for _filter_matches helper."""

    def test_plain_substring_backward_compat(self):
        """Original behavior: plain string matches as substring."""
        assert _filter_matches("attn", "model.layers.0.self_attn.q_proj.weight")
        assert _filter_matches("mlp", "model.layers.0.mlp.down_proj.weight")
        assert _filter_matches("embed", "model.embed_tokens.weight")

    def test_plain_substring_false(self):
        assert not _filter_matches("mlp", "model.layers.0.self_attn.q_proj.weight")
        assert not _filter_matches("attn", "model.layers.0.mlp.down_proj.weight")

    def test_linear_attn_substring_collision(self):
        """Plain 'attn' still matches 'linear_attn' (backward-compatible behavior).
        Users should use '*self_attn*' to avoid this."""
        # This is the known limitation documented in _filter_matches docstring.
        assert _filter_matches("attn", "model.layers.0.linear_attn.A_log")

    def test_glob_self_attn_excludes_linear_attn(self):
        """Glob pattern '*self_attn*' matches self_attn but NOT linear_attn."""
        assert _filter_matches("*self_attn*", "model.layers.0.self_attn.q_proj.weight")
        assert not _filter_matches("*self_attn*", "model.layers.0.linear_attn.A_log")
        assert not _filter_matches("*self_attn*", "model.layers.0.linear_attn.dt_bias")

    def test_glob_star_wildcard(self):
        assert _filter_matches("*.weight", "model.layers.0.mlp.down_proj.weight")
        assert not _filter_matches("*.weight", "model.layers.0.mlp.down_proj.bias")

    def test_glob_question_mark(self):
        assert _filter_matches("layer?.mlp", "model.layer0.mlp.weight")
        assert not _filter_matches("layer?.mlp", "model.layer10.mlp.weight")

    def test_empty_filter(self):
        assert not _filter_matches("", "model.layers.0.self_attn.q_proj.weight")

    def test_empty_tensor_name(self):
        assert not _filter_matches("attn", "")


class TestEvaluateSetting:
    """Integration tests for evaluate_setting with conditional parameter lists."""

    def _make_setting(self, *filter_value_pairs, default=1.0):
        conds = [
            ConditionalParameter(filter=f, value=v) for f, v in filter_value_pairs
        ]
        conds.append(ConditionalParameter(filter=None, value=default))
        return conds

    def test_glob_filter_self_attn_only(self):
        """Verify that '*self_attn*' filter selects the correct density for
        self_attn weights while leaving linear_attn weights at the default."""
        setting = self._make_setting(("*self_attn*", 0.03), default=1.0)

        q_proj = "model.layers.0.self_attn.q_proj.weight"
        a_log = "model.layers.0.linear_attn.A_log"

        assert evaluate_setting(q_proj, setting) == pytest.approx(0.03)
        assert evaluate_setting(a_log, setting) == pytest.approx(1.0)

    def test_plain_attn_matches_both(self):
        """Plain 'attn' substring matches both self_attn and linear_attn
        (documented backward-compatible behavior)."""
        setting = self._make_setting(("attn", 0.03), default=1.0)

        q_proj = "model.layers.0.self_attn.q_proj.weight"
        a_log = "model.layers.0.linear_attn.A_log"

        assert evaluate_setting(q_proj, setting) == pytest.approx(0.03)
        # Both match under plain substring — callers should use '*self_attn*' to avoid this
        assert evaluate_setting(a_log, setting) == pytest.approx(0.03)

    def test_wildcard_star_matches_all(self):
        setting = self._make_setting(("*", 0.5), default=1.0)
        assert evaluate_setting("anything.weight", setting) == pytest.approx(0.5)

    def test_none_filter_matches_all(self):
        conds = [ConditionalParameter(filter=None, value=0.7)]
        assert evaluate_setting("anything.weight", conds) == pytest.approx(0.7)

    def test_first_match_wins(self):
        """evaluate_setting returns value for first matching conditional."""
        setting = self._make_setting(
            ("*self_attn*", 0.03),
            ("mlp", 0.05),
            default=1.0,
        )
        assert evaluate_setting("model.layers.0.self_attn.q_proj.weight", setting) == pytest.approx(0.03)
        assert evaluate_setting("model.layers.0.mlp.down_proj.weight", setting) == pytest.approx(0.05)
        assert evaluate_setting("model.embed_tokens.weight", setting) == pytest.approx(1.0)
