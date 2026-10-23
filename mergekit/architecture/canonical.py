# Copyright 2026 Ryan Gillespie (mgillr). Apache-2.0 (mergekit licence).
"""Canonical parameter-name mapping for cross-architecture merging.

First increment of the cross-architecture proposal (#708): a shared
role-key space so that parameters from different architecture families
can be addressed uniformly before any merge method runs.

    model.layers.0.self_attn.q_proj.weight   ->  layer_0.attn_q.weight
    model.layers.11.mlp.up_proj.weight       ->  layer_11.ffn_up.weight
    model.embed_tokens.weight                ->  embed_tokens.weight
    model.norm.weight                        ->  final_norm.weight

This module is deliberately dependency-free: it maps names to names and
passes values through by reference. The Llama-family layout (Llama,
Qwen, Mistral, Gemma, TinyLlama and friends) is the highest-traffic
case and ships first; the remaining families from the source system
(gpt2, bert, neox, opt, t5, phi) follow in the same pattern once the
module's home is settled. The shape bridge (pad/truncate + optional
per-key alignment) is a separate increment.

The detector set is exercised end-to-end by the cross-family merge
pipeline that produced the published `Optitransfer/*converge*`
collectives (9 models across 4 architecture families, merged into one
set of weights).
"""
from __future__ import annotations

import re
from typing import Any, Mapping

# Canonical roles, one schema for every family:
DECODER_ROLES = (
    "pre_attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "pre_ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
    "post_attn_norm",
    "post_ffn_norm",
)

GLOBAL_KEYS = (
    "embed_tokens.weight",
    "embed_positions.weight",
    "final_norm.weight",
    "final_norm.bias",
    "head_lm.weight",
    "head_lm.bias",
)

_LAYER_RE = re.compile(r"^layer_(\d+)\.([a-z_]+)\.(weight|bias)$")


def layer_key(layer_idx: int, role: str, param: str = "weight") -> str:
    """Canonical key for a per-layer role: ``layer_3.attn_q.weight``."""
    return f"layer_{layer_idx}.{role}.{param}"


def global_key(name: str) -> str:
    """Canonical key for a non-layer parameter."""
    return name


def is_canonical(key: str) -> bool:
    m = _LAYER_RE.match(key)
    if m and m.group(2) in DECODER_ROLES:
        return True
    return key in GLOBAL_KEYS


class LlamaFamilyDetector:
    """Llama-layout checkpoints: Llama, Qwen, Mistral, Gemma, TinyLlama.

    Detects the ``model.layers.N.self_attn.*`` layout (distinguished
    from OPT's ``model.decoder.layers.``) and rewrites it into the
    canonical schema. Pure name remapping; values pass through by
    reference and unmapped keys (rotary buffers, position ids, optimizer
    state) are dropped.
    """

    name = "llama_family"
    family = "llama"

    _layer_re = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

    _TAIL = {
        "input_layernorm.weight": ("pre_attn_norm", "weight"),
        "post_attention_layernorm.weight": ("pre_ffn_norm", "weight"),
        "self_attn.q_proj.weight": ("attn_q", "weight"),
        "self_attn.q_proj.bias": ("attn_q", "bias"),
        "self_attn.k_proj.weight": ("attn_k", "weight"),
        "self_attn.k_proj.bias": ("attn_k", "bias"),
        "self_attn.v_proj.weight": ("attn_v", "weight"),
        "self_attn.v_proj.bias": ("attn_v", "bias"),
        "self_attn.o_proj.weight": ("attn_o", "weight"),
        "self_attn.o_proj.bias": ("attn_o", "bias"),
        "mlp.gate_proj.weight": ("ffn_gate", "weight"),
        "mlp.up_proj.weight": ("ffn_up", "weight"),
        "mlp.down_proj.weight": ("ffn_down", "weight"),
    }

    _GLOBALS = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "final_norm.weight",
        "lm_head.weight": "head_lm.weight",
    }

    def detect(self, state_dict: Mapping[str, Any]) -> bool:
        if any(k.startswith("model.decoder.") for k in state_dict):
            return False  # OPT layout
        return any(k.startswith("model.layers.") for k in state_dict) and any(
            ".self_attn.q_proj." in k for k in state_dict
        )

    def canonicalize(
        self, state_dict: Mapping[str, Any]
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in state_dict.items():
            g = self._GLOBALS.get(key)
            if g is not None:
                out[g] = value
                continue
            m = self._layer_re.match(key)
            if m is None:
                continue
            tail = self._TAIL.get(m.group(2))
            if tail is not None:
                out[layer_key(int(m.group(1)), tail[0], tail[1])] = value
        return out

    def decanonicalize(
        self, canonical: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Inverse of :meth:`canonicalize` (round-trippable)."""
        rev_tail = {v: k for k, v in self._TAIL.items()}
        rev_glob = {v: k for k, v in self._GLOBALS.items()}
        out: dict[str, Any] = {}
        for key, value in canonical.items():
            g = rev_glob.get(key)
            if g is not None:
                out[g] = value
                continue
            m = _LAYER_RE.match(key)
            if m and (m.group(2), m.group(3)) in rev_tail:
                out[
                    f"model.layers.{m.group(1)}.{rev_tail[(m.group(2), m.group(3))]}"
                ] = value
        return out


class BertFamilyDetector:
    """BERT / RoBERTa encoder layout (post-norm; DistilBERT excluded).

    Post-norm LayerNorms map onto the post_attn_norm / post_ffn_norm
    slots so comparison within a BERT set is consistent.
    """

    name = "bert"
    family = "bert"

    _layer_re = re.compile(r"^(?:bert|roberta)\.encoder\.layer\.(\d+)\.(.+)$")

    _TAIL = {
        "attention.self.query.weight": ("attn_q", "weight"),
        "attention.self.query.bias": ("attn_q", "bias"),
        "attention.self.key.weight": ("attn_k", "weight"),
        "attention.self.key.bias": ("attn_k", "bias"),
        "attention.self.value.weight": ("attn_v", "weight"),
        "attention.self.value.bias": ("attn_v", "bias"),
        "attention.output.dense.weight": ("attn_o", "weight"),
        "attention.output.dense.bias": ("attn_o", "bias"),
        "attention.output.LayerNorm.weight": ("post_attn_norm", "weight"),
        "attention.output.LayerNorm.bias": ("post_attn_norm", "bias"),
        "intermediate.dense.weight": ("ffn_up", "weight"),
        "intermediate.dense.bias": ("ffn_up", "bias"),
        "output.dense.weight": ("ffn_down", "weight"),
        "output.dense.bias": ("ffn_down", "bias"),
        "output.LayerNorm.weight": ("post_ffn_norm", "weight"),
        "output.LayerNorm.bias": ("post_ffn_norm", "bias"),
    }

    def detect(self, state_dict: Mapping[str, Any]) -> bool:
        return any(
            k.startswith(("bert.", "roberta.")) for k in state_dict
        ) and any(".attention.self.query." in k for k in state_dict)

    def canonicalize(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in state_dict.items():
            if key.endswith("embeddings.word_embeddings.weight"):
                out["embed_tokens.weight"] = value
                continue
            m = self._layer_re.match(key)
            if m is None:
                continue
            tail = self._TAIL.get(m.group(2))
            if tail is not None:
                out[layer_key(int(m.group(1)), tail[0], tail[1])] = value
        return out


class OPTDetector:
    """OPT decoder layout: ``model.decoder.layers.N.*``."""

    name = "opt"
    family = "opt"

    _layer_re = re.compile(r"^model\.decoder\.layers\.(\d+)\.(.+)$")

    _TAIL = {
        "self_attn_layer_norm.weight": ("pre_attn_norm", "weight"),
        "self_attn_layer_norm.bias": ("pre_attn_norm", "bias"),
        "self_attn.q_proj.weight": ("attn_q", "weight"),
        "self_attn.q_proj.bias": ("attn_q", "bias"),
        "self_attn.k_proj.weight": ("attn_k", "weight"),
        "self_attn.k_proj.bias": ("attn_k", "bias"),
        "self_attn.v_proj.weight": ("attn_v", "weight"),
        "self_attn.v_proj.bias": ("attn_v", "bias"),
        "self_attn.out_proj.weight": ("attn_o", "weight"),
        "self_attn.out_proj.bias": ("attn_o", "bias"),
        "final_layer_norm.weight": ("pre_ffn_norm", "weight"),  # per-layer fc norm name in OPT
        "final_layer_norm.bias": ("pre_ffn_norm", "bias"),
        "fc1.weight": ("ffn_up", "weight"),
        "fc1.bias": ("ffn_up", "bias"),
        "fc2.weight": ("ffn_down", "weight"),
        "fc2.bias": ("ffn_down", "bias"),
    }

    _GLOBALS = {
        "model.decoder.embed_tokens.weight": "embed_tokens.weight",
        "model.decoder.embed_positions.weight": "embed_positions.weight",
        "model.decoder.final_layer_norm.weight": "final_norm.weight",
        "model.decoder.final_layer_norm.bias": "final_norm.bias",
        "lm_head.weight": "head_lm.weight",
    }

    def detect(self, state_dict: Mapping[str, Any]) -> bool:
        return any(
            k.startswith("model.decoder.layers.") for k in state_dict
        ) and any(".self_attn.q_proj." in k for k in state_dict)

    def canonicalize(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in state_dict.items():
            g = self._GLOBALS.get(key)
            if g is not None:
                out[g] = value
                continue
            m = self._layer_re.match(key)
            if m is None:
                continue
            tail = self._TAIL.get(m.group(2))
            if tail is not None:
                out[layer_key(int(m.group(1)), tail[0], tail[1])] = value
        return out
