import pytest

try:
    from mergekit.architecture.canonical import (
        LlamaFamilyDetector,
        is_canonical,
        layer_key,
    )
except ImportError:  # torch not installed: load the module standalone
    import importlib.util
    from pathlib import Path

    _spec = importlib.util.spec_from_file_location(
        "canonical",
        Path(__file__).parents[1] / "mergekit" / "architecture" / "canonical.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    LlamaFamilyDetector = _mod.LlamaFamilyDetector
    is_canonical = _mod.is_canonical
    layer_key = _mod.layer_key


@pytest.fixture
def llama_sd():
    return {
        "model.embed_tokens.weight": object(),
        "model.layers.0.self_attn.q_proj.weight": object(),
        "model.layers.0.self_attn.k_proj.bias": object(),
        "model.layers.0.input_layernorm.weight": object(),
        "model.layers.0.mlp.up_proj.weight": object(),
        "model.layers.11.self_attn.o_proj.weight": object(),
        "model.layers.11.post_attention_layernorm.weight": object(),
        "model.norm.weight": object(),
        "lm_head.weight": object(),
        "model.layers.0.self_attn.rotary_emb.inv_freq": object(),  # dropped
    }


def test_detect_accepts_llama_layout(llama_sd):
    assert LlamaFamilyDetector().detect(llama_sd)


def test_detect_rejects_opt_layout():
    assert not LlamaFamilyDetector().detect(
        {"model.decoder.layers.0.self_attn.q_proj.weight": object()}
    )


def test_detect_rejects_gpt2_layout():
    assert not LlamaFamilyDetector().detect(
        {"transformer.h.0.attn.c_attn.weight": object()}
    )


def test_canonicalize_maps_roles(llama_sd):
    out = LlamaFamilyDetector().canonicalize(llama_sd)
    assert out["layer_0.attn_q.weight"] is llama_sd[
        "model.layers.0.self_attn.q_proj.weight"
    ]
    assert out["layer_11.attn_o.weight"] is llama_sd[
        "model.layers.11.self_attn.o_proj.weight"
    ]
    assert out["embed_tokens.weight"] is llama_sd["model.embed_tokens.weight"]
    assert out["final_norm.weight"] is llama_sd["model.norm.weight"]


def test_canonicalize_drops_buffers(llama_sd):
    out = LlamaFamilyDetector().canonicalize(llama_sd)
    assert not any("inv_freq" in k for k in out)
    assert all(is_canonical(k) for k in out)


def test_round_trip(llama_sd):
    det = LlamaFamilyDetector()
    canon = det.canonicalize(llama_sd)
    back = det.decanonicalize(canon)
    # exact key-set and value identity on the mappable subset;
    # dropped buffers (inv_freq) are gone by contract
    expected = {k: v for k, v in llama_sd.items() if "inv_freq" not in k}
    assert back == expected


def test_layer_key_shape():
    assert layer_key(3, "ffn_down") == "layer_3.ffn_down.weight"
    assert is_canonical("layer_3.ffn_down.weight")
    assert not is_canonical("layer_3.not_a_role.weight")
    assert not is_canonical("model.layers.3.mlp.down_proj.weight")
