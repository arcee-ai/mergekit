import pytest

try:
    from mergekit.architecture.canonical import (
        LlamaFamilyDetector,
        is_canonical,
        layer_key,
    )
    import mergekit.architecture.canonical as _canon  # noqa: F401
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
    _canon = _mod


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


def test_bert_detect_and_map():
    sd = {
        "bert.embeddings.word_embeddings.weight": object(),
        "bert.encoder.layer.0.attention.self.query.weight": object(),
        "bert.encoder.layer.0.attention.output.LayerNorm.weight": object(),
        "bert.encoder.layer.1.output.dense.bias": object(),
    }
    assert _canon.BertFamilyDetector().detect(sd)
    out = _canon.BertFamilyDetector().canonicalize(sd)
    assert out["layer_0.attn_q.weight"] is sd[
        "bert.encoder.layer.0.attention.self.query.weight"
    ]
    assert out["layer_0.post_attn_norm.weight"] is sd[
        "bert.encoder.layer.0.attention.output.LayerNorm.weight"
    ]
    assert out["layer_1.ffn_down.bias"] is sd["bert.encoder.layer.1.output.dense.bias"]
    assert all(is_canonical(k) for k in out)


def test_opt_detect_and_map():
    sd = {
        "model.decoder.embed_tokens.weight": object(),
        "model.decoder.layers.0.self_attn.q_proj.weight": object(),
        "model.decoder.layers.2.fc1.weight": object(),
        "model.decoder.final_layer_norm.weight": object(),
    }
    assert _canon.OPTDetector().detect(sd)
    out = _canon.OPTDetector().canonicalize(sd)
    assert out["layer_0.attn_q.weight"] is sd[
        "model.decoder.layers.0.self_attn.q_proj.weight"
    ]
    assert out["layer_2.ffn_up.weight"] is sd["model.decoder.layers.2.fc1.weight"]
    assert out["final_norm.weight"] is sd["model.decoder.final_layer_norm.weight"]

def test_opt_maps_ffn_norm_bias():
    # regression: Bugbot finding — per-layer FFN LayerNorm bias was dropped
    sd = {"model.decoder.layers.0.final_layer_norm.bias": object()}
    out = _canon.OPTDetector().canonicalize(sd)
    assert out["layer_0.pre_ffn_norm.bias"] is sd[
        "model.decoder.layers.0.final_layer_norm.bias"
    ]
    assert all(is_canonical(k) for k in out)


def test_cross_detector_exclusivity():
    llama = {"model.layers.0.self_attn.q_proj.weight": object()}
    assert not _canon.OPTDetector().detect(llama)
    assert not _canon.BertFamilyDetector().detect(llama)
