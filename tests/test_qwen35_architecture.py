import tempfile

import pytest

from mergekit.architecture import arch_info_for_config
from mergekit.common import set_config_value
from mergekit.config import InputModelDefinition, MergeConfiguration
from tests.common import run_and_check_merge

qwen35 = pytest.importorskip("transformers.models.qwen3_5")
qwen35_moe = pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)


def _dense_config():
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        tie_word_embeddings=True,
    )
    vision = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text,
        vision_config=vision,
        tie_word_embeddings=True,
    )


def _moe_config():
    text = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        tie_word_embeddings=True,
    )
    vision = Qwen3_5MoeVisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    return Qwen3_5MoeConfig(
        architectures=["Qwen3_5MoeForConditionalGeneration"],
        text_config=text,
        vision_config=vision,
        tie_word_embeddings=True,
    )


def _dense_text_config():
    config = _dense_config().text_config
    config.architectures = ["Qwen3_5ForCausalLM"]
    return config


def _moe_text_config():
    config = _moe_config().text_config
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    return config


def _save_model(model_cls, config, path):
    model = model_cls(config)
    model.save_pretrained(path, safe_serialization=True)
    return str(path), set(model.state_dict().keys())


def _arch_names(config):
    arch = arch_info_for_config(config)
    return {weight.name for weight in arch.all_weights(config)}


def test_qwen35_dense_architecture_covers_transformers_keys():
    config = _dense_config()
    model = Qwen3_5ForConditionalGeneration(config)
    state_keys = set(model.state_dict().keys())
    arch_keys = _arch_names(config)

    assert state_keys <= arch_keys
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" in arch_keys
    assert "model.language_model.layers.3.self_attn.q_proj.weight" in arch_keys
    assert "mtp.layers.0.mlp.gate_proj.weight" in arch_keys
    assert "mtp.fc.weight" in arch_keys


@pytest.mark.parametrize(
    ("config", "model_cls"),
    [
        (_dense_config(), Qwen3_5ForConditionalGeneration),
        (_moe_config(), Qwen3_5MoeForConditionalGeneration),
    ],
)
def test_qwen35_full_attention_bias_covers_output_projection_bias(
    config, model_cls
):
    config.text_config.attention_bias = True
    model = model_cls(config)
    state_keys = set(model.state_dict().keys())
    arch_keys = _arch_names(config)

    assert "model.language_model.layers.3.self_attn.o_proj.bias" in state_keys
    assert state_keys <= arch_keys


def test_qwen35_moe_architecture_covers_transformers_keys_and_mtp_experts():
    config = _moe_config()
    model = Qwen3_5MoeForConditionalGeneration(config)
    state_keys = set(model.state_dict().keys())
    arch_keys = _arch_names(config)

    assert state_keys <= arch_keys
    assert "model.language_model.layers.0.mlp.experts.gate_up_proj" in arch_keys
    assert "model.language_model.layers.0.mlp.shared_expert_gate.weight" in arch_keys
    assert "model.language_model.layers.3.self_attn.q_proj.weight" in arch_keys
    assert "mtp.layers.0.mlp.experts.3.down_proj.weight" in arch_keys
    assert "mtp.layers.0.mlp.shared_expert_gate.weight" in arch_keys


@pytest.mark.parametrize(
    ("config", "model_cls"),
    [
        (_dense_text_config(), Qwen3_5ForCausalLM),
        (_moe_text_config(), Qwen3_5MoeForCausalLM),
    ],
)
def test_qwen35_text_only_architecture_uses_top_level_mtp_config_key(
    config, model_cls
):
    model = model_cls(config)
    arch = arch_info_for_config(config)
    arch_keys = {weight.name for weight in arch.all_weights(config)}
    mtp_num_layers_key = arch.modules["mtp"].architecture.num_layers_config_key()

    assert set(model.state_dict().keys()) <= arch_keys
    assert mtp_num_layers_key == "mtp_num_hidden_layers"
    set_config_value(config, mtp_num_layers_key, 0)
    assert config.mtp_num_hidden_layers == 0


def test_qwen35_dense_passthrough_merge():
    with tempfile.TemporaryDirectory() as a:
        model_a, _ = _save_model(Qwen3_5ForConditionalGeneration, _dense_config(), a)
        config = MergeConfiguration(
            merge_method="passthrough",
            models=[InputModelDefinition(model=model_a)],
            dtype="bfloat16",
        )
        run_and_check_merge(config)


def test_qwen35_moe_linear_merge():
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        model_a, _ = _save_model(Qwen3_5MoeForConditionalGeneration, _moe_config(), a)
        model_b, _ = _save_model(Qwen3_5MoeForConditionalGeneration, _moe_config(), b)
        config = MergeConfiguration(
            merge_method="linear",
            models=[
                InputModelDefinition(model=model_a, parameters={"weight": 0.5}),
                InputModelDefinition(model=model_b, parameters={"weight": 0.5}),
            ],
            dtype="bfloat16",
        )
        run_and_check_merge(config)
