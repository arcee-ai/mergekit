import math

import pytest
from transformers import LlamaConfig

from mergekit.moe.config import MoEMergeConfig, is_bad_config
from mergekit.moe.qwen import QwenMoE
from mergekit.moe.router import get_gate_params


@pytest.fixture
def moe_config(tmp_path):
    models = []
    for name in ("base", "expert_a", "expert_b", "shared"):
        path = tmp_path / name
        LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
        ).save_pretrained(path)
        models.append(str(path))
    return MoEMergeConfig.model_validate(
        {
            "base_model": models[0],
            "experts": [{"source_model": model} for model in models[1:3]],
            "shared_experts": [{"source_model": models[3]}],
            "gate_mode": "uniform_random",
        }
    )


@pytest.mark.parametrize("mode", ["random", "uniform_random"])
def test_promptless_random_gates(moe_config, mode):
    moe_config.gate_mode = mode
    assert not is_bad_config(moe_config)

    experts = moe_config.experts + moe_config.shared_experts
    gates = get_gate_params(moe_config.base_model, None, experts, mode=mode)
    assert gates.shape == (2, 3, 16)
    if mode == "uniform_random":
        scale = math.sqrt(1.0 / 16)
        assert (gates >= -scale).all()
        assert (gates <= scale).all()


@pytest.mark.parametrize("mode", ["random", "uniform_random"])
def test_qwen_promptless_random_gates(moe_config, mode):
    moe_config.gate_mode = mode
    assert QwenMoE().supports_config(moe_config)


@pytest.mark.parametrize("mode", ["random", "uniform_random"])
@pytest.mark.parametrize("experts_per_token", [0, 3])
def test_random_gates_retain_expert_count_checks(moe_config, mode, experts_per_token):
    moe_config.gate_mode = mode
    moe_config.experts_per_token = experts_per_token
    assert is_bad_config(moe_config)


def test_uniform_random_retains_same_model_check(moe_config):
    moe_config.experts[1].source_model = moe_config.experts[0].source_model
    assert is_bad_config(moe_config)
    assert not is_bad_config(moe_config, allow_all_same=True)


def test_prompt_based_modes_still_require_prompts(moe_config):
    moe_config.gate_mode = "hidden"
    assert is_bad_config(moe_config)
    assert not QwenMoE().supports_config(moe_config)


@pytest.mark.parametrize("mode", ["random", "uniform_random"])
def test_qwen_still_requires_one_shared_expert(moe_config, mode):
    moe_config.gate_mode = mode
    moe_config.shared_experts = []
    assert not QwenMoE().supports_config(moe_config)
