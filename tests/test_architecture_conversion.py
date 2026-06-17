import torch
import transformers

from mergekit.architecture import arch_info_for_config
from mergekit.architecture.conversion import convert_checkpoint_tensors


def test_qwen3_moe_expert_weights_convert_to_v5_layout():
    sources = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.full((2, 3), 1.0),
        "model.layers.0.mlp.experts.1.gate_proj.weight": torch.full((2, 3), 2.0),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((2, 3), 3.0),
        "model.layers.0.mlp.experts.1.up_proj.weight": torch.full((2, 3), 4.0),
    }

    converted = convert_checkpoint_tensors(
        "qwen3_moe",
        sources,
        "model.layers.0.mlp.experts.gate_up_proj",
    )

    assert converted.shape == (2, 4, 3)
    assert converted[0, 0, 0] == 1
    assert converted[1, 0, 0] == 2
    assert converted[0, 2, 0] == 3
    assert converted[1, 2, 0] == 4


def test_mixtral_old_checkpoint_names_convert_to_v5_layout():
    sources = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.full((2, 3), 1.0),
        "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.full((2, 3), 2.0),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.full((2, 3), 3.0),
        "model.layers.0.block_sparse_moe.experts.1.w3.weight": torch.full((2, 3), 4.0),
    }

    converted = convert_checkpoint_tensors(
        "mixtral",
        sources,
        "model.layers.0.mlp.experts.gate_up_proj",
    )

    assert converted.shape == (2, 4, 3)
    assert converted[0, 0, 0] == 1
    assert converted[1, 0, 0] == 2
    assert converted[0, 2, 0] == 3
    assert converted[1, 2, 0] == 4


def test_mixtral_architecture_uses_json_v5_layout():
    cfg = transformers.MixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
    )
    cfg.architectures = ["MixtralForCausalLM"]

    arch = arch_info_for_config(cfg)
    names = {w.name for w in arch.all_weights(cfg)}

    assert "model.layers.0.mlp.experts.gate_up_proj" in names
    assert "model.layers.0.mlp.experts.down_proj" in names
    assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" not in names


def test_qwen3_moe_architecture_uses_json_v5_layout():
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
    )
    cfg.architectures = ["Qwen3MoeForCausalLM"]

    arch = arch_info_for_config(cfg)
    names = {w.name for w in arch.all_weights(cfg)}

    assert "model.layers.0.mlp.experts.gate_up_proj" in names
    assert "model.layers.0.mlp.experts.down_proj" in names
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" not in names
