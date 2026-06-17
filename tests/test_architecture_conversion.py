import torch
import transformers
from safetensors.torch import save_file

from mergekit.architecture import arch_info_for_config
from mergekit.architecture.auto import infer_architecture_info
from mergekit.architecture.conversion import (
    can_convert_checkpoint_keys,
    convert_checkpoint_tensors,
)
from mergekit.common import ModelReference
from mergekit.options import MergeOptions


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

    torch.testing.assert_close(
        converted,
        torch.stack(
            [
                torch.cat(
                    [
                        sources["model.layers.0.mlp.experts.0.gate_proj.weight"],
                        sources["model.layers.0.mlp.experts.0.up_proj.weight"],
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        sources["model.layers.0.mlp.experts.1.gate_proj.weight"],
                        sources["model.layers.0.mlp.experts.1.up_proj.weight"],
                    ],
                    dim=0,
                ),
            ],
            dim=0,
        ),
    )


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

    torch.testing.assert_close(
        converted,
        torch.stack(
            [
                torch.cat(
                    [
                        sources["model.layers.0.block_sparse_moe.experts.0.w1.weight"],
                        sources["model.layers.0.block_sparse_moe.experts.0.w3.weight"],
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        sources["model.layers.0.block_sparse_moe.experts.1.w1.weight"],
                        sources["model.layers.0.block_sparse_moe.experts.1.w3.weight"],
                    ],
                    dim=0,
                ),
            ],
            dim=0,
        ),
    )


def test_mixtral_writer_keys_convert_to_v5_layout():
    sources = {
        "model.layers.0.mlp.experts.0.w1.weight": torch.full((2, 3), 1.0),
        "model.layers.0.mlp.experts.1.w1.weight": torch.full((2, 3), 2.0),
        "model.layers.0.mlp.experts.0.w3.weight": torch.full((2, 3), 3.0),
        "model.layers.0.mlp.experts.1.w3.weight": torch.full((2, 3), 4.0),
    }

    converted = convert_checkpoint_tensors(
        "mixtral",
        sources,
        "model.layers.0.mlp.experts.gate_up_proj",
    )

    torch.testing.assert_close(
        converted,
        torch.stack(
            [
                torch.cat(
                    [
                        sources["model.layers.0.mlp.experts.0.w1.weight"],
                        sources["model.layers.0.mlp.experts.0.w3.weight"],
                    ],
                    dim=0,
                ),
                torch.cat(
                    [
                        sources["model.layers.0.mlp.experts.1.w1.weight"],
                        sources["model.layers.0.mlp.experts.1.w3.weight"],
                    ],
                    dim=0,
                ),
            ],
            dim=0,
        ),
    )


def test_key_conversion_requires_complete_single_pattern_groups():
    assert not can_convert_checkpoint_keys(
        "mixtral",
        {"model.layers.0.block_sparse_moe.experts.0.w2.weight"},
        "model.layers.0.mlp.experts.down_proj",
    )


def test_key_conversion_requires_matching_wildcard_groups():
    assert not can_convert_checkpoint_keys(
        "mixtral",
        {
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w1.weight",
            "model.layers.0.block_sparse_moe.experts.1.w3.weight",
            "model.layers.0.block_sparse_moe.experts.2.w3.weight",
        },
        "model.layers.0.mlp.experts.gate_up_proj",
    )


def test_key_conversion_requires_complete_multi_pattern_groups():
    assert not can_convert_checkpoint_keys(
        "mixtral",
        {
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        },
        "model.layers.0.mlp.experts.gate_up_proj",
    )


def test_key_conversion_supports_later_one_to_many_targets():
    assert can_convert_checkpoint_keys(
        "hrm_text",
        {"layers.0.mlp.gate_up_proj.weight"},
        "layers.0.mlp.up_proj.weight",
    )


def test_tensor_conversion_supports_later_one_to_many_targets():
    converted = convert_checkpoint_tensors(
        "hrm_text",
        {"layers.0.mlp.gate_up_proj.weight": torch.arange(12).reshape(4, 3)},
        "layers.0.mlp.up_proj.weight",
    )

    torch.testing.assert_close(converted, torch.arange(6, 12).reshape(2, 3))


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


def test_auto_inference_uses_transformers_v5_layout_with_old_checkpoint_keys(
    tmp_path,
):
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
    cfg.architectures = ["UnknownMixtralForCausalLM"]
    cfg.save_pretrained(tmp_path)
    save_file(
        {
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.1.w3.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.0.w2.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.1.w2.weight": torch.zeros(2, 3),
        },
        tmp_path / "model.safetensors",
    )
    model_ref = ModelReference.parse(str(tmp_path))

    arch = infer_architecture_info((model_ref,), model_ref, MergeOptions())
    weights = {w.name: w for w in arch.all_weights(cfg)}

    assert "model.layers.0.mlp.experts.gate_up_proj" in weights
    assert "model.layers.0.mlp.experts.down_proj" in weights
    assert not weights["model.layers.0.mlp.experts.gate_up_proj"].optional
    assert (
        "model.layers.${layer_index}.block_sparse_moe.experts.0.w1.weight"
        not in weights
    )


def test_auto_inference_marks_template_optional_if_missing_in_any_layer(tmp_path):
    cfg = transformers.MixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
    )
    cfg.architectures = ["UnknownMixtralForCausalLM"]
    cfg.save_pretrained(tmp_path)
    save_file(
        {
            "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.1.w1.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.zeros(2, 3),
            "model.layers.0.block_sparse_moe.experts.1.w3.weight": torch.zeros(2, 3),
            "model.layers.1.block_sparse_moe.experts.0.w1.weight": torch.zeros(2, 3),
            "model.layers.1.block_sparse_moe.experts.1.w1.weight": torch.zeros(2, 3),
            "model.layers.1.block_sparse_moe.experts.0.w3.weight": torch.zeros(2, 3),
        },
        tmp_path / "model.safetensors",
    )
    model_ref = ModelReference.parse(str(tmp_path))

    arch = infer_architecture_info((model_ref,), model_ref, MergeOptions())
    module_arch = arch.modules["default"].architecture
    weights = {w.name: w for w in module_arch.layer_weights(0, cfg)}

    assert weights["model.layers.0.mlp.experts.gate_up_proj"].optional
