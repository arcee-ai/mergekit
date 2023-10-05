from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel
from transformers import PretrainedConfig


class ArchitectureInfo(ABC):
    @abstractmethod
    def pre_weights(self) -> List[str]:
        """Return a list of all weights preceding the first layer."""
        ...

    @abstractmethod
    def post_weights(self) -> List[str]:
        """Return a list of all weights following the final layer."""
        ...

    @abstractmethod
    def layer_weight_formats(self) -> List[str]:
        """Return a list of format strings all weights associated with a layer."""
        ...

    def num_layers(self, config: PretrainedConfig) -> int:
        return config.num_hidden_layers


class StaticTensorNames(BaseModel, ArchitectureInfo):
    name: str

    pre_weight_names: List[str]  # weights applied before first layer
    post_weight_names: List[str]  # weights applied after last layer
    layer_prefix_format: str
    layer_weight_suffixes: List[str]

    class Config:
        frozen = True

    def pre_weights(self) -> List[str]:
        return self.pre_weight_names

    def post_weights(self) -> List[str]:
        return self.post_weight_names

    def layer_weight_formats(self) -> List[str]:
        res = []
        for suffix in self.layer_weight_suffixes:
            res.append(self.layer_prefix_format + "." + suffix)
        return res


LLAMA_INFO = StaticTensorNames(
    name="LlamaForCausalLM",
    pre_weight_names=["model.embed_tokens.weight"],
    post_weight_names=["model.norm.weight", "lm_head.weight"],
    layer_prefix_format="model.layers.{idx}",
    layer_weight_suffixes=[
        "input_layernorm.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ],
)

MISTRAL_INFO = StaticTensorNames(
    name="MistralForCausalLM",
    # lol
    **LLAMA_INFO.model_dump(exclude="name"),
)

GPT_NEOX_INFO = StaticTensorNames(
    name="GPTNeoXForCausalLM",
    pre_weight_names=["gpt_neox.embed_in.weight"],
    post_weight_names=[
        "gpt_neox.final_layer_norm.bias",
        "gpt_neox.final_layer_norm.weight",
        "embed_out.weight",
    ],
    layer_prefix_format="gpt_neox.layers.{idx}",
    layer_weight_suffixes=sum(
        [
            [f"{prefix}.weight", f"{prefix}.bias"]
            for prefix in [
                "attention.dense",
                "attention.query_key_value",
                "input_layernorm",
                "mlp.dense_4h_to_h",
                "mlp.dense_h_to_4h",
                "post_attention_layernorm",
            ]
        ],
        start=[],
    )
    + ["attention.bias", "attention.masked_bias", "attention.rotary_emb.inv_freq"],
)

QWEN_INFO = StaticTensorNames(
    name="QWenLMHeadModel",
    pre_weight_names=["transformer.wte.weight"],
    post_weight_names=["transformer.ln_f.weight", "lm_head.weight"],
    layer_prefix_format="transformer.h.{idx}",
    layer_weight_suffixes=[
        "attn.c_attn.bias",
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "ln_1.weight",
        "ln_2.weight",
        "mlp.c_proj.weight",
        "mlp.w1.weight",
        "mlp.w2.weight",
    ],
)


class PhiTensorNames(ArchitectureInfo):
    architecture_name: str = "MixFormerSequentialForCausalLM"

    def __init__(self, config: PretrainedConfig):
        self.config = config

    def pre_weights(self) -> List[str]:
        return ["layers.0.wte.weight"]

    def post_weights(self) -> List[str]:
        fake_layer_idx = self.config.n_layer + 1
        return [
            f"layers.{fake_layer_idx}.{suffix}"
            for suffix in ["linear.bias", "linear.weight", "ln.bias", "ln.weight"]
        ]

    def layer_weight_formats(self, layer_idx: int) -> List[str]:
        return [
            f"layers.{layer_idx}.{suffix}"
            for suffix in [
                "ln.bias",
                "ln.weight",
                "mixer.Wqkv.bias",
                "mixer.Wqkv.weight",
                "mixer.out_proj.bias",
                "mixer.out_proj.weight",
                "mixer.rotary_emb.inv_freq",
                "mlp.fc1.bias",
                "mlp.fc1.weight",
                "mlp.fc2.bias",
                "mlp.fc2.weight",
            ]
        ]

    def num_layers(self, config: PretrainedConfig) -> int:
        return config.n_layer


def get_architecture_info(config: PretrainedConfig) -> StaticTensorNames:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")

    arch_name = config.architectures[0]
    if arch_name == PhiTensorNames.architecture_name:
        return PhiTensorNames(config)

    supported = [LLAMA_INFO, MISTRAL_INFO, GPT_NEOX_INFO, QWEN_INFO]
    for arch in supported:
        if arch.name == arch_name:
            return arch

    raise RuntimeError(f"Unsupported architecture {arch_name}")
