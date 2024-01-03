# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from abc import ABC, abstractmethod
from typing import List, Optional

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

    @abstractmethod
    def embed_weights(self) -> List[str]:
        ...

    def num_layers(self, config: PretrainedConfig) -> int:
        return config.num_hidden_layers

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"


class StaticTensorNames(ArchitectureInfo, BaseModel, frozen=True):
    name: str

    pre_weight_names: List[str]  # weights applied before first layer
    post_weight_names: List[str]  # weights applied after last layer
    embed_weight_names: List[str]  # weights for embed/lm_head
    layer_prefix_format: str
    layer_weight_suffixes: List[str]
    num_layers_key: Optional[str] = None

    def pre_weights(self) -> List[str]:
        return self.pre_weight_names

    def post_weights(self) -> List[str]:
        return self.post_weight_names

    def embed_weights(self) -> List[str]:
        return self.embed_weight_names

    def layer_weight_formats(self) -> List[str]:
        res = []
        for suffix in self.layer_weight_suffixes:
            res.append(self.layer_prefix_format + "." + suffix)
        return res

    def num_layers_config_key(self) -> str:
        if self.num_layers_key:
            return self.num_layers_key
        return super().num_layers_config_key()

    def num_layers(self, config: PretrainedConfig) -> int:
        return getattr(config, self.num_layers_config_key())


LLAMA_INFO = StaticTensorNames(
    name="LlamaForCausalLM",
    pre_weight_names=["model.embed_tokens.weight"],
    post_weight_names=["model.norm.weight", "lm_head.weight"],
    embed_weight_names=["model.embed_tokens.weight", "lm_head.weight"],
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
    **LLAMA_INFO.model_dump(exclude=["name"]),
)


STABLELM_INFO = StaticTensorNames(
    name="StableLMEpochForCausalLM",
    post_weight_names=LLAMA_INFO.post_weight_names + ["model.norm.bias"],
    layer_weight_suffixes=LLAMA_INFO.layer_weight_suffixes
    + [
        "input_layernorm.bias",
        "post_attention_layernorm.bias",
    ],
    **LLAMA_INFO.model_dump(
        exclude=["name", "layer_weight_suffixes", "post_weight_names"]
    ),
)

GPT_NEOX_INFO = StaticTensorNames(
    name="GPTNeoXForCausalLM",
    pre_weight_names=["gpt_neox.embed_in.weight"],
    post_weight_names=[
        "gpt_neox.final_layer_norm.bias",
        "gpt_neox.final_layer_norm.weight",
        "embed_out.weight",
    ],
    embed_weight_names=["gpt_neox.embed_in.weight", "embed_out.weight"],
    layer_prefix_format="gpt_neox.layers.{idx}",
    layer_weight_suffixes=sum(
        (
            [f"{prefix}.weight", f"{prefix}.bias"]
            for prefix in [
                "attention.dense",
                "attention.query_key_value",
                "input_layernorm",
                "mlp.dense_4h_to_h",
                "mlp.dense_h_to_4h",
                "post_attention_layernorm",
            ]
        ),
        start=[],
    )
    + ["attention.bias", "attention.masked_bias", "attention.rotary_emb.inv_freq"],
)

GPT2_INFO = StaticTensorNames(
    name="GPT2LMHeadModel",
    pre_weight_names=["wte.weight", "wpe.weight"],
    post_weight_names=["ln_f.weight", "ln_f.bias"],
    embed_weight_names=["wte.weight"],
    layer_prefix_format="h.{idx}",
    layer_weight_suffixes=[
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    ],
    num_layers_key="n_layer",
)

GPT2_SEQCLASS_INFO = StaticTensorNames(
    name="GPT2ForSequenceClassification",
    pre_weight_names=["transformer.wte.weight", "transformer.wpe.weight"],
    post_weight_names=[
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "score.weight",
    ],
    layer_prefix_format="transformer.h.{idx}",
    embed_weight_names=GPT2_INFO.embed_weight_names,
    layer_weight_suffixes=GPT2_INFO.layer_weight_suffixes,
    num_layers_key=GPT2_INFO.num_layers_key,
)


QWEN_INFO = StaticTensorNames(
    name="QWenLMHeadModel",
    pre_weight_names=["transformer.wte.weight"],
    post_weight_names=["transformer.ln_f.weight", "lm_head.weight"],
    embed_weight_names=["transformer.wte.weight", "lm_head.weight"],
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

CHATGLM_INFO = StaticTensorNames(
    name="ChatGLMModel",
    pre_weight_names=[
        "transformer.embedding.word_embeddings.weight",
        "transformer.rotary_pos_emb.inv_freq",
    ],
    post_weight_names=[
        "transformer.encoder.final_layernorm.weight",
        "transformer.output_layer.weight",
    ],
    embed_weight_names=[
        "transformer.embedding.word_embeddings.weight",
        "transformer.output_layer.weight",
    ],
    layer_prefix_format="transformer.encoder.layers.{idx}",
    layer_weight_suffixes=[
        "input_layernorm.weight",
        "mlp.dense_4h_to_h.weight",
        "mlp.dense_h_to_4h.weight",
        "post_attention_layernorm.weight",
        "self_attention.dense.weight",
        "self_attention.query_key_value.bias",
        "self_attention.query_key_value.weight",
    ],
)


class PhiTensorNames(ArchitectureInfo):
    architecture_name: str = "MixFormerSequentialForCausalLM"

    def __init__(self, config: PretrainedConfig):
        self.config = config

    def __eq__(self, rhs: "PhiTensorNames"):
        if not isinstance(rhs, PhiTensorNames):
            return False
        return self.num_layers() == rhs.num_layers()

    def pre_weights(self) -> List[str]:
        return ["layers.0.wte.weight"]

    def post_weights(self) -> List[str]:
        fake_layer_idx = self.config.n_layer + 1
        return [
            f"layers.{fake_layer_idx}.{suffix}"
            for suffix in ["linear.bias", "linear.weight", "ln.bias", "ln.weight"]
        ]

    def embed_weights(self) -> List[str]:
        fake_layer_idx = self.config.n_layer + 1
        return [
            "layers.0.wte.weight",
            f"layers.{fake_layer_idx}.linear.weight",
            f"layers.{fake_layer_idx}.linear.bias",
        ]

    def layer_weight_formats(self) -> List[str]:
        return [
            ("layers.{idx}." + suffix)
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

    def num_layers_config_key(self) -> str:
        return "n_layer"


PHI2_INFO = StaticTensorNames(
    name="PhiForCausalLM",
    pre_weight_names=["transformer.embd.wte.weight"],
    post_weight_names=[
        "lm_head.linear.bias",
        "lm_head.linear.weight",
        "lm_head.ln.bias",
        "lm_head.ln.weight",
    ],
    embed_weight_names=["lm_head.linear.weight", "transformer.embd.wte.weight"],
    layer_prefix_format="transformer.h.{idx}",
    layer_weight_suffixes=[
        "ln.bias",
        "ln.weight",
        "mixer.out_proj.bias",
        "mixer.out_proj.weight",
        "mixer.Wqkv.bias",
        "mixer.Wqkv.weight",
        "mlp.fc1.bias",
        "mlp.fc1.weight",
        "mlp.fc2.bias",
        "mlp.fc2.weight",
    ],
)


def get_architecture_info(config: PretrainedConfig) -> StaticTensorNames:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")

    arch_name = config.architectures[0]
    if arch_name == PhiTensorNames.architecture_name:
        return PhiTensorNames(config)

    supported = [
        LLAMA_INFO,
        MISTRAL_INFO,
        GPT_NEOX_INFO,
        QWEN_INFO,
        GPT2_INFO,
        GPT2_SEQCLASS_INFO,
        CHATGLM_INFO,
        STABLELM_INFO,
        PHI2_INFO,
    ]
    for arch in supported:
        if arch.name == arch_name:
            return arch

    raise RuntimeError(f"Unsupported architecture {arch_name}")
