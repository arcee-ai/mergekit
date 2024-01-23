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
from typing import ClassVar, List, Optional

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

    def all_weights(self, config: PretrainedConfig) -> List[str]:
        num_layers = self.num_layers(config)
        tensor_names = list(self.pre_weights())
        for layer_idx in range(num_layers):
            for f in self.layer_weight_formats():
                tensor_names.append(f.format(idx=layer_idx))
        tensor_names.extend(self.post_weights())
        return tensor_names


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


class MixtralTensorNames(ArchitectureInfo, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralTensorNames(num_local_experts=config.num_local_experts)

    def pre_weights(self) -> List[str]:
        return MISTRAL_INFO.pre_weights()

    def post_weights(self) -> List[str]:
        return MISTRAL_INFO.post_weights()

    def embed_weights(self) -> List[str]:
        return MISTRAL_INFO.embed_weights()

    def num_layers_config_key(self) -> str:
        return MISTRAL_INFO.num_layers_config_key()

    def layer_weight_formats(self) -> List[str]:
        num_experts = self.num_local_experts
        res = [fmt for fmt in MISTRAL_INFO.layer_weight_formats() if ".mlp." not in fmt]
        for expert_idx in range(num_experts):
            for param in ("w1", "w2", "w3"):
                fmt = (
                    MISTRAL_INFO.layer_prefix_format
                    + f".block_sparse_moe.experts.{expert_idx}.{param}.weight"
                )
                res.append(fmt)
        res.append(MISTRAL_INFO.layer_prefix_format + ".block_sparse_moe.gate.weight")
        return res


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

JAIS_INFO = StaticTensorNames(
    name="JAISLMHeadModel",
    pre_weight_names=["transformer.wte.weight", "transformer.relative_pe.slopes"],
    post_weight_names=["transformer.ln_f.weight", "transformer.ln_f.bias"],
    embed_weight_names=["transformer.wte.weight"],
    layer_prefix_format="transformer.h.{idx}",
    layer_weight_suffixes=[
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_1.weight",
        "ln_1.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_fc2.weight",
        "mlp.c_fc2.bias",
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

FALCON_INFO = StaticTensorNames(
    name="FalconForCausalLM",
    pre_weight_names=["transformer.word_embeddings.weight"],
    post_weight_names=[
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
    ],
    embed_weight_names=["transformer.word_embeddings.weight", "lm_head.weight"],
    layer_prefix_format="transformer.h.{idx}",
    layer_weight_suffixes=[
        "ln_attn.bias",
        "ln_attn.weight",
        "ln_mlp.bias",
        "ln_mlp.weight",
        "mlp.dense_4h_to_h.weight",
        "mlp.dense_h_to_4h.weight",
        "self_attention.dense.weight",
        "self_attention.query_key_value.weight",
    ],
)


class PhiTensorNames(ArchitectureInfo, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixFormerSequentialForCausalLM"
    n_layer: int

    def from_config(cls, config: PretrainedConfig):
        return PhiTensorNames(n_layer=config.n_layer)

    def pre_weights(self) -> List[str]:
        return ["layers.0.wte.weight"]

    def post_weights(self) -> List[str]:
        fake_layer_idx = self.n_layer
        return [
            f"layers.{fake_layer_idx}.{suffix}"
            for suffix in ["linear.bias", "linear.weight", "ln.bias", "ln.weight"]
        ]

    def embed_weights(self) -> List[str]:
        fake_layer_idx = self.n_layer
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
    num_layers_key="n_layer",
)


PHI2_INFO_AGAIN_BUT_DIFFERENT = StaticTensorNames(
    name="PhiForCausalLM",
    pre_weight_names=["model.embed_tokens.weight"],
    post_weight_names=[
        "lm_head.bias",
        "lm_head.weight",
        "model.final_layernorm.bias",
        "model.final_layernorm.weight",
    ],
    embed_weight_names=["lm_head.weight", "model.embed_tokens.weight"],
    layer_prefix_format="model.layers.{idx}",
    layer_weight_suffixes=[
        "input_layernorm.bias",
        "input_layernorm.weight",
        "self_attn.dense.bias",
        "self_attn.dense.weight",
        "self_attn.q_proj.bias",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.v_proj.weight",
        "mlp.fc1.bias",
        "mlp.fc1.weight",
        "mlp.fc2.bias",
        "mlp.fc2.weight",
    ],
)


BAICHUAN_INFO = StaticTensorNames(
    name="BaichuanForCausalLM",
    pre_weight_names=["model.embed_tokens.weight"],
    post_weight_names=["model.norm.weight", "lm_head.weight"],
    embed_weight_names=["model.embed_tokens.weight", "lm_head.weight"],
    layer_prefix_format="model.layers.{idx}",
    layer_weight_suffixes=[
        "input_layernorm.weight",
        "self_attn.W_pack.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        "mlp.up_proj.weight",
    ],
)


def get_architecture_info(config: PretrainedConfig) -> StaticTensorNames:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")

    arch_name = config.architectures[0]
    if arch_name == PhiTensorNames.ARCHITECTURE_NAME:
        return PhiTensorNames.from_config(config)
    if arch_name == MixtralTensorNames.ARCHITECTURE_NAME:
        return MixtralTensorNames.from_config(config)

    if arch_name == PHI2_INFO.name:
        if config.model_type == "phi-msft":
            return PHI2_INFO
        elif config.model_type == "phi":
            return PHI2_INFO_AGAIN_BUT_DIFFERENT

    supported = [
        LLAMA_INFO,
        MISTRAL_INFO,
        GPT_NEOX_INFO,
        QWEN_INFO,
        GPT2_INFO,
        GPT2_SEQCLASS_INFO,
        CHATGLM_INFO,
        STABLELM_INFO,
        JAIS_INFO,
        BAICHUAN_INFO,
        FALCON_INFO,
    ]
    for arch in supported:
        if arch.name == arch_name:
            return arch

    raise RuntimeError(f"Unsupported architecture {arch_name}")
