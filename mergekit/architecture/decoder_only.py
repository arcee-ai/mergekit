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

from typing import List, Optional

from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ModuleArchitecture,
    StaticLayeredModuleArchitecture,
    WeightInfo,
)

LLAMA_INFO = StaticLayeredModuleArchitecture(
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

MISTRAL_INFO = StaticLayeredModuleArchitecture(
    name="MistralForCausalLM",
    # lol
    **LLAMA_INFO.model_dump(exclude=["name"]),
)


STABLELM_INFO = StaticLayeredModuleArchitecture(
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

GPT_NEOX_INFO = StaticLayeredModuleArchitecture(
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

GPT2_INFO = StaticLayeredModuleArchitecture(
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

JAIS_INFO = StaticLayeredModuleArchitecture(
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

GPT2_SEQCLASS_INFO = StaticLayeredModuleArchitecture(
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


QWEN_INFO = StaticLayeredModuleArchitecture(
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

CHATGLM_INFO = StaticLayeredModuleArchitecture(
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


class PhiDecoderArchitecture(ModuleArchitecture):
    architecture_name: str = "MixFormerSequentialForCausalLM"
    num_configured_layers: int

    def __init__(self, config: PretrainedConfig):
        self.num_configured_layers = getattr(config, self.num_layers_config_key)

    def __eq__(self, rhs: ModuleArchitecture):
        if not isinstance(rhs, PhiDecoderArchitecture):
            return False
        return self.num_layers() == rhs.num_layers()

    def pre_weights(self) -> List[WeightInfo]:
        return [WeightInfo(name="layers.0.wte.weight", is_embed=True)]

    def post_weights(self) -> List[WeightInfo]:
        fake_layer_idx = self.num_configured_layers + 1
        return [
            WeightInfo(
                name=f"layers.{fake_layer_idx}.{suffix}", is_embed="linear" in suffix
            )
            for suffix in ["linear.bias", "linear.weight", "ln.bias", "ln.weight"]
        ]

    def num_layers(self, config: PretrainedConfig) -> int:
        return self.num_configured_layers

    def layer_weights(self, index: int) -> Optional[List[WeightInfo]]:
        return [
            WeightInfo(name=("layers.{idx}." + suffix).format(idx=index))
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

    def slicable(self) -> bool:
        return True

    def num_layers_config_key(self) -> str:
        return "n_layer"


PHI2_INFO = StaticLayeredModuleArchitecture(
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


PHI2_INFO_AGAIN_BUT_DIFFERENT = StaticLayeredModuleArchitecture(
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


def get_decoder_only_arch(
    arch_name: str, config: PretrainedConfig
) -> Optional[ModuleArchitecture]:
    if arch_name == PhiDecoderArchitecture.architecture_name:
        return PhiDecoderArchitecture(config)

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
    ]
    for arch in supported:
        if arch.name == arch_name:
            return arch
