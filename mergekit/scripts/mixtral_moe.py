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

import logging
import os
import sys
from typing import Dict, List, Optional, Union

import click
import torch
import tqdm
import transformers
import yaml
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

import mergekit.architecture
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io import LazyTensorLoader, TensorWriter
from mergekit.merge import MergeOptions
from mergekit.options import add_merge_options

# Create a Mixtral MoE from a set of equally-sized Mistral (or Llama) models.
# Takes the path to a yml config and an output path.
# Config schema is the two classes below.


class Expert(BaseModel):
    source_model: str

    positive_prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    noise_scale: Optional[float] = None

    @property
    def model_ref(self):
        return ModelReference.parse(self.source_model)


class MistralMOEConfig(BaseModel):
    base_model: str
    experts: List[Expert]
    gate_mode: str = "hidden"  # possible values: "hidden", "cheap_embed", "random"
    # "hidden" uses hidden state vectors for the given prompts for each layer
    # "cheap_embed" uses the average of token embeddings for the prompts, same for each layer
    # "random" is random
    dtype: Optional[str] = None
    experts_per_token: int = 2


def get_hidden_states(
    model: Union[MistralForCausalLM, LlamaForCausalLM],
    tokenized: transformers.BatchEncoding,
    average: bool = True,
) -> List[torch.Tensor]:
    with torch.no_grad():
        output: CausalLMOutputWithPast = model(
            **tokenized.to(model.device), output_hidden_states=True, return_dict=True
        )
    hidden_states = torch.stack(
        output.hidden_states[:-1]
    )  # (num_layers, batch_size, seq_len, hidden_size)
    if average:
        # use average over sequence
        hidden_states = hidden_states.sum(dim=2) / hidden_states.shape[2]
    else:
        # take last value
        hidden_states = hidden_states[:, :, -1, :]
    return hidden_states.sum(dim=1) / hidden_states.shape[1]


def get_cheap_embedding(
    embed: torch.Tensor,
    tokenized: Dict[str, torch.Tensor],
    num_layers: int,
    vocab_size: int,
) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(
        tokenized["input_ids"], num_classes=vocab_size
    )  # (batch_size, seq_len, 32000)
    h = onehot.float() @ embed.float()  # (batch_size, seq_len, hidden_size)
    embedded = (
        (h * tokenized["attention_mask"].unsqueeze(-1))
        .sum(dim=1)
        .sum(dim=0, keepdim=True)
    )  # (1, hidden_size)
    res = embedded / embedded.norm(dim=-1, keepdim=True).clamp(
        min=1e-8
    )  # (1, hidden_size)
    return res.repeat(num_layers, 1)


def tokenize_prompts(
    prompts: List[str], tokenizer: transformers.PreTrainedTokenizerBase
):
    return tokenizer(
        [tokenizer.bos_token or "" + p for p in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )


def get_gate_params(
    model_ref: ModelReference,
    tokenizer: transformers.PreTrainedTokenizerBase,
    experts: List[Expert],
    mode: str = "hidden",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lazy_unpickle: bool = False,
    trust_remote_code: bool = False,
    device: str = "auto",
):
    gate_vecs = []
    _do_it = None

    model_cfg = model_ref.config(trust_remote_code=trust_remote_code)

    if mode == "random":
        return torch.randn(
            (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
        )
    elif mode == "cheap_embed":
        embed = LazyTensorLoader(
            model_ref.tensor_index(), lazy_unpickle=lazy_unpickle
        ).get_tensor("model.embed_tokens.weight")

        def _do_it(tokenized):
            return get_cheap_embedding(
                embed,
                tokenized,
                num_layers=model_cfg.num_hidden_layers,
                vocab_size=model_cfg.vocab_size,
            )

    elif mode in ("hidden", "hidden_avg", "hidden_last"):
        model = AutoModelForCausalLM.from_pretrained(
            model_ref.model.path,
            revision=model_ref.model.revision,
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code,
        )

        def _do_it(tokenized):
            return get_hidden_states(
                model, tokenized=tokenized, average=mode == "hidden_avg"
            )

    gate_vecs = []
    for expert in tqdm.tqdm(experts, desc="expert prompts"):
        hidden_states = _do_it(tokenize_prompts(expert.positive_prompts, tokenizer))
        if expert.negative_prompts:
            hidden_states -= _do_it(
                tokenize_prompts(expert.negative_prompts, tokenizer)
            )

        hidden_states /= hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gate_vecs.append(hidden_states)
    gate_vecs = torch.stack(gate_vecs, dim=0)  # (num_expert, num_layer, hidden_size)
    return gate_vecs.permute(1, 0, 2)


def warn_degenerate_gates(gate_vecs: torch.Tensor, threshold: float = 5.0):
    degen_indices = []
    num_layers, _num_experts, _hidden_size = gate_vecs.shape
    for idx in range(num_layers):
        c = torch.linalg.cond(gate_vecs[idx, :, :].float())
        if c > threshold:
            degen_indices.append(idx)

    if degen_indices:
        if len(degen_indices) == 1:
            layer_str = f"layer {degen_indices[0]}"
            verb = "has"
        elif len(degen_indices) == 2:
            layer_str = f"layers {' and '.join(map(str, degen_indices))}"
            verb = "have"
        elif len(degen_indices) >= num_layers:
            layer_str = "ALL layers"
            verb = "have"
        else:
            layer_str = (
                "layers "
                + ", ".join(map(str, degen_indices[:-1]))
                + ", and "
                + str(degen_indices[-1])
            )
            verb = "have"

        logging.warning(
            f"{layer_str} {verb} degenerate routing parameters "
            "- your prompts may be too similar."
        )
        logging.warning("One or more experts will be underutilized in your model.")


def is_bad_config(config: MistralMOEConfig, allow_all_same: bool = False) -> bool:
    if len(config.experts) < 2:
        logging.error("Must include at least two experts.")
        return True

    if config.gate_mode == "random":
        return False  # eh we're good

    def prompt_tup(e: Expert):
        return (tuple(e.positive_prompts), tuple(e.negative_prompts or []))

    # let's just nip this trend in the bud
    p_first = prompt_tup(config.experts[0])
    if all(prompt_tup(e) == p_first for e in config.experts[1:]):
        logging.error(
            "Your positive and negative prompts are identical for all experts. This will not produce a functioning MoE."
        )
        logging.error(
            "For each expert, `positive_prompts` must contain one or more example prompt reflecting what should be routed to that expert."
        )
        return True

    if not allow_all_same:
        if all(
            e.source_model == config.experts[0].source_model for e in config.experts[1:]
        ):
            logging.error(
                "All of your expert models are the same. This will produce "
                "a model that uses more resources but gives the exact same output. "
                "If you plan to train the model after merging, proceed with the "
                "--i-understand-this-is-not-useful-without-training flag."
            )
            return True


def build(
    config: MistralMOEConfig,
    out_path: str,
    merge_options: MergeOptions,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    device: str = "auto",
    allow_all_same: bool = False,
):
    if is_bad_config(config, allow_all_same=allow_all_same):
        sys.exit(1)

    if config.experts_per_token < 1:
        logging.error("Experts per token must be >= 1")
        sys.exit(1)
    if config.experts_per_token > len(config.experts):
        logging.error("Experts per token must be <= number of experts")
        sys.exit(1)

    base_model = ModelReference.parse(config.base_model)
    base_cfg = base_model.config(trust_remote_code=merge_options.trust_remote_code)
    if not isinstance(base_cfg, MistralConfig):
        base_cfg_mistral = MistralConfig(**base_cfg.to_dict())
        base_cfg_mistral.sliding_window = None
        base_cfg_mistral.max_position_embeddings = base_cfg.max_position_embeddings
        base_cfg = base_cfg_mistral

    out_cfg = MixtralConfig(**base_cfg.to_dict())
    out_cfg.architectures = ["MixtralForCausalLM"]
    out_cfg.num_local_experts = len(config.experts)
    out_cfg.num_experts_per_tok = config.experts_per_token
    out_cfg.sliding_window = None
    if config.dtype:
        out_cfg.torch_dtype = config.dtype
    out_cfg.save_pretrained(out_path)

    if (out_cfg.num_local_experts & (out_cfg.num_local_experts - 1)) != 0:
        logging.warning(
            f"Your model has {out_cfg.num_local_experts} experts, which is "
            "not a power of two. The model will not be usable in llama.cpp."
        )

    loaders: Dict[ModelReference, LazyTensorLoader] = {}
    for model in tqdm.tqdm(
        [base_model] + [e.model_ref for e in config.experts], desc="Warm up loaders"
    ):
        loaders[model] = LazyTensorLoader(
            model.tensor_index(cache_dir=merge_options.transformers_cache),
            lazy_unpickle=merge_options.lazy_unpickle,
        )

    base_loader = loaders.get(base_model)
    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )

    if config.dtype:
        out_dtype = dtype_from_name(config.dtype)
    elif base_cfg.torch_dtype:
        out_dtype = base_cfg.torch_dtype
        if isinstance(out_dtype, str):
            out_dtype = dtype_from_name(out_dtype)
    else:
        out_dtype = None

    logging.info("Copying parameters...")
    MISTRAL_INFO = mergekit.architecture.MISTRAL_INFO
    for weight_info in MISTRAL_INFO.pre_weights(base_cfg) + MISTRAL_INFO.post_weights(
        base_cfg
    ):
        tensor_name = weight_info.name
        tensor = base_loader.get_tensor(tensor_name)
        if not out_dtype:
            # All else has failed, take the first dtype we see
            out_dtype = tensor.dtype
        writer.save_tensor(
            tensor_name, tensor.to(dtype=out_dtype), clone=merge_options.clone_tensors
        )

    for layer_idx in range(base_cfg.num_hidden_layers):
        for weight_info in MISTRAL_INFO.layer_weights(index=layer_idx, config=base_cfg):
            tensor_name = weight_info.name

            if ".mlp." in tensor_name:
                for moe_index, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace(
                        ".mlp.gate_proj", f".block_sparse_moe.experts.{moe_index}.w1"
                    )
                    expert_name = expert_name.replace(
                        ".mlp.down_proj", f".block_sparse_moe.experts.{moe_index}.w2"
                    )
                    expert_name = expert_name.replace(
                        ".mlp.up_proj", f".block_sparse_moe.experts.{moe_index}.w3"
                    )
                    expert_loader = loaders.get(expert.model_ref)
                    tensor = expert_loader.get_tensor(tensor_name)
                    if expert.noise_scale:
                        tensor += torch.randn_like(tensor) * expert.noise_scale
                    writer.save_tensor(
                        expert_name, tensor.to(dtype=out_dtype), clone=True
                    )
                continue
            writer.save_tensor(
                tensor_name, base_loader.get_tensor(tensor_name).to(dtype=out_dtype)
            )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model.model.path, revision=base_model.model.revision
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info("Getting gate parameters...")
    gate_vecs = get_gate_params(
        base_model,
        tokenizer,
        config.experts,
        mode=config.gate_mode,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lazy_unpickle=merge_options.lazy_unpickle,
        trust_remote_code=merge_options.trust_remote_code,
        device=device,
    )
    # gate_vecs: (num_layers, num_experts, hidden_size)

    warn_degenerate_gates(gate_vecs)

    for layer_idx in range(base_cfg.num_hidden_layers):
        writer.save_tensor(
            f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
            gate_vecs[layer_idx, :, :].contiguous().to(dtype=out_dtype),
        )
    writer.finalize()

    if merge_options.copy_tokenizer:
        logging.info("Saving tokenizer...")
        tokenizer.save_pretrained(out_path, safe_serialization=True)

    logging.info("Done.")


@click.command("mergekit-moe")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("out_path", type=click.Path())
@click.option(
    "--load-in-4bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 4bit for computing hidden states",
)
@click.option(
    "--load-in-8bit",
    is_flag=True,
    type=bool,
    default=False,
    help="Load model in 8bit for computing hidden states",
)
@click.option(
    "--device",
    type=str,
    default="auto",
    help="Device to use to compute embeddings",
    show_default=True,
)
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@click.option(
    "--i-understand-this-is-not-useful-without-training",
    type=bool,
    default=False,
    is_flag=True,
    help="Really make the questionable model you want.",
)
@add_merge_options
def main(
    config_path: str,
    out_path: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    device: str,
    merge_options: MergeOptions,
    verbose: bool,
    i_understand_this_is_not_useful_without_training: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if merge_options.cuda:
        logging.warning(
            '--cuda is a no-op for mergekit-moe, use "--device cuda" instead'
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config_source = file.read()

    config = MistralMOEConfig.model_validate(yaml.safe_load(config_source))
    build(
        config,
        out_path=out_path,
        merge_options=merge_options,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        device=device,
        allow_all_same=i_understand_this_is_not_useful_without_training,
    )

    if merge_options.write_model_card:
        # TODO: generate a README.md as well
        with open(
            os.path.join(out_path, "mergekit_moe_config.yml"), "w", encoding="utf-8"
        ) as fp:
            fp.write(config_source)


if __name__ == "__main__":
    main()
