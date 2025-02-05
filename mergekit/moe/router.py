# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import math
from typing import Dict, List, Union

import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from mergekit.common import ModelReference
from mergekit.moe.config import Expert


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
        [(tokenizer.bos_token or "") + p for p in prompts],
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
    elif mode == "uniform_random":
        in_features = model_cfg.hidden_size
        scale = math.sqrt(1.0 / in_features)
        return (
            torch.rand(
                (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
            )
            * 2
            * scale
            - scale
        )
    elif mode == "cheap_embed":
        embed = model_ref.lazy_loader(lazy_unpickle=lazy_unpickle).get_tensor(
            "model.embed_tokens.weight"
        )

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
