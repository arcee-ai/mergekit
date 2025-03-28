# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import enum
from typing import List

import torch
import transformers

from mergekit.tokenizer.normalization import NormalizedToken, unnormalize_token


class SubwordMethod(enum.Enum):
    MEAN = "mean"
    SUM = "sum"
    WEIGHTED_MEAN = "weighted_mean"
    FIRST_LAST = "first_last"


def subword_approximate(
    orig_embed: torch.Tensor,
    target_tokens: List[NormalizedToken],
    is_lm_head: bool,
    tok_0: transformers.PreTrainedTokenizerBase,
    subword_method: SubwordMethod = SubwordMethod.MEAN,
) -> torch.Tensor:
    res = torch.zeros(
        len(target_tokens),
        orig_embed.shape[1],
        device=orig_embed.device,
        dtype=orig_embed.dtype,
    )
    for idx, token in enumerate(target_tokens):
        text = unnormalize_token(token)
        token_ids = tok_0(text, add_special_tokens=False)["input_ids"]

        if subword_method in (SubwordMethod.MEAN, SubwordMethod.SUM):
            for id in token_ids:
                res[idx] += orig_embed[id]
            if subword_method == SubwordMethod.MEAN and len(token_ids) > 0:
                res[idx] /= len(token_ids)
        elif subword_method == SubwordMethod.WEIGHTED_MEAN:
            weights = list(range(1, len(token_ids) + 1))
            if not is_lm_head:
                # for embed_tokens, want last token to have highest weight
                # (vs. first token for lm_head)
                weights = weights[::-1]
            for id, weight in zip(token_ids, weights):
                res[idx] += weight * orig_embed[id]
            if len(token_ids) > 0:
                res[idx] /= sum(weights)
        elif subword_method == SubwordMethod.FIRST_LAST:
            if len(token_ids) == 0:
                continue
            if is_lm_head:
                res[idx] = orig_embed[token_ids[0]]
            else:
                res[idx] = orig_embed[token_ids[-1]]
        else:
            raise ValueError(f"Unknown subword method: {subword_method}")
    return res
