# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1


import logging
from typing import List

import torch

from mergekit.architecture import WeightInfo


def rectify_embed_sizes(weight_info: WeightInfo, tensors: List[torch.Tensor]):
    # TODO: use arch_info.embed_weights() instead
    if weight_info.is_embed and all(len(t.shape) == 2 for t in tensors):
        # special case - if lm_head.weight or embed_tokens.weight have a size
        # mismatch, take the largest common submatrix of all of them
        if take_common_submatrix(tensors):
            logging.warning(
                f"Using common submatrix of size {tensors[0].shape} for {weight_info.name}"
            )


def take_common_submatrix(tensors: List[torch.Tensor]) -> bool:
    min_size = [None, None]
    for t in tensors:
        for idx in range(2):
            if min_size[idx] is None or t.shape[idx] < min_size[idx]:
                min_size[idx] = t.shape[idx]

    if not all(t.shape == torch.Size(min_size) for t in tensors):
        for idx in range(len(tensors)):
            tensors[idx] = tensors[idx][: min_size[0], : min_size[1]]
        return True
    return False
