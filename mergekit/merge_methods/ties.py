# Copyright (C) 2023 Charles O. Goddard
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
from typing import Dict, Optional

import torch
from typing_extensions import Literal

from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod


class TiesMerge(MergeMethod):
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs,
    ) -> torch.Tensor:
        tensors = {tr.model: value for (tr, value) in input_tensors.items()}
        base = tensors[config.base_model]

        # resolve dtype for mask
        mask_dtype = (
            torch.int8 if config.parameter("int8_mask", default=False) else base.dtype
        )

        deltas = []
        weights = []
        keys = list(tensors.keys())
        for model in keys:
            if model == config.base_model:
                continue

            x = tensors[model].to(base.dtype)
            if x.shape != base.shape:
                if "lm_head" in parameter_name or "embed_tokens" in parameter_name:
                    x = x[: base.shape[0], : base.shape[1]]
                    logging.warning(f"Using submatrix of {model}:{parameter_name}")
                else:
                    logging.warning(
                        f"skipping {model}:{parameter_name} due to size mismatch"
                    )
                    continue

            if (x == base).view(-1).all():
                continue

            deltas.append(
                sparsify(x - base, config.parameter("density", model, default=0.33))
            )
            weights.append(config.parameter("weight", model, default=1.0))

            del tensors[model]
            del x

        if deltas:
            deltas = torch.stack(deltas, dim=0)
            weights = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
            while len(deltas.shape) > len(weights.shape):
                weights.unsqueeze_(-1)

            weighted_deltas = weights * deltas

            mask = get_mask(
                weighted_deltas,
                method=config.parameter("consensus_method", default="sum"),
                mask_dtype=mask_dtype,
            )

            mixed_delta = (weighted_deltas * mask).sum(dim=0)

            if config.parameter("normalize", default=True):
                divisor = (weights * mask).sum(dim=0)
                divisor[divisor == 0] = 1
                mixed_delta /= divisor

            res = base + mixed_delta
        else:
            res = base

        return res.to(base.dtype)


def sparsify(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = (sign * delta.abs()).sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign
