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
from typing import Dict, List, Optional, Tuple

import torch

from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod


class TaskArithmeticMerge(MergeMethod):
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs,
    ) -> torch.Tensor:
        deltas, weights, base = get_task_vectors(parameter_name, config, input_tensors)

        if deltas:
            deltas = torch.stack(deltas, dim=0)
            weights = torch.tensor(weights, dtype=deltas.dtype, device=deltas.device)
            while len(deltas.shape) > len(weights.shape):
                weights.unsqueeze_(-1)

            mixed_delta = (weights * deltas).sum(dim=0)

            if config.parameter("normalize", default=False):
                divisor = weights.sum(dim=0)
                divisor[divisor.abs() < 1e-8] = 1
                mixed_delta /= divisor

            res = base + mixed_delta
        else:
            res = base

        return res.to(base.dtype)


def get_task_vectors(
    parameter_name: str,
    config: ConfigReader,
    input_tensors: Dict[TensorReference, torch.Tensor],
    skip_same: bool = False,
    default_weight: Optional[float] = None,
) -> Tuple[List[torch.Tensor], List[float], torch.Tensor]:
    tensors = {tr.model: value for (tr, value) in input_tensors.items()}
    keys = list(tensors.keys())
    base = tensors[config.base_model]

    deltas = []
    weights = []
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

        if skip_same and (x == base).view(-1).all():
            continue

        deltas.append(x - base)
        weights.append(
            config.parameter(
                "weight",
                model,
                default=default_weight,
                required=default_weight is None,
            )
        )

        del tensors[model]
        del x
    return deltas, weights, base
