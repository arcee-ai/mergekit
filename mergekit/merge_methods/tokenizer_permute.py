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

from typing import Dict

import torch

from mergekit.common import ModelReference
from mergekit.config import ConfigReader
from mergekit.graph import TensorReference
from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.slerp import slerp


class TokenizerPermutationMerge(MergeMethod):
    def __call__(
        self,
        input_tensors: Dict[TensorReference, torch.Tensor],
        embed_permutations: Dict[ModelReference, Dict[int, int]],
        config: ConfigReader,
        **_kwargs,
    ) -> torch.Tensor:
        if not input_tensors:
            return None
        if len(input_tensors) == 1:
            return list(input_tensors.values())[1]

        use_slerp = config.parameter("embed_slerp", default=False)

        models = []
        expanded = []
        masks = []
        weights = []
        for tr in input_tensors:
            models.append(tr.model)

            x = input_tensors[tr]
            p = embed_permutations[tr.model]

            xp = torch.zeros((len(p), x.shape[-1]), dtype=x.dtype, device=x.device)
            mask = torch.zeros((len(p),), dtype=torch.bool, device=x.device)
            for out_idx in p:
                in_idx = p[out_idx]
                if in_idx < 0:
                    continue

                xp[out_idx, :] = x[in_idx, :]
                mask[out_idx] = 1

            expanded.append(xp)
            masks.append(mask)

            is_base = tr.model == config.base_model
            if use_slerp:
                t = config.parameter("t", required=True)
                weight = (1.0 - t) if is_base else t
            else:
                weight = config.parameter("weight", model=tr.model, default=1.0)
            weights.append(weight)

        expanded = torch.stack(expanded, dim=0)
        masks = torch.stack(masks, dim=0).unsqueeze(-1)
        weights = (
            torch.tensor(weights, dtype=expanded.dtype, device=expanded.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        total_weight = (masks * weights).sum(dim=0)
        scale = 1 / total_weight
        scale[total_weight.abs() < 1e-8] = 0

        linear_merged = (expanded * weights * masks).sum(dim=0) * scale

        if use_slerp:
            if expanded.shape[0] != 2:
                raise RuntimeError("SLERP takes exactly two models")

            if models[0] == config.base_model:
                v0 = expanded[0, ...]
                v1 = expanded[1, ...]
            else:
                v0 = expanded[1, ...]
                v1 = expanded[0, ...]

            t = config.parameter("t", required=True)
            res = slerp(t, v0, v1)
            need_linear = (masks.sum(dim=0) != 2).squeeze(dim=-1)
            res[need_linear, :] = linear_merged[need_linear, :].to(
                device=res.device, dtype=res.dtype
            )
            return res

        return linear_merged
