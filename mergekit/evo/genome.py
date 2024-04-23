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

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from pydantic import BaseModel, model_validator

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration

METHOD_PARAM_MAPS = {
    "linear": ["weight"],
    "task_arithmetic": ["weight"],
    "ties": ["weight", "density"],
    "dare_ties": ["weight", "density"],
    "slerp": ["t"],
}


class ModelGenomeDefinition(BaseModel, frozen=True):
    models: List[ModelReference]
    merge_method: str
    base_model: Optional[ModelReference] = None
    tokenizer_source: Optional[str] = None
    layer_granularity: int = 1
    normalize: Optional[bool] = None
    allow_negative_weights: bool = False

    @model_validator(mode="after")
    def validate(self):
        assert self.merge_method in METHOD_PARAM_MAPS, "Invalid merge method"
        assert self.layer_granularity > 0, "layer_granularity must be positive"

        if self.merge_method in ["ties", "dare_ties", "task_arithmetic"]:
            assert self.base_model is not None, "base_model is required for this method"
        else:
            assert self.base_model is None, "base_model is not used for this method"

        return self


class ModelGenome:
    definiton: ModelGenomeDefinition
    num_layers: int
    _input_config_example: transformers.PretrainedConfig

    def __init__(
        self, definition: ModelGenomeDefinition, trust_remote_code: bool = False
    ):
        self.definition = definition

        self._input_config_example = self.definition.models[0].config(
            trust_remote_code=trust_remote_code
        )
        self.num_layers = self._input_config_example.num_hidden_layers

        assert (
            self.num_layers % self.definition.layer_granularity == 0
        ), "Number of layers must be a multiple of layer_granularity"

    def initial_genotype(self, random: bool = False) -> torch.Tensor:
        """Generate an initial genotype for the given number of layers."""
        n_layer_groups = self.num_layers // self.definition.layer_granularity
        n_models = len(self.definition.models)
        n_params = len(METHOD_PARAM_MAPS[self.definition.merge_method])

        if random:
            return torch.rand(n_layer_groups, n_models, n_params)
        else:
            x0_t = torch.zeros(n_layer_groups, n_models, n_params)
            # weight is always first
            x0_t[:, :, 0] = 1 / n_models
            if n_params > 1:
                # sometimes followed by density
                x0_t[:, :, 1:] = 1
            return x0_t

    def genotype_merge_config(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> MergeConfiguration:
        """Convert a genotype tensor to a mergekit configuration."""

        genotype = self._to_torch(genotype)

        (n_layer_groups, n_models, n_params) = genotype.shape
        assert n_layer_groups * self.definition.layer_granularity == self.num_layers
        assert n_models == len(self.definition.models)
        assert n_params == len(METHOD_PARAM_MAPS[self.definition.merge_method])

        slices = []
        for layer_idx in range(
            0,
            n_layer_groups * self.definition.layer_granularity,
            self.definition.layer_granularity,
        ):
            s = {
                "sources": [
                    {
                        "model": self.definition.models[i],
                        "layer_range": [
                            layer_idx,
                            layer_idx + self.definition.layer_granularity,
                        ],
                    }
                    for i in range(n_models)
                ]
            }

            if self.definition.merge_method == "slerp":
                # Choose the two models with the highest weight and
                # calculate the interpolation parameter t
                chosen = torch.topk(
                    genotype[layer_idx // self.definition.layer_granularity, :, 0], 2
                )
                t = torch.softmax(chosen.values, dim=-1)[1].item()
                s["parameters"] = {"t": t}
                s["base_model"] = self.definition.models[chosen.indices[0].item()]
                s["sources"] = [
                    s["sources"][chosen.indices[0].item()],
                    s["sources"][chosen.indices[1].item()],
                ]
                if self.definition.tokenizer_source:
                    s["sources"][0]["parameters"] = {"weight": 1 - t}
                    s["sources"][1]["parameters"] = {"weight": t}
            else:
                for model_idx in range(n_models):
                    params = {}
                    for param_idx, param in enumerate(
                        METHOD_PARAM_MAPS[self.definition.merge_method]
                    ):
                        params[param] = genotype[
                            layer_idx // self.definition.layer_granularity,
                            model_idx,
                            param_idx,
                        ]
                        if param == "density":
                            # ensure density is in [0, 1]
                            params[param] = torch.abs(params[param]).clamp(0, 1).item()
                        if not self.definition.allow_negative_weights and param in [
                            "weight",
                            "t",
                        ]:
                            params[param] = torch.abs(params[param]).item()
                    s["sources"][model_idx]["parameters"] = params

            if self.definition.base_model and (
                self.definition.base_model not in self.definition.models
            ):
                s["sources"].append(
                    {
                        "model": self.definition.base_model,
                        "layer_range": [
                            layer_idx,
                            layer_idx + self.definition.layer_granularity,
                        ],
                    }
                )

            slices.append(s)

        normalize = self.definition.normalize
        if normalize is None:
            normalize = self.definition.merge_method in ["ties", "dare_ties", "linear"]
        return MergeConfiguration.model_validate(
            {
                "merge_method": self.definition.merge_method,
                "slices": slices,
                "parameters": {
                    "normalize": normalize,
                    "int8_mask": True,
                },
                "dtype": "bfloat16",
                "base_model": self.definition.base_model,
                "tokenizer_source": self.definition.tokenizer_source,
            }
        )

    def _to_torch(self, genotype: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not isinstance(genotype, torch.Tensor):
            genotype = torch.tensor(genotype)
        if len(genotype.shape) == 1:
            genotype = genotype.view(
                self.num_layers // self.definition.layer_granularity,
                len(self.definition.models),
                -1,
            )

        return genotype

    def gene_names(self) -> Dict[Tuple[int, int, int], str]:
        """Return a mapping from genotype indices to names."""
        res = {}
        for i in range(self.num_layers // self.definition.layer_granularity):
            for j in range(len(self.definition.models)):
                for k in range(len(METHOD_PARAM_MAPS[self.definition.merge_method])):
                    param_name = METHOD_PARAM_MAPS[self.definition.merge_method][k]
                    res[(i, j, k)] = f"lg{i}_m{j}_{param_name}"
        return res

    def genotype_to_param_arrays(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """Convert a genotype tensor to a dictionary of numpy arrays."""
        genotype = self._to_torch(genotype)

        res = {}
        for idx, param_name in enumerate(
            METHOD_PARAM_MAPS[self.definition.merge_method]
        ):
            for model_idx, model in enumerate(self.definition.models):
                model_name = os.path.basename(model.model.path)
                res[f"{model_name}_{param_name}"] = genotype[:, model_idx, idx]

        return res
