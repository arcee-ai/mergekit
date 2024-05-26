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
from typing import Any, Dict, List, Optional, Union

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


class InvalidGenotypeError(RuntimeError):
    pass


class ModelGenomeDefinition(BaseModel, frozen=True):
    models: List[ModelReference]
    merge_method: str
    base_model: Optional[ModelReference] = None
    tokenizer_source: Optional[str] = None
    layer_granularity: int = 0
    normalize: Optional[bool] = None
    allow_negative_weights: bool = False
    filters: Optional[List[str]] = None
    smooth: bool = False

    @model_validator(mode="after")
    def validate(self):
        assert self.merge_method in METHOD_PARAM_MAPS, "Invalid merge method"

        if self.merge_method in ["ties", "dare_ties", "task_arithmetic"]:
            assert self.base_model is not None, "base_model is required for this method"

        if self.merge_method == "slerp":
            assert not self.smooth, "smooth is not supported for slerp merge method"
            assert (
                not self.filters
            ), "tensor name filtering is not supported for slerp merge method"

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
            self.definition.layer_granularity < 1
            or self.num_layers % self.definition.layer_granularity == 0
        ), "Number of layers must be a multiple of layer_granularity"

    def initial_genotype(self, random: bool = False) -> torch.Tensor:
        """Generate an initial genotype for the given number of layers."""
        if self.definition.layer_granularity > 0:
            n_layer_groups = self.num_layers // self.definition.layer_granularity
        else:
            n_layer_groups = 1
        n_param_sets = len(self.definition.filters or []) + 1
        n_models = len(self.definition.models)
        n_params = len(METHOD_PARAM_MAPS[self.definition.merge_method])

        if random:
            return torch.rand(n_layer_groups, n_models, n_param_sets, n_params)
        else:
            x0_t = torch.zeros(n_layer_groups, n_models, n_param_sets, n_params)
            # weight is always first
            x0_t[:, :, :, 0] = 1 / n_models
            if n_params > 1:
                # sometimes followed by density
                x0_t[:, :, :, 1:] = 1
            return x0_t

    def genotype_merge_config(
        self, genotype: Union[torch.Tensor, np.ndarray]
    ) -> MergeConfiguration:
        """Convert a genotype tensor to a mergekit configuration."""

        genotype = self._to_torch(genotype)

        (n_layer_groups, n_models, n_param_sets, n_params) = genotype.shape
        if self.definition.layer_granularity > 0:
            assert n_layer_groups * self.definition.layer_granularity == self.num_layers
        assert n_models == len(self.definition.models)
        assert n_params == len(METHOD_PARAM_MAPS[self.definition.merge_method])

        if self.definition.merge_method == "slerp":
            slices = self._slerp_slices(genotype)
            models = None
        else:
            param_arrays = {}
            for param_idx, param in enumerate(
                METHOD_PARAM_MAPS[self.definition.merge_method]
            ):
                values = genotype[:, :, :, param_idx]
                if param == "density":
                    # ensure density is in [0, 1]
                    values = torch.abs(values).clamp(0, 1)
                if not self.definition.allow_negative_weights and param in [
                    "weight",
                    "t",
                ]:
                    values = torch.abs(values)
                param_arrays[param] = values

            if self.definition.smooth:
                slices = None
                models = self._smooth_config_models(n_param_sets, param_arrays)
            else:
                models = None
                slices = self._discrete_config_slices(
                    n_layer_groups, n_param_sets, param_arrays
                )

        normalize = self.definition.normalize
        if normalize is None:
            normalize = self.definition.merge_method in ["ties", "dare_ties", "linear"]
        return MergeConfiguration.model_validate(
            {
                "merge_method": self.definition.merge_method,
                "slices": slices,
                "models": models,
                "parameters": {
                    "normalize": normalize,
                    "int8_mask": True,
                },
                "dtype": "bfloat16",
                "base_model": self.definition.base_model,
                "tokenizer_source": self.definition.tokenizer_source,
            }
        )

    def _discrete_config_slices(
        self,
        n_layer_groups: int,
        n_param_sets: int,
        param_arrays: Dict[str, torch.Tensor],
    ) -> List[Dict]:
        """Generate merge config output slices for non-interpolated parameters."""
        slices = []
        layer_step = (
            self.definition.layer_granularity
            if self.definition.layer_granularity > 0
            else self.num_layers
        )
        for slice_idx in range(n_layer_groups):
            sources = []
            for model_idx, model in enumerate(self.definition.models):
                params = {}
                if n_param_sets > 1:
                    for param, values in param_arrays.items():
                        params[param] = []
                        for set_idx in range(n_param_sets):
                            value = values[
                                slice_idx,
                                model_idx,
                                set_idx,
                            ]
                            filter_ = (self.definition.filters + [None])[set_idx]
                            params[param].append(
                                {"filter": filter_, "value": value.item()}
                            )
                else:
                    for param, values in param_arrays.items():
                        params[param] = values[
                            slice_idx,
                            model_idx,
                            0,
                        ].item()

                sources.append(
                    {
                        "model": model,
                        "layer_range": [
                            slice_idx * layer_step,
                            (slice_idx + 1) * layer_step,
                        ],
                        "parameters": params,
                    }
                )

            if self.definition.base_model and (
                self.definition.base_model not in self.definition.models
            ):
                sources.append(
                    {
                        "model": self.definition.base_model,
                        "layer_range": [
                            slice_idx * layer_step,
                            (slice_idx + 1) * layer_step,
                        ],
                    }
                )
            slices.append({"sources": sources})
        return slices

    def _smooth_config_models(
        self, n_param_sets: int, param_arrays: Dict[str, torch.Tensor]
    ) -> List[Dict]:
        """Generate merge config model section with parameter interpolation."""
        models = []
        for model_idx, model in enumerate(self.definition.models):
            params = {}
            if n_param_sets > 1:
                for param, values in param_arrays.items():
                    params[param] = []
                    for set_idx in range(n_param_sets):
                        value = values[:, model_idx, set_idx]
                        filter_ = (self.definition.filters + [None])[set_idx]
                        params[param].append(
                            {
                                "filter": filter_,
                                "value": _unpack_single_element(value.tolist()),
                            }
                        )
            else:
                for param, values in param_arrays.items():
                    params[param] = _unpack_single_element(
                        values[:, model_idx, 0].tolist()
                    )

            models.append(
                {
                    "model": model,
                    "layer_range": [0, self.num_layers],
                    "parameters": params,
                }
            )

        if self.definition.base_model and (
            self.definition.base_model not in self.definition.models
        ):
            models.append({"model": self.definition.base_model})
        return models

    def _slerp_slices(self, genotype: torch.Tensor) -> List[Dict]:
        """Generate merge config output slices for SLERP.

        This method is a bit more complex because it requires choosing the
        two models with the highest weight for each layer group and calculating
        the interpolation parameter t. Parameter interpolation and component
        splitting are not supported because it's too hard and I don't want to.
        """
        n_layer_groups, n_models, _, _ = genotype.shape
        layer_step = (
            self.definition.layer_granularity
            if self.definition.layer_granularity > 0
            else self.num_layers
        )
        slices = []
        for slice_idx in range(n_layer_groups):
            s = {
                "sources": [
                    {
                        "model": self.definition.models[i],
                        "layer_range": [
                            slice_idx * layer_step,
                            (slice_idx + 1) * layer_step,
                        ],
                    }
                    for i in range(n_models)
                ]
            }

            # Choose the two models with the highest weight and
            # calculate the interpolation parameter t
            chosen = torch.topk(genotype[slice_idx, :, 0, 0], 2)
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

            if self.definition.base_model and (
                self.definition.base_model not in self.definition.models
            ):
                s["sources"].append(
                    {
                        "model": self.definition.base_model,
                        "layer_range": [
                            slice_idx * layer_step,
                            (slice_idx + 1) * layer_step,
                        ],
                    }
                )

            slices.append(s)
        return slices

    def _to_torch(self, genotype: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Convert a genotype to a torch tensor of the correct shape."""
        if not isinstance(genotype, torch.Tensor):
            genotype = torch.tensor(genotype)
        if len(genotype.shape) == 1:
            num_layer_groups = (
                self.num_layers // self.definition.layer_granularity
                if self.definition.layer_granularity > 0
                else 1
            )
            genotype = genotype.view(
                num_layer_groups,
                len(self.definition.models),
                len(self.definition.filters or []) + 1,
                -1,
            )

        if len(genotype.shape) != 4:
            logging.error(f"Invalid genotype shape: {genotype.shape}")
            raise InvalidGenotypeError(
                "Invalid genotype shape - must be 4D tensor or 1D array"
            )

        return genotype

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
                for set_idx, filter_ in enumerate(
                    (self.definition.filters or []) + [None]
                ):
                    suffix = ""
                    if filter_ is not None:
                        suffix = f"_{filter_}"
                    res[f"{model_name}_{param_name}{suffix}"] = genotype[
                        :, model_idx, set_idx, idx
                    ]

        return res


def _unpack_single_element(x: List) -> Any:
    if len(x) == 1:
        return x[0]
    return x
