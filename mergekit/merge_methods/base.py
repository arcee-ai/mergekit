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

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import torch
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from mergekit.common import ModelReference
from mergekit.config import ConfigReader, MergeConfiguration
from mergekit.graph import RuleSet, TensorReference


class MergeMethod(ABC):
    @abstractmethod
    def __call__(
        self,
        parameter_name: str,
        input_tensors: Dict[TensorReference, torch.Tensor],
        config: ConfigReader,
        **kwargs
    ) -> torch.Tensor:
        ...

    def general_dependencies(self) -> Sequence[TensorReference]:
        """List any tensors necessary for *every* merge operation"""
        return []

    def input_layer_dependencies(
        self, model: ModelReference, layer_idx: int
    ) -> Sequence[TensorReference]:
        """List any tensors necessary when input includes a specific layer"""
        return []

    def add_rules(self, rules: RuleSet):
        """Add any rules necessary for execution to the set"""
        pass

    def model_out_config(self, config: MergeConfiguration) -> PretrainedConfig:
        """Return a configuration for the resulting model."""
        if config.base_model:
            return ModelReference.parse(config.base_model).config()
        return config.referenced_models()[0].config()

    def model_tokenizer(self, config: MergeConfiguration) -> PreTrainedTokenizerBase:
        """Return a tokenizer appropriate for the resulting model."""
        if config.base_model:
            path = ModelReference.parse(config.base_model).path
        else:
            path = config.referenced_models()[0].path
        return AutoTokenizer.from_pretrained(path)
