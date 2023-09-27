from abc import ABC, abstractmethod
from typing import Dict, Sequence

import torch
from transformers import PretrainedConfig

from common import ModelReference
from config import ConfigReader, MergeConfiguration
from graph import RuleSet, TensorReference


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
