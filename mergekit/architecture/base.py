# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from transformers import PretrainedConfig

from mergekit.common import get_config_value


class WeightInfo(BaseModel, frozen=True):
    """Information about an individual weight tensor in a model.

    Attributes:
        name (str):
            The name of the tensor representing the weight.
        is_embed (bool):
            Indicates whether the weight is for an embedding or language model head.
        optional (bool):
            Indicates whether the weight can be omitted from a model.
        aliases (Optional[List[str]]):
            List of alternative names for the weight, if applicable.
        force_dtype (Optional[str]):
            Mandatory dtype for the weight, if applicable.
    """

    name: str
    is_embed: bool = False
    optional: bool = False
    aliases: Optional[Tuple[str, ...]] = None
    force_dtype: Optional[str] = None
    tied_names: Optional[Tuple[str, ...]] = None


def _prefix_weight(weight: WeightInfo, prefix: Optional[str] = None) -> WeightInfo:
    if prefix is None:
        return weight
    return WeightInfo(
        name=prefix + weight.name,
        aliases=tuple(prefix + alias for alias in weight.aliases or ()) or None,
        tied_names=tuple(prefix + tied_name for tied_name in weight.tied_names or ())
        or None,
        **weight.model_dump(exclude={"name", "aliases", "tied_names"}),
    )


class ModuleArchitecture(ABC):
    @abstractmethod
    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return a list of all weights preceding the first layer."""
        ...

    @abstractmethod
    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return a list of all weights following the final layer."""
        ...

    @abstractmethod
    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        """Return a list of all weights associated with a given layer."""
        ...

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"

    def num_layers(self, config: PretrainedConfig) -> int:
        """Return the number of layers in a model."""
        return get_config_value(config, self.num_layers_config_key())

    def all_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return all weights associated with a model."""
        num_layers = self.num_layers(config)
        res = list(self.pre_weights(config))
        for layer_idx in range(num_layers):
            res.extend(self.layer_weights(layer_idx, config))
        res.extend(self.post_weights(config))
        return res


class ConfiguredModuleArchitecture(
    BaseModel, frozen=True, arbitrary_types_allowed=True
):
    info: ModuleArchitecture
    config: PretrainedConfig
    weight_prefix: Optional[str] = None

    def num_layers(self) -> int:
        return self.info.num_layers(self.config)

    def pre_weights(self) -> List[WeightInfo]:
        return [
            _prefix_weight(w, self.weight_prefix)
            for w in self.info.pre_weights(self.config)
        ]

    def post_weights(self) -> List[WeightInfo]:
        return [
            _prefix_weight(w, self.weight_prefix)
            for w in self.info.post_weights(self.config)
        ]

    def layer_weights(self, index: int) -> List[WeightInfo]:
        return [
            _prefix_weight(w, self.weight_prefix)
            for w in self.info.layer_weights(index, self.config)
        ]

    def all_weights(self) -> List[WeightInfo]:
        return [
            _prefix_weight(w, self.weight_prefix)
            for w in self.info.all_weights(self.config)
        ]


class ModuleDefinition(BaseModel, frozen=True, arbitrary_types_allowed=True):
    architecture: ModuleArchitecture
    weight_prefix: Optional[str] = None
    subfolder: Optional[str] = None


class ModelArchitecture(BaseModel, frozen=True):
    modules: Dict[str, ModuleDefinition]
    architectures: List[str]
    expected_model_type: str = Field(alias="model_type")
    tagalong_files: Optional[List[str]] = None
    vocab_size_config_key: Optional[str] = None

    def all_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        res = []
        for module in self.modules.values():
            for weight_info in module.architecture.all_weights(config=config):
                res.append(_prefix_weight(weight_info, module.weight_prefix))
        return res


class ConfiguredModelArchitecture(BaseModel, frozen=True, arbitrary_types_allowed=True):
    info: ModelArchitecture
    config: PretrainedConfig

    def all_weights(self) -> List[WeightInfo]:
        return self.info.all_weights(self.config)

    def get_module(self, module_name: str) -> ConfiguredModuleArchitecture:
        return ConfiguredModuleArchitecture(
            info=self.info.modules[module_name].architecture,
            config=self.config,
            weight_prefix=self.info.modules[module_name].weight_prefix,
        )
