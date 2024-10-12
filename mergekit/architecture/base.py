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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel
from transformers import PretrainedConfig
from typing_extensions import Literal


class WeightInfo(BaseModel, frozen=True):
    """Information about an individual weight tensor in a model.

    Attributes:
        name (str):
            The name of the tensor representing the weight.
        is_embed (bool):
            Indicates whether the weight is for an embedding or language model head.
        input_space (Optional[str]):
            The name of the input space associated with the weight, if applicable.
        output_space (Optional[str]):
            The name of the output space associated with the weight, if applicable.
        optional (bool):
            Indicates whether the weight can be omitted from a model.
        aliases (Optional[List[str]]):
            List of alternative names for the weight, if applicable.
        force_dtype (Optional[str]):
            Mandatory dtype for the weight, if applicable.
    """

    name: str
    is_embed: bool = False
    input_space: Optional[str] = None
    output_space: Optional[str] = None
    optional: bool = False
    aliases: Optional[Tuple[str, ...]] = None
    force_dtype: Optional[str] = None
    head_split: Literal[None, "input", "output"] = None
    is_kq: Optional[bool] = False


class ProceduralSpaceInfo(BaseModel, frozen=True):
    """Defines a procedural space computed from one or more other spaces.

    Currently only supports residual connections.

    Attributes:
        name (str): The name of the space defined.
        type (str): The type of procedural space.
        inputs (List[str]): List of names of spaces used to define this space."""

    name: str
    type: Literal["residual"]
    inputs: List[str]


def _prefix_weight(weight: WeightInfo, prefix: Optional[str] = None) -> WeightInfo:
    if prefix is None:
        return weight
    return WeightInfo(
        name=prefix + weight.name,
        **weight.model_dump(exclude={"name"}),
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

    @abstractmethod
    def sliceable(self) -> bool:
        """
        Return True if the layers of this architecture can be meaningfully sliced.
        """
        ...

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"

    def num_layers(self, config: PretrainedConfig) -> int:
        """Return the number of layers in a model."""
        return getattr(config, self.num_layers_config_key())

    def all_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return all weights associated with a model."""
        num_layers = self.num_layers(config)
        res = list(self.pre_weights(config))
        for layer_idx in range(num_layers):
            res.extend(self.layer_weights(layer_idx, config))
        res.extend(self.post_weights(config))
        return res

    def procedural_spaces(self, config: PretrainedConfig) -> List[ProceduralSpaceInfo]:
        """Return a list of all procedurally defined spaces in a model."""
        return []

    def has_defined_spaces(self) -> bool:
        """
        Return True if this architecture defines space information needed for
        matching-based merge methods.
        """
        return False


class ModuleConfiguredArchitecture(
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

    def procedural_spaces(self) -> List[ProceduralSpaceInfo]:
        return self.info.procedural_spaces(self.config)

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

    def all_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        res = []
        for module in self.modules.values():
            for weight_info in module.architecture.all_weights(config=config):
                res.append(_prefix_weight(weight_info, module.weight_prefix))
        return res
