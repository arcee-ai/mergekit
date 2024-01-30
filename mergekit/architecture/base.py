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
from typing import Dict, List, Optional

from pydantic import BaseModel


class WeightInfo(BaseModel):
    name: str
    is_embed: bool = False

    def prefixed_name(self, prefix: Optional[str] = None):
        if prefix:
            return prefix + self.name
        return self.name


class ModuleArchitecture(ABC):
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of layers in this module."""
        ...

    @abstractmethod
    def layer_weights(self, index: int) -> Optional[List[WeightInfo]]:
        """Return a list of all weights associated with a given layer."""
        ...

    @abstractmethod
    def pre_weights(self) -> List[WeightInfo]:
        """Return a list of all weights preceding the first layer."""
        ...

    @abstractmethod
    def post_weights(self) -> List[WeightInfo]:
        """Return a list of all weights following the final layer."""
        ...

    @abstractmethod
    def slicable(self) -> bool:
        """Return True if the architecture can be sliced meaningfully."""
        ...

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"

    def all_weights(self) -> List[str]:
        num_layers = self.num_layers()
        tensor_names = list(self.pre_weights())
        for layer_idx in range(num_layers):
            tensor_names.extend(self.layer_weights(index=layer_idx))
        tensor_names.extend(self.post_weights())
        return [ti.name if isinstance(ti, WeightInfo) else ti for ti in tensor_names]


class ModuleDefinition(BaseModel, frozen=True, arbitrary_types_allowed=True):
    architecture: ModuleArchitecture
    weight_prefix: Optional[str] = None
    config_prefix: Optional[str] = None
    subfolder: Optional[str] = None


class ModelArchitecture(BaseModel, frozen=True, arbitrary_types_allowed=True):
    modules: Dict[str, ModuleDefinition]

    def all_weights(self) -> List[str]:
        tensor_names = []
        for module in self.modules.values():
            prefix = module.weight_prefix or ""
            for name in module.architecture.all_weights():
                tensor_names.append(prefix + name)
        return tensor_names


class StaticLayeredModuleArchitecture(ModuleArchitecture, BaseModel, frozen=True):
    name: str

    pre_weight_names: List[str]
    post_weight_names: List[str]
    embed_weight_names: List[str]
    layer_prefix_format: str
    layer_weight_suffixes: List[str]
    num_layers_key: Optional[str] = None
    is_slicable: bool = True
    configured_num_layers: Optional[int] = None

    def num_layers(self) -> int:
        if not self.configured_num_layers:
            raise RuntimeError(
                "num_layers() called on module with no configured_num_layers set"
            )
        return self.configured_num_layers

    def layer_weights(self, index: int) -> Optional[List[WeightInfo]]:
        if index >= self.configured_num_layers:
            return None
        res = []
        for suffix in self.layer_weight_suffixes:
            name = self.layer_prefix_format.format(idx=index) + "." + suffix
            res.append(WeightInfo(name=name, is_embed=name in self.embed_weight_names))
        return res

    def pre_weights(self) -> List[WeightInfo]:
        return [
            WeightInfo(name=name, is_embed=name in self.embed_weight_names)
            for name in self.pre_weight_names
        ]

    def post_weights(self) -> List[WeightInfo]:
        return [
            WeightInfo(name=name, is_embed=name in self.embed_weight_names)
            for name in self.post_weight_names
        ]

    def num_layers_config_key(self) -> str:
        if self.num_layers_key:
            return self.num_layers_key
        return super().num_layers_config_key()

    def slicable(self) -> bool:
        return self.is_slicable
