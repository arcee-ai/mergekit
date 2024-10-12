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

import importlib.resources
import string
from typing import ClassVar, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from transformers import PretrainedConfig

import mergekit._data.architectures
from mergekit.architecture.base import (
    ModuleArchitecture,
    ProceduralSpaceInfo,
    WeightInfo,
)


class JSONLayerTemplates(BaseModel, frozen=True):
    weights: List[WeightInfo]
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None


class JSONArchitectureDefinition(BaseModel, frozen=True):
    expected_model_type: str = Field(alias="model_type")
    architectures: List[str]
    pre_weights: List[WeightInfo]
    layer_templates: JSONLayerTemplates
    post_weights: List[WeightInfo]
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None
    num_layers_config_key: Optional[str] = None


class TemplateWithArithmetic(string.Template):
    idpattern = r"(?a:[_a-z][_a-z0-9]*([+-]1)?)"


def _template_substitution(
    template: str, num_layers: int, layer_idx: Optional[int] = None
) -> str:
    if "{" not in template:
        return template

    substitutions = {
        "num_layers": num_layers,
        "num_layers+1": num_layers + 1,
        "num_layers-1": num_layers - 1,
    }

    if layer_idx is not None:
        substitutions.update(
            {
                "layer_index": layer_idx,
                "layer_index+1": layer_idx + 1,
                "layer_index-1": layer_idx - 1,
            }
        )

    return TemplateWithArithmetic(template).substitute(substitutions)


class JsonArchitectureInfo(ModuleArchitecture, BaseModel, frozen=True):
    definition: JSONArchitectureDefinition

    def _substitute(
        self,
        item: Union[WeightInfo, ProceduralSpaceInfo],
        config: PretrainedConfig,
        layer_idx: Optional[int] = None,
    ) -> Union[WeightInfo, ProceduralSpaceInfo]:
        num_layers = self.num_layers(config)

        obj_dict = item.model_dump(mode="json", exclude_unset=True)
        for key in obj_dict:
            if isinstance(obj_dict[key], str):
                obj_dict[key] = _template_substitution(
                    obj_dict[key], num_layers, layer_idx
                )
            elif isinstance(obj_dict[key], list):
                obj_dict[key] = [
                    (
                        _template_substitution(s, num_layers, layer_idx)
                        if isinstance(s, str)
                        else s
                    )
                    for s in obj_dict[key]
                ]
        return type(item).model_validate(obj_dict)

    def name(self) -> str:
        return self.definition.expected_model_type

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            self._substitute(wi, config=config) for wi in self.definition.pre_weights
        ]

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        return [
            self._substitute(wi, config=config, layer_idx=index)
            for wi in self.definition.layer_templates.weights
        ]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            self._substitute(wi, config=config) for wi in self.definition.post_weights
        ]

    def sliceable(self) -> bool:
        return True

    def procedural_spaces(self, config: PretrainedConfig) -> List[ProceduralSpaceInfo]:
        res = []
        for s in self.definition.procedural_spaces or []:
            res.append(self._substitute(s, config=config))
        for idx in range(self.num_layers(config)):
            for s in self.definition.layer_templates.procedural_spaces or []:
                res.append(self._substitute(s, config=config, layer_idx=idx))
        return res

    def has_defined_spaces(self) -> bool:
        if (
            self.definition.procedural_spaces
            or self.definition.layer_templates.procedural_spaces
        ):
            return True
        for wi in (
            self.definition.layer_templates.weights
            + self.definition.pre_weights
            + self.definition.post_weights
        ):
            if wi.input_space or wi.output_space:
                return True
        return False

    def num_layers_config_key(self) -> str:
        return self.definition.num_layers_config_key


class MixtralTensorNames(ModuleArchitecture, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    def name(self) -> str:
        return "mixtral"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralTensorNames(num_local_experts=config.num_local_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_INFO.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_INFO.post_weights(config)

    def num_layers_config_key(self) -> str:
        return MISTRAL_INFO.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        num_experts = self.num_local_experts
        prefix = f"model.layers.{index}"
        tensor_names = []
        for expert_idx in range(num_experts):
            for param in ("w1", "w2", "w3"):
                tensor_names.append(
                    prefix + f".block_sparse_moe.experts.{expert_idx}.{param}.weight"
                )
        tensor_names.append(prefix + ".block_sparse_moe.gate.weight")
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        for weight_info in MISTRAL_INFO.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue
            res.append(weight_info)
        return res

    def sliceable(self) -> bool:
        return True

    def has_defined_spaces(self) -> bool:
        return False


def _load_json_arch(name: str) -> JsonArchitectureInfo:
    text = importlib.resources.read_text(mergekit._data.architectures, name)
    return JsonArchitectureInfo(
        definition=JSONArchitectureDefinition.model_validate_json(text)
    )


def _load_all_architectures() -> (
    Tuple[List[JsonArchitectureInfo], Dict[str, List[JsonArchitectureInfo]]]
):
    architectures: List[JsonArchitectureInfo] = []
    for f in importlib.resources.contents(mergekit._data.architectures):
        if f.lower().endswith(".json"):
            architectures.append(_load_json_arch(f))

    name_to_arch: Dict[str, List[JsonArchitectureInfo]] = {}
    for arch_info in architectures:
        for name in arch_info.definition.architectures:
            name_to_arch[name] = name_to_arch.get(name, [])
            name_to_arch[name].append(arch_info)
    return architectures, name_to_arch


JSON_ARCHITECTURES, NAME_TO_ARCH = _load_all_architectures()
MISTRAL_INFO = _load_json_arch("mistral.json")
QWEN2_INFO = _load_json_arch("qwen2.json")


def get_decoder_only_arch(
    arch_name: str, config: PretrainedConfig
) -> Optional[ModuleArchitecture]:
    if arch_name == MixtralTensorNames.ARCHITECTURE_NAME:
        return MixtralTensorNames.from_config(config)

    if arch_name not in NAME_TO_ARCH:
        raise RuntimeError(f"Unsupported architecture {arch_name}")

    candidates = list(NAME_TO_ARCH[arch_name])
    if len(candidates) == 1:
        return candidates[0]

    for c in candidates:
        if c.definition.expected_model_type == config.model_type:
            return c
