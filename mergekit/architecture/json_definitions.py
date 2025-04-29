# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import importlib
import importlib.resources
import json
import string
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from transformers import PretrainedConfig
from typing_extensions import Literal

import mergekit._data.architectures
from mergekit.architecture.base import (
    ModelArchitecture,
    ModuleArchitecture,
    ModuleDefinition,
    WeightInfo,
)


class JsonLayerTemplates(BaseModel, frozen=True):
    weights: List[WeightInfo]


class JsonModuleArchDef(BaseModel, frozen=True):
    expected_model_type: str = Field(alias="model_type")
    architectures: List[str]
    pre_weights: List[WeightInfo]
    layer_templates: JsonLayerTemplates
    post_weights: List[WeightInfo]
    num_layers_config_key: Optional[str] = None
    override_num_layers: Optional[int] = None


class JsonModuleArchitecture(ModuleArchitecture, BaseModel, frozen=True):
    kind: Literal["module"] = "module"
    definition: JsonModuleArchDef

    def _substitute(
        self,
        item: WeightInfo,
        config: PretrainedConfig,
        layer_idx: Optional[int] = None,
    ) -> WeightInfo:
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

    def num_layers_config_key(self) -> str:
        return self.definition.num_layers_config_key

    def num_layers(self, config):
        if self.definition.override_num_layers is not None:
            return self.definition.override_num_layers
        return super().num_layers(config)


class JsonModuleDefinition(BaseModel, frozen=True):
    architecture: JsonModuleArchDef
    weight_prefix: Optional[str] = None
    subfolder: Optional[str] = None


class JsonModularArchitectureDefinition(BaseModel, frozen=True):
    kind: Literal["modular"]
    modules: Dict[str, JsonModuleDefinition]
    architectures: List[str]
    expected_model_type: str = Field(alias="model_type")
    tagalong_files: Optional[List[str]] = None
    vocab_size_config_key: Optional[str] = None


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


def _load_architecture_json(text: str) -> ModelArchitecture:
    data = json.loads(text)
    kind = data.get("kind", "module")
    if kind == "modular":
        parsed = JsonModularArchitectureDefinition.model_validate_json(text)
        return ModelArchitecture(
            modules={
                k: ModuleDefinition(
                    architecture=JsonModuleArchitecture(definition=v.architecture),
                    weight_prefix=v.weight_prefix,
                    subfolder=v.subfolder,
                )
                for k, v in parsed.modules.items()
            },
            architectures=parsed.architectures,
            model_type=parsed.expected_model_type,
            tagalong_files=parsed.tagalong_files,
            vocab_size_config_key=parsed.vocab_size_config_key,
        )
    elif data.get("kind", "module") == "module":
        module = JsonModuleArchitecture(
            definition=JsonModuleArchDef.model_validate(data)
        )
        return ModelArchitecture(
            modules={"default": ModuleDefinition(architecture=module)},
            architectures=module.definition.architectures,
            model_type=module.definition.expected_model_type,
        )
    else:
        raise RuntimeError(f"Unexpected architecture kind: {data['kind']}")


def _load_all_architectures() -> (
    Tuple[List[ModelArchitecture], Dict[str, List[ModelArchitecture]]]
):
    architectures: List[ModelArchitecture] = []
    for f in importlib.resources.files(mergekit._data.architectures).iterdir():
        if f.is_file() and f.name.lower().endswith(".json"):
            text = f.read_text()
            architectures.append(_load_architecture_json(text))

    name_to_arch: Dict[str, List[JsonModuleArchitecture]] = {}
    for arch_info in architectures:
        for arch_name in arch_info.architectures:
            name_to_arch[arch_name] = name_to_arch.get(arch_name, [])
            name_to_arch[arch_name].append(arch_info)
    return architectures, name_to_arch


JSON_ARCHITECTURES, NAME_TO_ARCH = _load_all_architectures()
