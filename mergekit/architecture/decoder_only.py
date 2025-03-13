# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import string
from typing import List, Optional

from pydantic import BaseModel, Field
from transformers import PretrainedConfig
from typing_extensions import Literal

from mergekit.architecture.base import (
    ModuleArchitecture,
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

    def sliceable(self) -> bool:
        return True

    def num_layers_config_key(self) -> str:
        return self.definition.num_layers_config_key

    def num_layers(self, config):
        if self.definition.override_num_layers is not None:
            return self.definition.override_num_layers
        return super().num_layers(config)
