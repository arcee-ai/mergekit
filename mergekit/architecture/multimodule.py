# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Dict, List, Optional
from typing_extensions import Literal

from pydantic import BaseModel, Field

from mergekit.architecture.base import (
    ModuleDefinition,
    ModelArchitecture,
)
from mergekit.architecture.decoder_only import JsonModuleArchDef, JsonModuleArchitecture


class JsonModuleDefinition(BaseModel, frozen=True):
    architecture: JsonModuleArchDef
    weight_prefix: Optional[str] = None
    subfolder: Optional[str] = None


class JsonModularArchitectureDefinition(BaseModel, frozen=True):
    kind: Literal["modular"]
    modules: Dict[str, JsonModuleDefinition]
    architectures: List[str]
    expected_model_type: str = Field(alias="model_type")


def parse_modular_architecture_definition(
    text: str,
) -> JsonModularArchitectureDefinition:
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
    )
