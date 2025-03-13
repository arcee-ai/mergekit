# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1


import importlib
import importlib.resources
import json
from typing import Dict, List, Tuple

import mergekit._data.architectures
from mergekit.architecture.base import (
    ModelArchitecture,
    ModuleDefinition,
)
from mergekit.architecture.decoder_only import JsonModuleArchitecture, JsonModuleArchDef
from mergekit.architecture.multimodule import (
    parse_modular_architecture_definition,
)


def _load_architecture_json(name: str) -> ModelArchitecture:
    with importlib.resources.open_text(mergekit._data.architectures, name) as f:
        text = f.read()
    data = json.loads(text)
    kind = data.get("kind", "module")
    if kind == "modular":
        return parse_modular_architecture_definition(text)
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
    for f in importlib.resources.contents(mergekit._data.architectures):
        if f.lower().endswith(".json"):
            architectures.append(_load_architecture_json(f))

    name_to_arch: Dict[str, List[JsonModuleArchitecture]] = {}
    for arch_info in architectures:
        for arch_name in arch_info.architectures:
            name_to_arch[arch_name] = name_to_arch.get(arch_name, [])
            name_to_arch[arch_name].append(arch_info)
    return architectures, name_to_arch


JSON_ARCHITECTURES, NAME_TO_ARCH = _load_all_architectures()
