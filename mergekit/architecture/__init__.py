# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import TYPE_CHECKING, Optional

from transformers import PretrainedConfig

from mergekit.architecture.auto import infer_architecture_info
from mergekit.architecture.base import (
    ConfiguredModelArchitecture,
    ConfiguredModuleArchitecture,
    ModelArchitecture,
    ModuleArchitecture,
    ModuleDefinition,
    WeightInfo,
)
from mergekit.architecture.json_definitions import NAME_TO_ARCH
from mergekit.architecture.mixtral import MixtralTensorNames
from mergekit.options import MergeOptions

if TYPE_CHECKING:
    from mergekit.config import MergeConfiguration

logger = logging.getLogger(__name__)


def arch_info_for_config(config: PretrainedConfig) -> Optional[ModelArchitecture]:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")
    arch_name = config.architectures[0]

    if arch_name == MixtralTensorNames.ARCHITECTURE_NAME:
        module = MixtralTensorNames.from_config(config)
        return ModelArchitecture(
            modules={"default": ModuleDefinition(architecture=module)},
            architectures=[arch_name],
        )
    elif arch_name in NAME_TO_ARCH:
        candidates = list(NAME_TO_ARCH[arch_name])
        if len(candidates) == 1:
            return candidates[0]

        for c in candidates:
            if c.expected_model_type == config.model_type:
                return c
        logger.warning(
            f"Multiple architectures for {arch_name}, none match model type {config.model_type}"
        )

    logger.warning(f"No JSON architecture found for {arch_name}")
    return None


def get_architecture_info(
    config: "MergeConfiguration", options: MergeOptions
) -> ModelArchitecture:
    models = config.referenced_models()
    if not models:
        raise ValueError("No models referenced in config")

    model_arch_info = [
        arch_info_for_config(m.config(trust_remote_code=options.trust_remote_code))
        for m in models
    ]
    if all(arch is not None for arch in model_arch_info):
        if not options.allow_crimes and any(
            arch != model_arch_info[0] for arch in model_arch_info
        ):
            raise RuntimeError(
                "Must specify --allow-crimes to attempt to mix different architectures"
            )
        return model_arch_info[0]

    # try to infer from all models
    return infer_architecture_info(models, config.base_model, options)


__all__ = [
    "ModelArchitecture",
    "ModuleArchitecture",
    "ModuleDefinition",
    "ConfiguredModuleArchitecture",
    "ConfiguredModelArchitecture",
    "WeightInfo",
    "get_architecture_info",
    "arch_info_for_config",
]
