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


from transformers import PretrainedConfig

from mergekit.architecture.base import (
    ModelArchitecture,
    ModuleArchitecture,
    ModuleConfiguredArchitecture,
    ModuleDefinition,
    WeightInfo,
)
from mergekit.architecture.decoder_only import get_decoder_only_arch


def get_architecture_info(config: PretrainedConfig) -> ModelArchitecture:
    if len(config.architectures) != 1:
        raise RuntimeError("More than one architecture in config?")
    arch_name = config.architectures[0]

    if decoder := get_decoder_only_arch(arch_name, config=config):
        return ModelArchitecture(
            modules={"decoder": ModuleDefinition(architecture=decoder)}
        )

    raise RuntimeError(f"Unsupported architecture {arch_name}")


__all__ = [
    "ModelArchitecture",
    "ModuleArchitecture",
    "ModuleDefinition",
    "ModuleConfiguredArchitecture",
    "WeightInfo",
    "get_architecture_info",
]
