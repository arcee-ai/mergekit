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
from typing import List, Optional

import torch

from mergekit.moe.config import MoEMergeConfig
from mergekit.options import MergeOptions


class MoEOutputArchitecture(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the architecture."""
        pass

    @abstractmethod
    def supports_config(
        self,
        config: MoEMergeConfig,
        explain: bool = False,
        trust_remote_code: bool = False,
    ) -> bool:
        """Return whether this architecture supports the given config.

        If `explain` is True, log an explanation of why the config is not supported."""
        pass

    @abstractmethod
    def write_model(
        self,
        out_path: str,
        config: MoEMergeConfig,
        merge_options: MergeOptions,
        router_weights: List[torch.Tensor],
        shared_router_weights: Optional[List[torch.Tensor]] = None,
    ):
        """Write the config and tensors for the output MoE to the given path."""
        pass
