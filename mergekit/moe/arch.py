# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

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
