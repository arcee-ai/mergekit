# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

"""Graph adapter for legacy-style, tokenizer-unaware vocabulary truncation."""

from typing import Dict, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import TensorGroup
from mergekit.merge_methods.preprocessing import truncate_vocabulary


class TruncatedVocabulary(Task[Dict[ModelReference, torch.Tensor]]):
    gather_tensors: GatherTensors
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor]
    ) -> Dict[ModelReference, torch.Tensor]:
        group = truncate_vocabulary(
            TensorGroup.from_tensors(
                list(tensors.values()),
                ids=list(tensors),
                name=self.weight_info.name,
                vocabulary_axis=self.weight_info.vocabulary_axis,
            )
        )
        return {entry.id: entry.tensor for entry in group.entries}
