# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)


class PassthroughMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if len(tensors) != 1:
            raise RuntimeError("Passthrough merge expects exactly one tensor")

        model, tensor = list(tensors.items())[0]
        scale = self.tensor_parameters[model].data.get("scale", None)
        if scale is not None:
            tensor = tensor * scale

        noise_scale = self.tensor_parameters[model].data.get("noise_scale", None)
        if noise_scale is not None and noise_scale != 0.0:
            noise_seed = self.tensor_parameters[model].data.get("noise_seed", 42)
            noise_generator = torch.Generator()
            if noise_seed is not None:
                noise_generator = noise_generator.manual_seed(int(noise_seed))
                print("applying noise_seed")

            print(f"Noise Generator Seed: {noise_generator.initial_seed()}")
            random_tensor = torch.empty_like(tensor).normal_(generator=noise_generator)
            noisy_tensor = random_tensor * noise_scale

            noise_variance = self.tensor_parameters[model].data.get("noise_variance", False)
            if noise_variance is not None and noise_variance != 0.0:
                noisy_tensor = noisy_tensor * (tensor.std() * noise_variance)
                print("applying noise_variance")

            tensor = tensor + noisy_tensor

            print(f"noise_scale={noise_scale}, noise_seed={noise_seed}, noise_variance={noise_variance}")

        return tensor

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class PassthroughMerge(MergeMethod):
    def name(self) -> str:
        return "passthrough"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Passthrough"

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="scale", required=False, default_value=None),
            ConfigParameterDef(name="noise_scale", required=False, default_value=None),
            ConfigParameterDef(name="noise_variance", required=False, default_value=None),
            ConfigParameterDef(name="noise_seed", required=False, default_value=None)
        ]

    def make_task(
        self,
        *,
        tensors: MergeTensorInput,
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task:
        return PassthroughMergeTask(
            gather_tensors=tensors, tensor_parameters=tensor_parameters
        )
