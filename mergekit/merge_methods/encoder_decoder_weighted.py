# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

from typing import Any, Dict, List, Optional, Tuple

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.merge_methods.base import ConfigParameterDef, MergeMethod, MergeTensorInput


class EncoderDecoderWeightedMerge(MergeMethod):
    """Merge method that applies different weights to encoder and decoder components.
    
    This method allows for separate weighting of encoder and decoder components,
    which is particularly useful for encoder-decoder architectures like Whisper
    where you might want to prioritize one component over the other.
    """

    def name(self) -> str:
        """Return the name of the merge method."""
        return "encoder_decoder_weighted"

    def pretty_name(self) -> str:
        """Return the pretty name of the merge method."""
        return "Encoder-Decoder Weighted Merge"

    def reference_url(self) -> Optional[str]:
        """Return a reference URL for the merge method."""
        return "https://github.com/arcee-ai/mergekit/blob/main/docs/methods.md#encoder-decoder-weighted-merge"

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        """Return the tensor parameters for the merge method.
        
        Returns:
            A list of parameter definitions for the merge method.
        """
        return [
            ConfigParameterDef(name="encoder_weight", required=True),
            ConfigParameterDef(name="decoder_weight", required=True),
            ConfigParameterDef(name="cross_weight", required=False),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task:
        """Create a task for merging tensors with different weights for encoder and decoder.
        
        Args:
            output_weight: Information about the weight being merged.
            tensors: The tensors to merge.
            parameters: Global parameters for the merge method.
            tensor_parameters: Parameters for each model.
            base_model: The base model, if any.
            
        Returns:
            A task for merging the tensors.
        """
        return EncoderDecoderWeightedMergeTask(
            tensors=tensors,
            tensor_parameters=tensor_parameters,
            weight_info=output_weight,
        )


class EncoderDecoderWeightedMergeTask(Task[torch.Tensor]):
    """Task for merging tensors with different weights for encoder and decoder."""
    
    tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        """Return the arguments for the task."""
        return {"tensors": self.tensors}

    def uses_accelerator(self) -> bool:
        """Return whether the task uses an accelerator."""
        return True

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights so they sum to 1.0.
        
        Args:
            weights: The weights to normalize.
            
        Returns:
            The normalized weights.
        """
        total = sum(weights)
        if total == 0:
            # Avoid division by zero
            return [1.0 / len(weights)] * len(weights)
        return [w / total for w in weights]

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        """Execute the task.

        Args:
            tensors: The tensors to merge.

        Returns:
            The merged tensor.
        """
        if not tensors:
            raise ValueError("No tensors to merge")

        # Extract weights with default values
        encoder_weights = []
        decoder_weights = []
        cross_weights = []
        
        for model in tensors.keys():
            params = self.tensor_parameters[model]
            # Use default value of 1.0 if parameter is not present or is None
            encoder_weight = params["encoder_weight"] if "encoder_weight" in params and params["encoder_weight"] is not None else 1.0
            decoder_weight = params["decoder_weight"] if "decoder_weight" in params and params["decoder_weight"] is not None else 1.0
            
            # Calculate cross_weight if not provided
            if "cross_weight" in params and params["cross_weight"] is not None:
                cross_weight = params["cross_weight"]
            else:
                cross_weight = (encoder_weight + decoder_weight) / 2
            
            encoder_weights.append(encoder_weight)
            decoder_weights.append(decoder_weight)
            cross_weights.append(cross_weight)
        
        # Normalize weights
        encoder_weights = self._normalize_weights(encoder_weights)
        decoder_weights = self._normalize_weights(decoder_weights)
        cross_weights = self._normalize_weights(cross_weights)
        
        # Determine which set of weights to use based on the tensor name
        tensor_name = self.weight_info.name
        weights = encoder_weights
        
        if "decoder" in tensor_name:
            if "encoder_attn" in tensor_name:
                weights = cross_weights
            else:
                weights = decoder_weights
        
        # Perform weighted average
        result = None
        for i, (model, tensor) in enumerate(tensors.items()):
            weight = weights[i]
            if result is None:
                result = tensor * weight
            else:
                result += tensor * weight
                
        return result
        
    def group_label(self) -> Optional[str]:
        """Return the group label for the task."""
        return self.tensors.group_label() 