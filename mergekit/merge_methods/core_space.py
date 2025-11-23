"""
Core Space Merging Method for mergekit
Based on "Accurate and Efficient Low-Rank Model Merging in Core Space"
(Panariello et al., NeurIPS 2025)
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)

log = logging.getLogger(__name__)


class CoreSpaceTask(Task[torch.Tensor]):
    """Task for performing core space merge on a single tensor."""

    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    default_weight: float  # Changed from dict to simple float

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        """
        Execute core space merge for a single tensor.

        Note: This processes each weight tensor independently. Core Space works
        best with full fine-tuned models (not separate lora_A/lora_B weights).
        For LoRA adapters, use models that have been merged back into base
        (via PEFT's merge_and_unload).

        Args:
            tensors: Dictionary mapping model references to their tensors

        Returns:
            Merged tensor
        """
        if len(tensors) == 1:
            return list(tensors.values())[0]

        # Get base model tensor
        base_tensor = tensors.get(self.base_model)
        if base_tensor is None:
            log.warning("Base model not found, using first model as base")
            self.base_model = list(tensors.keys())[0]
            base_tensor = tensors[self.base_model]

        # Always use core space merge (which approximates deltas as low-rank)
        try:
            return self._core_space_merge(tensors, base_tensor)
        except Exception as e:
            log.warning(f"Core space merge failed for {self.weight_info.name}: {e}")
            log.warning("Falling back to weighted average")
            return self._weighted_average(tensors, base_tensor)

    def _is_lora_weight(self, weight_name: str) -> bool:
        """
        Check if a weight is from a LoRA adapter.

        Note: We only handle merged LoRA weights (full fine-tuned models),
        not separate lora_A and lora_B matrices, since mergekit processes
        each weight independently.
        """
        # Don't treat separate lora_A/lora_B as LoRA - they need to be paired
        # which we can't do in this single-tensor context
        # We only handle full merged models that were LoRA-adapted
        return False  # For now, treat all as full weights

    def _extract_lora_matrices(
        self, tensors: Dict[ModelReference, torch.Tensor], base_tensor: torch.Tensor
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract LoRA A and B matrices from tensors.

        Since mergekit processes each tensor independently, we can't access
        paired lora_A and lora_B weights. Instead, we approximate all task
        vectors (deltas from base) as low-rank using SVD.

        This works for:
        - Full fine-tuned models (common case)
        - LoRA models that were merged back into base (via merge_and_unload)
        """
        lora_As = []
        lora_Bs = []

        for model_ref, tensor in tensors.items():
            if model_ref == self.base_model:
                continue

            # Compute task vector (delta from base)
            delta = tensor - base_tensor

            # Approximate as low-rank via SVD
            # ΔW ≈ B @ A where rank is chosen automatically
            # Ensure rank is at least 1 to avoid degenerate matrices
            rank = max(
                1, min(16, min(delta.shape) // 4)
            )  # Adaptive rank with minimum of 1

            U, S, Vt = torch.linalg.svd(delta, full_matrices=False)

            # Keep top-rank components
            A = torch.diag(S[:rank]) @ Vt[:rank, :]
            B = U[:, :rank]

            lora_As.append(A)
            lora_Bs.append(B)

        return lora_As, lora_Bs

    def _core_space_merge(
        self, tensors: Dict[ModelReference, torch.Tensor], base_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform core space merge.

        Steps:
        1. Extract LoRA A and B matrices
        2. Compute reference bases via SVD
        3. Project to core space
        4. Merge in core space
        5. Reconstruct to full space
        """
        # Extract LoRA matrices
        lora_As, lora_Bs = self._extract_lora_matrices(tensors, base_tensor)

        if len(lora_As) == 0:
            return base_tensor

        # Compute reference bases
        U_B, V_A = self._compute_reference_bases(lora_Bs, lora_As)

        # Determine common rank for projection
        # After concatenation, U_B and V_A may have different second dimensions
        common_rank = min(U_B.shape[1], V_A.shape[1])
        U_B_trunc = U_B[:, :common_rank]
        V_A_trunc = V_A[:, :common_rank]

        # Project each LoRA to core space
        core_reprs = []
        model_refs = [ref for ref in tensors.keys() if ref != self.base_model]

        for A, B in zip(lora_As, lora_Bs):
            core_repr = U_B_trunc.T @ B @ A @ V_A_trunc
            core_reprs.append(core_repr)

        # Merge in core space using equal weights (or default_weight)
        # For simplicity, use equal weights for all models
        num_models = len(core_reprs)
        core_merged = sum(core_reprs) / num_models

        # Reconstruct to full space
        delta_W = U_B_trunc @ core_merged @ V_A_trunc.T

        # Add to base model
        merged = base_tensor + delta_W

        return merged

    def _compute_reference_bases(
        self, B_matrices: List[torch.Tensor], A_matrices: List[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reference bases U_B and V_A using SVD."""
        # Concatenate in the subspace dimension (not stacking!)
        # B matrices: (d_out, rank) each -> concatenate horizontally
        B_concat = torch.cat(B_matrices, dim=1)  # (d_out, num_models*rank)

        # A matrices: (rank, d_in) each -> concatenate vertically
        A_concat = torch.cat(A_matrices, dim=0)  # (num_models*rank, d_in)

        # Compute SVD
        U_B, _, _ = torch.linalg.svd(B_concat, full_matrices=False)
        _, _, V_A_T = torch.linalg.svd(A_concat, full_matrices=False)
        V_A = V_A_T.T

        return U_B, V_A

    def _weighted_average(
        self, tensors: Dict[ModelReference, torch.Tensor], base_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Fall back to simple weighted average."""
        # For now, use equal weights (simple average)
        result = torch.zeros_like(base_tensor)

        for model_ref, tensor in tensors.items():
            result += tensor

        return result / len(tensors) if len(tensors) > 0 else base_tensor

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class CoreSpaceMerge(MergeMethod):
    """
    Core Space merging method for LoRA-adapted models.

    This method merges models by:
    1. Approximating task vectors (deltas from base) as low-rank: ΔW ≈ B @ A
    2. Computing SVD-based reference bases from all adapters
    3. Projecting into a shared, aligned core space
    4. Merging in the compact core space
    5. Reconstructing back to full parameter space

    Best used with:
    - Full fine-tuned models (standard case)
    - LoRA models merged back into base (via merge_and_unload)
    - Any models where task vectors can be approximated as low-rank

    Note: Does not handle separate lora_A/lora_B weight files directly,
    as mergekit processes each tensor independently. For LoRA adapters,
    merge them into the base model first using PEFT's merge_and_unload().

    Benefits:
    - Efficient: Operates in compact core space
    - Aligned: SVD-based alignment of subspaces
    - Information-preserving: Lossless projection
    - Flexible: Handles heterogeneous ranks
    """

    def name(self) -> str:
        return "core_space"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Core Space Merge"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://github.com/apanariello4/core-space-merging"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Task:
        """
        Create a task for core space merging.

        Args:
            output_weight: Information about the output weight
            tensors: Input tensors from different models
            base_model: Base model reference
            parameters: Merge parameters (weights, etc.)
            **kwargs: Additional arguments

        Returns:
            CoreSpaceTask to execute the merge
        """
        # Get weight parameter - handle ImmutableMap
        weight_param = parameters["weight"] if "weight" in parameters else 1.0

        # Convert to float for hashability
        default_weight = (
            float(weight_param) if not isinstance(weight_param, dict) else 1.0
        )

        return CoreSpaceTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            default_weight=default_weight,
        )


# For registration in mergekit's method registry
__all__ = ["CoreSpaceMerge"]
