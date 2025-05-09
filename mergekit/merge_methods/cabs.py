# mergekit/merge_methods/cabs.py
import logging
import torch
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field, validator
from typing_extensions import override, Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)

# --- Helper function for n:m structural pruning ---
def prune_n_m_structural(
    tensor: torch.Tensor,
    n_val: int,
    m_val: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return tensor.clone(), torch.zeros_like(tensor, dtype=torch.bool)
    original_shape = tensor.shape
    device = tensor.device
    flat_tensor_orig_values = tensor.flatten().clone() 
    num_elements = flat_tensor_orig_values.numel()
    if m_val <= 0:
        logging.error(f"Tensor shape {original_shape}: m_val ({m_val}) must be positive. No pruning.")
        return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    if n_val < 0 or n_val > m_val:
        logging.error(f"Tensor shape {original_shape}: n_val ({n_val}) invalid. No pruning.")
        return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    if n_val == 0:
        return torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
    if n_val == m_val:
        return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    padding = (m_val - (num_elements % m_val)) % m_val
    if padding > 0:
        flat_tensor_padded = torch.cat(
            (flat_tensor_orig_values, torch.zeros(padding, device=device, dtype=tensor.dtype))
        )
    else:
        flat_tensor_padded = flat_tensor_orig_values
    reshaped_tensor = flat_tensor_padded.reshape(-1, m_val)
    if reshaped_tensor.numel() == 0: 
        return torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
    magnitudes = torch.abs(reshaped_tensor)
    _, top_n_indices_in_blocks = torch.topk(magnitudes, k=n_val, dim=1)
    nm_mask_blocks = torch.zeros_like(reshaped_tensor, dtype=torch.bool, device=device)
    nm_mask_blocks.scatter_(1, top_n_indices_in_blocks, True)
    nm_mask_flat_padded = nm_mask_blocks.flatten()
    if padding > 0:
        nm_mask_unpadded = nm_mask_flat_padded[:-padding]
    else:
        nm_mask_unpadded = nm_mask_flat_padded
    final_mask_reshaped = nm_mask_unpadded.reshape(original_shape)
    final_pruned_tensor = tensor * final_mask_reshaped
    return final_pruned_tensor, final_mask_reshaped

# --- Mergekit Method Definition ---
class CABSMerge(MergeMethod, BaseModel, frozen=True):
    # These fields are part of the method's configuration, settable via YAML.
    # Pydantic uses these defaults if not provided in YAML for the 'cabs' method block.
    default_n_m_ratio: Optional[Tuple[int, int]] = Field(
        default=None, 
        description="Optional global default [n, m] ratio for n:m pruning. E.g., [1, 4]."
    )
    pruning_order: Optional[List[str]] = Field(
        default=None,
        description="Optional: List of model source names (from YAML 'sources') defining the CA processing order."
    )
    
    # These are more like fixed properties of the method, not typically changed by user YAML for 'cabs'
    # but Pydantic treats them as fields that can be initialized.
    # Mergekit's MergeMethod.create will pass YAML params, potentially overriding these if keys match.
    # It's safer to have these as fixed return values in name(), pretty_name(), etc. if they are truly static.
    # However, to allow Mergekit's create(**kwargs) to work seamlessly if it tries to pass them,
    # we keep them as fields with defaults.
    method_name_override: Optional[str] = Field(default=None, exclude=True) # For internal use if variants are registered
    method_pretty_name_override: Optional[str] = Field(default=None, exclude=True)
    method_reference_url_override: Optional[str] = Field(default=None, exclude=True)


    @validator('default_n_m_ratio', pre=True, always=True)
    def check_default_n_m_ratio(cls, v: Any) -> Optional[Tuple[int, int]]:
        if v is not None:
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and
                    isinstance(v[0], int) and isinstance(v[1], int) and
                    0 <= v[0] <= v[1] and v[1] > 0):
                raise ValueError(
                    "default_n_m_ratio must be a tuple/list of two integers [n, m] "
                    "with 0 <= n <= m and m > 0, or null."
                )
            return tuple(v)
        return None

    @override
    def name(self) -> str:
        return self.method_name_override or "cabs"

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name_override or "Conflict-Aware and Balanced Sparsification"
    
    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url_override or "https://arxiv.org/abs/2503.01874"

    @override
    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="default_n_m_ratio", type="list[int]", required=False, default_value=None,
                description="Optional global default [n, m] ratio. Models can override this. Example: [1, 4]"
            ),
            ConfigParameterDef(
                name="pruning_order", type="list[str]", required=False, default_value=None,
                description="Optional: List of model source names (from YAML 'sources') defining the CA processing order."
            ),
            # These are not typically set by users for the primary "cabs" method, but allow for variants if needed.
            # ConfigParameterDef(name="method_name_override", type="str", required=False, advanced=True), 
            # ConfigParameterDef(name="method_pretty_name_override", type="str", required=False, advanced=True),
            # ConfigParameterDef(name="method_reference_url_override", type="str", required=False, advanced=True),
        ]

    @override
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="weight", type="float", required=False, default_value=1.0,
                description="Scaling coefficient (lambda) for this model's task vector."
            ),
            ConfigParameterDef(
                name="n_m_ratio", type="list[int]", required=False, default_value=None,
                description="Per-model [n, m] ratio for n:m pruning. Overrides global default_n_m_ratio. Example: [1, 2]"
            ),
        ]
    
    @override
    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any], # parameters from YAML for THIS method invocation
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        if base_model is None:
            logging.error("CABS merge requires a 'base_model' to be specified in the YAML.")
            raise ValueError("CABS merge requires a 'base_model'.")
        
        # 'self' is the instance from STATIC_MERGE_METHODS.
        # 'parameters' contains the YAML overrides for default_n_m_ratio, pruning_order, etc.
        # We should use 'parameters' to construct the CABSTask or configure the instance.
        # Mergekit's MergeMethod.create typically handles creating a new instance with YAML params.
        # So, self.default_n_m_ratio etc. on *this specific 'self'* instance will be the final ones.

        return CABSTask(
            method_config=self, # 'self' is the correctly configured CABSMerge instance
            tensors_input=tensors,
            base_model_ref=base_model,
            current_weight_info=output_weight,
            per_model_tensor_params=tensor_parameters,
        )

class CABSTask(Task[torch.Tensor]):
    method_config: CABSMerge 
    tensors_input: MergeTensorInput
    base_model_ref: ModelReference
    current_weight_info: WeightInfo
    per_model_tensor_params: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    
    FALLBACK_N_M_RATIO: Tuple[int, int] = (1, 4)

    @override
    def uses_accelerator(self) -> bool:
        return True

    @override
    def arguments(self) -> Dict[str, Task]:
        return {"tensors_arg": self.tensors_input}

    def _resolve_nm_ratio_for_model(self, model_ref: ModelReference) -> Tuple[int, int]:
        current_model_params = self.per_model_tensor_params.get(model_ref, ImmutableMap({}))
        per_model_nm_ratio_raw = current_model_params.get("n_m_ratio")

        if per_model_nm_ratio_raw is not None:
            if not (isinstance(per_model_nm_ratio_raw, (list, tuple)) and len(per_model_nm_ratio_raw) == 2 and
                    isinstance(per_model_nm_ratio_raw[0], int) and isinstance(per_model_nm_ratio_raw[1], int) and
                    0 <= per_model_nm_ratio_raw[0] <= per_model_nm_ratio_raw[1] and per_model_nm_ratio_raw[1] > 0):
                logging.warning(f"Invalid n_m_ratio {per_model_nm_ratio_raw} for model {model_ref.name} "
                                f"on tensor {self.current_weight_info.name}. "
                                f"Falling back.")
            else:
                return int(per_model_nm_ratio_raw[0]), int(per_model_nm_ratio_raw[1])

        if self.method_config.default_n_m_ratio is not None: # Use from configured instance
            return self.method_config.default_n_m_ratio

        logging.warning(f"No n_m_ratio specified for model {model_ref.name} and no global default "
                        f"for tensor {self.current_weight_info.name}. "
                        f"Using hardcoded fallback: {self.FALLBACK_N_M_RATIO}.")
        return self.FALLBACK_N_M_RATIO

    @override
    def execute(
        self,
        tensors_arg: Dict[ModelReference, torch.Tensor], 
        **_kwargs,
    ) -> torch.Tensor:
        if self.base_model_ref not in tensors_arg:
            logging.error(f"Base model '{self.base_model_ref.name}' tensor not found for weight '{self.current_weight_info.name}'.")
            device_str = self.current_weight_info.device_str() if hasattr(self.current_weight_info, 'device_str') else 'cpu'
            dtype_val = self.current_weight_info.dtype if hasattr(self.current_weight_info, 'dtype') else torch.float32
            return torch.empty(0, device=torch.device(device_str), dtype=dtype_val)

        target_device = tensors_arg[self.base_model_ref].device
        target_dtype = tensors_arg[self.base_model_ref].dtype
        merged_tensor_accumulator = tensors_arg[self.base_model_ref].clone().to(device=target_device, dtype=target_dtype)
        
        ordered_model_refs_for_ca: List[ModelReference] = []
        model_ref_by_name_map: Dict[str, ModelReference] = { ref.name: ref for ref in tensors_arg.keys() }
        
        current_pruning_order_names = self.method_config.pruning_order # Get from configured instance
        if current_pruning_order_names:
            for name in current_pruning_order_names:
                if name == self.base_model_ref.name: 
                    continue
                if name in model_ref_by_name_map:
                    ordered_model_refs_for_ca.append(model_ref_by_name_map[name])
                else:
                    logging.warning(f"Model '{name}' from pruning_order not found among available tensors "
                                    f"for weight '{self.current_weight_info.name}', skipping this entry in order.")
        else:
            sorted_non_base_names = sorted([ref.name for ref in tensors_arg.keys() if ref != self.base_model_ref])
            for name in sorted_non_base_names:
                 if name in model_ref_by_name_map: 
                    ordered_model_refs_for_ca.append(model_ref_by_name_map[name])
        
        if not ordered_model_refs_for_ca:
            logging.info(f"No non-base models to merge for weight '{self.current_weight_info.name}'. "
                         "Returning base tensor.")
            return merged_tensor_accumulator

        cumulative_param_mask = torch.zeros_like(merged_tensor_accumulator, dtype=torch.bool, device=target_device)
        
        for model_ref_current in ordered_model_refs_for_ca:
            if model_ref_current not in tensors_arg: 
                logging.warning(f"Tensor for model '{model_ref_current.name}' became unavailable during processing "
                                f"for weight '{self.current_weight_info.name}', skipping.")
                continue

            fine_tuned_tensor_val = tensors_arg[model_ref_current].to(device=target_device, dtype=target_dtype)
            base_tensor_for_diff = tensors_arg[self.base_model_ref].to(device=target_device, dtype=target_dtype)

            current_model_params_map = self.per_model_tensor_params.get(model_ref_current, ImmutableMap({}))
            scaling_coefficient = float(current_model_params_map.get("weight", 1.0))
            n_val_current, m_val_current = self._resolve_nm_ratio_for_model(model_ref_current)

            task_vector_val = fine_tuned_tensor_val - base_tensor_for_diff
            available_params_mask = ~cumulative_param_mask
            candidate_task_vector = task_vector_val * available_params_mask.to(task_vector_val.dtype)
            
            pruned_task_vector, newly_retained_mask = prune_n_m_structural(
                candidate_task_vector, 
                n_val_current, 
                m_val_current
            )
            
            merged_tensor_accumulator += scaling_coefficient * pruned_task_vector.to(merged_tensor_accumulator.dtype)
            cumulative_param_mask = torch.logical_or(
                cumulative_param_mask, 
                newly_retained_mask.to(device=cumulative_param_mask.device) 
            )
            
        return merged_tensor_accumulator

    @override
    def group_label(self) -> Optional[str]:
        return self.current_weight_info.name
