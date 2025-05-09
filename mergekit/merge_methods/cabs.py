import logging
import torch
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field, validator, root_validator # Keep BaseModel for CABSMerge for now if registry expects instance
from typing_extensions import override, Literal

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)

# --- Helper function for n:m structural pruning (remains the same) ---
def prune_n_m_structural(
    tensor: torch.Tensor,
    n_val: int,
    m_val: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return tensor.clone(), torch.zeros_like(tensor, dtype=torch.bool)
    original_shape = tensor.shape; device = tensor.device
    flat_tensor_orig_values = tensor.flatten().clone(); num_elements = flat_tensor_orig_values.numel()
    if m_val <= 0: logging.error(f"Tensor {original_shape}: m_val ({m_val}) must be positive."); return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    if n_val < 0 or n_val > m_val: logging.error(f"Tensor {original_shape}: n_val ({n_val}) invalid."); return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    if n_val == 0: return torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
    if n_val == m_val: return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    padding = (m_val-(num_elements % m_val))%m_val
    if padding > 0: flat_tensor_padded = torch.cat((flat_tensor_orig_values, torch.zeros(padding,device=device,dtype=tensor.dtype)))
    else: flat_tensor_padded = flat_tensor_orig_values
    reshaped_tensor = flat_tensor_padded.reshape(-1, m_val)
    if reshaped_tensor.numel()==0: return torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
    magnitudes = torch.abs(reshaped_tensor); _,top_n_indices_in_blocks = torch.topk(magnitudes,k=n_val,dim=1)
    nm_mask_blocks = torch.zeros_like(reshaped_tensor,dtype=torch.bool,device=device); nm_mask_blocks.scatter_(1,top_n_indices_in_blocks,True)
    nm_mask_flat_padded = nm_mask_blocks.flatten()
    if padding > 0: nm_mask_unpadded = nm_mask_flat_padded[:-padding]
    else: nm_mask_unpadded = nm_mask_flat_padded
    final_mask_reshaped = nm_mask_unpadded.reshape(original_shape); final_pruned_tensor = tensor * final_mask_reshaped
    return final_pruned_tensor, final_mask_reshaped

# --- Mergekit Method Definition ---
# We can keep CABSMerge as a Pydantic BaseModel if Mergekit's registry.py instantiates it directly
# and then MergeMethod.create re-instantiates with YAML parameters.
# Or, if registry.py stores the *class* and MergeMethod.create instantiates it once with YAML params,
# then it also works. Let's assume the latter for now for parameter passing simplicity.
class CABSMerge(MergeMethod, BaseModel, frozen=True): 
    # These fields capture parameters from YAML that are sibling to 'merge_method: cabs'
    # They are used if Mergekit passes them directly to CABSMerge constructor.
    # If parameters are *only* from the nested 'parameters:' block, these can be removed,
    # and CABSMerge becomes a simpler class just holding name/pretty_name.
    # For consistency with how other methods might receive their top-level params via kwargs to __init__
    # by MergeMethod.create, we define them here.
    default_n_val: Optional[int] = Field(default=None)
    default_m_val: Optional[int] = Field(default=None)
    pruning_order: Optional[List[str]] = Field(default=None)

    method_name_arg: str = Field("cabs", alias="method_name", exclude=True) 
    method_pretty_name_arg: Optional[str] = Field("Conflict-Aware N:M Sparsification", alias="method_pretty_name", exclude=True)
    method_reference_url_arg: Optional[str] = Field("https://arxiv.org/abs/2503.01874", alias="method_reference_url", exclude=True)

    @root_validator(pre=False, skip_on_failure=True)
    def check_default_n_m_consistency(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        n = values.get('default_n_val') # Get from the instance's fields
        m = values.get('default_m_val')
        if n is not None and m is None:
            raise ValueError("If 'default_n_val' is provided, 'default_m_val' must also be provided.")
        if m is not None and n is None:
            raise ValueError("If 'default_m_val' is provided, 'default_n_val' must also be provided.")
        if n is not None and m is not None:
            if not (isinstance(n, int) and n >= 0 and isinstance(m, int) and m > 0 and n <= m):
                raise ValueError(f"Invalid default n/m values: n={n}, m={m}. Ensure 0 <= n <= m and m > 0.")
        return values

    @override
    def name(self) -> str:
        return self.method_name_arg

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name_arg
    
    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url_arg

    @override
    def parameters(self) -> List[ConfigParameterDef]:
        # These declare what keys are expected in the YAML block for this method's global config.
        # If these keys are siblings to 'merge_method: cabs', Mergekit passes them to __init__.
        # If these keys are under a nested 'parameters:' block, Mergekit passes that block as
        # the 'parameters' argument to make_task.
        # Given KarcherMerge example, they are for the nested 'parameters:' block.
        return [
            ConfigParameterDef(name="default_n_val", type="int", required=False, default_value=None),
            ConfigParameterDef(name="default_m_val", type="int", required=False, default_value=None),
            ConfigParameterDef(name="pruning_order", type="list[str]", required=False, default_value=None),
        ]

    @override
    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", type="float", required=False, default_value=1.0),
            ConfigParameterDef(name="n_val", type="int", required=False, default_value=None),
            ConfigParameterDef(name="m_val", type="int", required=False, default_value=None),
        ]
    
    @override
    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference], 
        parameters: ImmutableMap[str, Any], # This map IS from the nested 'parameters:' block in YAML
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        if base_model is None:
            raise ValueError("CABS merge requires a 'base_model'.")
        
        # Extract global CABS parameters SAFELY from the 'parameters' ImmutableMap
        default_n_val_global: Optional[int] = None
        if "default_n_val" in parameters:
            val = parameters["default_n_val"]
            if isinstance(val, int): default_n_val_global = val
            elif val is not None: logging.warning(f"Expected int for default_n_val, got {type(val)}")
        
        default_m_val_global: Optional[int] = None
        if "default_m_val" in parameters:
            val = parameters["default_m_val"]
            if isinstance(val, int): default_m_val_global = val
            elif val is not None: logging.warning(f"Expected int for default_m_val, got {type(val)}")

        pruning_order_global: Optional[List[str]] = None
        if "pruning_order" in parameters:
            val = parameters["pruning_order"]
            if isinstance(val, list) and all(isinstance(s, str) for s in val):
                pruning_order_global = val
            elif val is not None:
                logging.warning(f"Expected list of strings for pruning_order, got {type(val)}")

        # Validation for n and m consistency if both are provided globally
        if default_n_val_global is not None and default_m_val_global is None:
            raise ValueError("If 'default_n_val' is provided in global parameters, 'default_m_val' must also be provided.")
        if default_m_val_global is not None and default_n_val_global is None:
            raise ValueError("If 'default_m_val' is provided in global parameters, 'default_n_val' must also be provided.")
        if default_n_val_global is not None and default_m_val_global is not None: # Both are provided
            if not (default_n_val_global >= 0 and default_m_val_global > 0 and default_n_val_global <= default_m_val_global):
                raise ValueError(f"Invalid global default n/m values: n={default_n_val_global}, m={default_m_val_global}. "
                                 "Ensure 0 <= n <= m and m > 0.")
        
        return CABSTask(
            global_default_n_val=default_n_val_global,
            global_default_m_val=default_m_val_global,
            global_pruning_order=pruning_order_global,
            tensors_input=tensors,
            base_model_ref=base_model, 
            current_weight_info=output_weight,
            per_model_tensor_params=tensor_parameters,
        )

class CABSTask(Task[torch.Tensor]):
    global_default_n_val: Optional[int]
    global_default_m_val: Optional[int]
    global_pruning_order: Optional[List[str]]
    
    tensors_input: MergeTensorInput
    base_model_ref: ModelReference 
    current_weight_info: WeightInfo
    per_model_tensor_params: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    
    FALLBACK_N_VAL: int = 1 
    FALLBACK_M_VAL: int = 4

    @override
    def uses_accelerator(self) -> bool: return True
    @override
    def arguments(self) -> Dict[str, Task]: return {"tensors_arg": self.tensors_input}

# In CABSTask class, within cabs.py

    def _resolve_n_and_m_for_model(self, model_ref: ModelReference) -> Tuple[int, int]:
        per_model_n_raw: Any = None # Use Any to catch the actual type passed
        per_model_m_raw: Any = None
        
        model_identifier_str = str(model_ref.model)
        logging.debug(f"Resolving n and m for model: {model_identifier_str}")

        if model_ref in self.per_model_tensor_params:
            current_model_inner_params_map = self.per_model_tensor_params[model_ref]
            logging.debug(f"  Per-model params for {model_identifier_str}: {dict(current_model_inner_params_map)}")
            
            if "n_val" in current_model_inner_params_map:
                per_model_n_raw = current_model_inner_params_map["n_val"]
                logging.debug(f"    Raw per-model n_val: {per_model_n_raw} (type: {type(per_model_n_raw)})")
            if "m_val" in current_model_inner_params_map:
                per_model_m_raw = current_model_inner_params_map["m_val"]
                logging.debug(f"    Raw per-model m_val: {per_model_m_raw} (type: {type(per_model_m_raw)})")

        # Attempt to convert to int if they are floats representing whole numbers
        def try_convert_to_int(val: Any, name: str) -> Optional[int]:
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                if val.is_integer():
                    return int(val)
                else:
                    logging.warning(f"    Cannot convert per-model {name} '{val}' to int as it's a non-whole float.")
                    return None 
            if val is not None: # Log if it's some other unexpected type
                 logging.warning(f"    Unexpected type for per-model {name}: {type(val)}. Expected int or float representing int.")
            return None

        n_candidate: Optional[int] = None
        m_candidate: Optional[int] = None

        if per_model_n_raw is not None:
            n_candidate = try_convert_to_int(per_model_n_raw, "n_val")
        if per_model_m_raw is not None:
            m_candidate = try_convert_to_int(per_model_m_raw, "m_val")

        # Check if per-model n and m are consistently provided and valid AFTER conversion attempt
        if n_candidate is not None and m_candidate is not None:
            if n_candidate >= 0 and m_candidate > 0 and n_candidate <= m_candidate:
                logging.debug(f"    Using per-model n_val={n_candidate}, m_val={m_candidate} for {model_identifier_str}")
                return n_candidate, m_candidate
            else:
                logging.warning(f"    Invalid per-model n_val/m_val after conversion: n={n_candidate}, m={m_candidate} "
                                f"for model {model_identifier_str}. Will try global defaults.")
        # If only one was provided or conversion failed for one, it's an incomplete/invalid pair
        elif n_candidate is not None or m_candidate is not None: 
            logging.warning(f"    Incomplete or invalid per-model n_val/m_val after conversion: n={n_candidate}, m={m_candidate} "
                            f"for model {model_identifier_str}. Both valid integers are required if one is set. Will try global defaults.")
        
        # Try global default parameters if per-model not valid or not fully set
        if self.global_default_n_val is not None and self.global_default_m_val is not None:
            # Global defaults already validated by CABSMerge.make_task
            logging.debug(f"    Using global default_n_val={self.global_default_n_val}, "
                          f"default_m_val={self.global_default_m_val} for {model_identifier_str}")
            return self.global_default_n_val, self.global_default_m_val

        logging.warning(f"    No valid per-model or global default n/m values for model {model_identifier_str} "
                        f"on tensor {self.current_weight_info.name}. "
                        f"Using hardcoded fallback: n={self.FALLBACK_N_VAL}, m={self.FALLBACK_M_VAL}.")
        return self.FALLBACK_N_VAL, self.FALLBACK_M_VAL

    @override
    def execute(self, tensors_arg: Dict[ModelReference, torch.Tensor], **_kwargs) -> torch.Tensor:
        base_model_identifier_str = str(self.base_model_ref.model)
        if self.base_model_ref not in tensors_arg:
            logging.error(f"Base model '{base_model_identifier_str}' not found for '{self.current_weight_info.name}'.")
            device_str = self.current_weight_info.device_str() if hasattr(self.current_weight_info,'device_str') and callable(self.current_weight_info.device_str) else 'cpu'
            dtype_val = self.current_weight_info.dtype if hasattr(self.current_weight_info,'dtype') else torch.float32
            return torch.empty(0, device=torch.device(device_str), dtype=dtype_val)

        target_device = tensors_arg[self.base_model_ref].device; target_dtype = tensors_arg[self.base_model_ref].dtype
        merged_tensor_accumulator = tensors_arg[self.base_model_ref].clone().to(device=target_device,dtype=target_dtype)
        ordered_model_refs_for_ca: List[ModelReference] = []
        model_ref_by_string_id: Dict[str, ModelReference] = {str(ref.model): ref for ref in tensors_arg.keys()}
        
        current_pruning_order_strings = self.global_pruning_order
        if current_pruning_order_strings:
            for id_str_in_order in current_pruning_order_strings:
                if id_str_in_order == base_model_identifier_str: continue
                if id_str_in_order in model_ref_by_string_id: ordered_model_refs_for_ca.append(model_ref_by_string_id[id_str_in_order])
                else: logging.warning(f"Model ID '{id_str_in_order}' from order not found for '{self.current_weight_info.name}'.")
        else:
            sorted_non_base_string_ids = sorted([str(ref.model) for ref in tensors_arg.keys() if str(ref.model) != base_model_identifier_str])
            for id_str in sorted_non_base_string_ids:
                 if id_str in model_ref_by_string_id: ordered_model_refs_for_ca.append(model_ref_by_string_id[id_str])
        
        if not ordered_model_refs_for_ca:
            logging.info(f"No non-base models for '{self.current_weight_info.name}'. Returning base."); return merged_tensor_accumulator
        cumulative_param_mask = torch.zeros_like(merged_tensor_accumulator,dtype=torch.bool,device=target_device)
        
        for model_ref_current in ordered_model_refs_for_ca:
            if model_ref_current not in tensors_arg: logging.warning(f"Tensor for '{str(model_ref_current.model)}' unavailable for '{self.current_weight_info.name}'."); continue
            fine_tuned_tensor_val = tensors_arg[model_ref_current].to(device=target_device,dtype=target_dtype)
            base_tensor_for_diff = tensors_arg[self.base_model_ref].to(device=target_device,dtype=target_dtype)
            scaling_coefficient = 1.0
            if model_ref_current in self.per_model_tensor_params:
                inner_params = self.per_model_tensor_params[model_ref_current]
                if "weight" in inner_params: 
                    val = inner_params["weight"]
                    if isinstance(val, (int,float)): scaling_coefficient = float(val)
                    else: logging.warning(f"Expected float for per-model weight, got {type(val)}")

            n_val_current,m_val_current = self._resolve_n_and_m_for_model(model_ref_current)
            task_vector_val = fine_tuned_tensor_val - base_tensor_for_diff
            available_params_mask = ~cumulative_param_mask
            candidate_task_vector = task_vector_val * available_params_mask.to(task_vector_val.dtype)
            pruned_task_vector,newly_retained_mask = prune_n_m_structural(candidate_task_vector,n_val_current,m_val_current)
            merged_tensor_accumulator += scaling_coefficient * pruned_task_vector.to(merged_tensor_accumulator.dtype)
            cumulative_param_mask = torch.logical_or(cumulative_param_mask,newly_retained_mask.to(device=cumulative_param_mask.device))
        return merged_tensor_accumulator

    @override
    def group_label(self) -> Optional[str]:
        return self.current_weight_info.name
