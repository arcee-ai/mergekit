import torch
from typing import List, Dict, Any, Optional
import time
import logging

def iso_c(task_vectors: List[torch.Tensor], tv_key: str, device: torch.device) -> Dict[str, Any]:
    with torch.no_grad():
        tvs = task_vectors
        new_vector = sum(tvs) / len(tvs)
        original_dtype = new_vector.dtype  # Store original dtype

        if (len(task_vectors[0].shape) == 2 and "embed_tokens" not in tv_key and "lm_head" not in tv_key):
            print(f"Computing SVD for {tv_key}... with shape {task_vectors[0].shape}")
            new_vector *= len(tvs)
            # Convert to float32 for SVD
            vec_fp32 = new_vector.to(torch.float32)
            U, S, V = torch.linalg.svd(vec_fp32, full_matrices=False)
            S_mean = torch.ones_like(S) * S.mean()

            # Perform matrix multiplication in float32 and convert back to original dtype
            new_vector = torch.linalg.multi_dot(
                (
                    U,
                    torch.diag(S_mean),
                    V,
                )
            ).to(original_dtype)  # Convert back to original dtype

    return new_vector

###############
#### TSV Merge Orthogonalization
def compute_and_sum_svd_mem_reduction(task_vectors: List[torch.Tensor], tv_key: str, device: torch.device) -> Dict[str, Any]:
    """
    Computes the Singular Value Decomposition (SVD) for each vector in the task_vectors,
    reduces the dimensionality of the vectors based on the sv_reduction factor, and concatenate
    the low-rank matrices. If the vector is not a 2D tensor or is "text_projection", it computes the mean of the vectors.
    Computation of the SVD is performed also for the second operation.

    Args:
        task_vectors (list): A list of task vector objects, where each object contains a
                             dictionary of vectors.
    Returns:
        dict: A dictionary containing the new vectors after SVD computation and merging.
    """
    sv_reduction = 1 / len(task_vectors)
    with torch.no_grad():
        new_vector = {}
        for i, task_vector in enumerate(task_vectors):
            vec = task_vector
            original_dtype = vec.dtype  # Store original dtype

            if (
                len(task_vector.shape) == 2
                and "embed_tokens" not in tv_key
                and "lm_head" not in tv_key
            ):
                print(f"Computing SVD for {tv_key}... with shape {task_vector.shape}")
                # Convert to float32 for SVD
                vec_fp32 = vec.to(torch.float32)
                u, s, v = torch.linalg.svd(vec_fp32, full_matrices=False)

                if i == 0:
                    sum_u = torch.zeros_like(u, device=device, dtype=torch.float32)
                    sum_s = torch.zeros_like(s, device=device, dtype=torch.float32)
                    sum_v = torch.zeros_like(v, device=device, dtype=torch.float32)
                reduced_index_s = int(s.shape[0] * sv_reduction)

                # select only the first reduced_index_s columns of u and place them
                sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                    :, :reduced_index_s
                ]
                sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                    :reduced_index_s
                ]
                # select only the first reduced_index_s rows of v and place them
                sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                    :reduced_index_s, :
                ]

            else:
                if i == 0:
                    new_vector = vec.clone()
                else:
                    new_vector += (vec - new_vector) / (i + 1)

        if (
            len(task_vector.shape) == 2
            and "embed_tokens" not in tv_key
            and "lm_head" not in tv_key
        ):
            # Perform final SVD operations in float32
            
            try:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            except torch._C._LinAlgError:
                print(f"[Retry with 'gesvd'] SVD failed for {tv_key}.")
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False, driver='gesvd')
            
            try:
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            except torch._C._LinAlgError:
                print(f"[Retry with 'gesvd'] SVD failed for {tv_key}.")
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False, driver='gesvd')

            # Perform matrix multiplication in float32
            new_vector = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag(sum_s),
                    u_v,
                    v_v,
                )
            ).to(original_dtype)  # Convert back to original dtype

    return new_vector

def subspace_boosting(
        merged_tv_key: str,
        merged_tv: torch.Tensor, 
        svd_thresh=0.01,
        cumsum=True, 
    ) -> Dict[str, Any]:
    """
    Subspace boosting for merging task vectors.

    Parameters:
        tv_flat_checks: Flattened task vectors.
        ptm_check: 
            Pretrained model.
        config: 
            Configuration object containing method parameters (e.g., config.method.k, config.method.use_ties).
        reset_thresh: default 20
            Threshold parameter used for ties merging. defaults to 20.
        svd_thresh: default 0.01
            Threshold for singular value boosting. If cumsum is True, used as a cumulative ratio threshold;
            otherwise used as a fraction of the total number of singular values. Defaults to 0.01.
        cumsum:
            Whether to use the cumulative sum approach for thresholding the singular values.
        remove_keys:
            Optional list of keys to remove from the state dict conversion.
    
    Returns:
        A merged flat vector representing the task vector after subspace boosting.

    Raises:
        ValueError: If the base_method is not one of the defined options.
    """
    
    # Merging approach for attention weight matrices
    #apply_to_attn = config.method.apply_to_attn
    # apply_to_attn=False: no subspace boosting for attention weights
    #if apply_to_attn not in [False, "full_attn", "per_qkv", "per_head"]:
    #    raise ValueError(f"Apply to attention method {apply_to_attn} not defined.")
    
    keys_to_eval = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ]
    
    if any(i in merged_tv_key for i in keys_to_eval) and isinstance(merged_tv, torch.Tensor):
        print(f"Applying subspace boosting to {merged_tv_key} with shape {merged_tv.shape}")
        
        # Store original dtype
        original_dtype = merged_tv.dtype
        
        # Convert to float32 for SVD
        merged_tv_fp32 = merged_tv.to(torch.float32)
        
        U, S, Vh = torch.linalg.svd(merged_tv_fp32, full_matrices=False)

        if cumsum:
            total_sum = S.sum()
            cumulative = torch.cumsum(S, dim=0)
            
            thresh = svd_thresh
            
            k = (cumulative / total_sum >= thresh).nonzero(as_tuple=False)
            
            if k.numel() == 0:
            # fallback: use smallest singular value
                cutoff_idx = -1
                print(f"[Warning] No valid SVD cutoff for {merged_tv_key}. Using full singular spectrum.")
            else:
                cutoff_idx = k[0].item()

            S_damped = torch.clamp(S, min=S[cutoff_idx])
        else:  # Clamping approach using the threshold as an index
            cutoff_idx = int(thresh * S.numel())
            S_damped = torch.clamp(S, min=S[cutoff_idx])

        # Perform matrix multiplication in FP32
        merged_tv = (U * S_damped.unsqueeze(0)) @ Vh
        
        # Convert back to original dtype
        merged_tv = merged_tv.to(original_dtype)

    return merged_tv