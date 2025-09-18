import torch
from typing import List, Dict, Any

def iso_c(task_vectors: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0]:
            tvs = [task_vector[key] for task_vector in task_vectors]
            new_vector[key] = sum(tvs) / len(tvs)

            if len(task_vectors[0][key].shape) == 2 and "text_projection" not in key:
                new_vector[key] *= len(tvs)
                U, S, V = torch.linalg.svd(new_vector[key], full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_mean),
                        V,
                    )
                )

    return new_vector

###############
#### TSV Merge Orthogonalization
def compute_and_sum_svd_mem_reduction(task_vectors: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
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
    print("Computing SVD...")
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0]:
            new_vector[key] = {}
            for i, task_vector in enumerate(task_vectors):
                vec = task_vector[key]

                if (
                    len(task_vector[key].shape) == 2
                    and "text_projection" not in key
                ):
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)

                    if i == 0:
                        print(f"Computed SVD for {key}...")
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
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
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (i + 1)

            if len(task_vector[key].shape) == 2 and "text_projection" not in key:
                u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

                new_vector[key] = torch.linalg.multi_dot(
                    (
                        u_u,
                        v_u,
                        torch.diag(sum_s),
                        u_v,
                        v_v,
                    )
                )
                '''

                if config.method.apply_subspace_boosting:
                    print("Applying subspace boosting...")
                    U, S, Vh = torch.linalg.svd(new_vector[key], full_matrices=False)

                    total_sum = S.sum()
                    cumulative = torch.cumsum(S, dim=0)
                    thresh = config.method.svd_thresh # svd threshold for the boosting

                    # only apply boosting to non-empty matrices
                    if total_sum.item() != 0:
                        k = (cumulative / total_sum >= thresh).nonzero(as_tuple=False)
                        cutoff_idx = k[0].item()

                        S_damped = torch.clamp(S, min=S[cutoff_idx])

                        new_vector[key] = (U * S_damped.unsqueeze(0)) @ Vh
                '''

    return new_vector

def subspace_boosting(
        merged_tv_state_dict: Dict[str, Any], 
        reset_thresh=20, # TODO: refactor the parameter list and just use the config
        svd_thresh=0.01,
        attn_svd_thresh=0.10,
        cumsum=True, 
        remove_keys=[]
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

    start_time = time.time_ns()

    for key, param in merged_tv_state_dict.items():
        if any(i in key for i in keys_to_eval) and isinstance(param, torch.Tensor):
            logging.info(f"Applying subspace boosting to {key} with shape {param.shape}")
            '''
            # Process attention weights per head or qkv
            if keys_to_eval[0] in key:
                if apply_to_attn == "per_head":
                    merged_tv_state_dict[key] = _per_head_subspace_boosting(param, config, config.method.attn_svd_thresh, cumsum)
                elif apply_to_attn == "per_qkv":
                    merged_tv_state_dict[key] = _per_qkv_subspace_boosting(param, config, config.method.attn_svd_thresh, cumsum)
            
            # Process full attention weights and MLP weights
            if apply_to_attn == "full_attn" or (keys_to_eval[0] not in key):
            '''
            U, S, Vh = torch.linalg.svd(param, full_matrices=False)

            # Clamping approach using the cumulative sum of singular values as the threshold
            if cumsum:
                total_sum = S.sum()
                cumulative = torch.cumsum(S, dim=0)
                
                # thresh = config.method.attn_svd_thresh if (keys_to_eval[0] in key) else svd_thresh
                thresh = svd_thresh
                
                k = (cumulative / total_sum >= thresh).nonzero(as_tuple=False)
                cutoff_idx = k[0].item()

                S_damped = torch.clamp(S, min=S[cutoff_idx])
            else: # Clamping approach using the threshold as an index
                cutoff_idx = int(thresh * S.numel())
                S_damped = torch.clamp(S, min=S[cutoff_idx])

            merged_tv_state_dict[key] = (U * S_damped.unsqueeze(0)) @ Vh

    end_time = time.time_ns()

    logging.info(f"Subspace Boosting took {(end_time - start_time) / 1_000_000} ms.")
    
    return merged_tv_state_dict