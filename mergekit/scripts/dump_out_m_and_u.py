import os
import pdb
from typing import List, Optional

import click
import numpy as np
import safetensors.torch
import scipy
import torch

from mergekit.scripts.zipit_utils import CovarianceMetric


def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def match_tensors_permute(
    r=0.5,
    get_merge_value=False,
    print_costs=False,
    no_absval=False,
    correlation_matrix=None,
):
    """
    This function is adapted from ZipIt! (https://github.com/gstoica27/ZipIt)

    Matches arbitrary models by permuting all to the spaces of the first in your graph list.
    Mimics Rebasin methods.
    """

    correlation = correlation_matrix

    O = correlation.shape[0]
    N = int(1 / (1 - r) + 0.5)
    Om = O // N
    device = correlation.device

    mats = [torch.eye(Om, device=device)]
    cost = 0
    for i in range(1, N):
        try:
            corr_matrix = correlation[:Om, Om * i : Om * (i + 1)].cpu().numpy()
            if no_absval == False:
                corr_matrix = np.absolute(corr_matrix)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                corr_matrix, maximize=True
            )
            cost = corr_matrix[row_ind, col_ind].sum()
            # correlation subset is is [0:4096, 4096:8192]
            # correlation between the first graph's and second graph's features
        except Exception as e:
            print(e)
            pdb.set_trace()

        new_mat = torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)]
        mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    if get_merge_value:
        merge_value = (
            correlation[:Om, Om * i : Om * (i + 1)]
            .cpu()
            .numpy()[row_ind, col_ind]
            .mean()
        )
        return merge.T, unmerge, merge_value
    if print_costs:
        cost = cost / merge.shape[0]
        print(f"cost: {cost}")

    return merge.T, unmerge, None, cost / merge.shape[0]

def match_tensors_zipit(
    metric, r=.5, a=0.3, b=.125, 
    correlation_matrix=None,
    **kwargs
):
    """
    ZipIt! matching algorithm. Given metric dict, computes matching as defined in paper. 
    Args:
    - metric: dictionary containing metrics. This must contain either a covariance or cossim matrix, and 
        must be [(num_models x model_feature_dim), (num_models x model_feature_dim)]. 
    - r: Amount to reduce total input feature dimension - this is num_models x model_feature_dim. This function will
        compute (un)merge matrix that goes from 
        (num_models x model_feature_dim) -> (1-r)*(num_models x model_feature_dim) = merged_feature_dim.
        E.g. if num_models=2, model_feature_dim=10 and r=.5, the matrix will map from 2x10=20 -> (1-.5)x2x10=10, or halve the 
        collective feature space of the models.
    - a: alpha hyperparameter as defined in Section 4.3 of our paper. 
    - b: beta hyperparameter as defined in Section 4.3 of our paper.
    - print_merges: whether to print computed (un)merge matrices.
    - get_merge_value default False, returns the sum of correlations over all the merges which the algorithm made. 
    - add_bias: whether to add a bias to the input. This should only be used if your module expects the input with bias offset.
    returns:
    - (un)merge matrices
    """
    if "covariance" in metric:
        sims = correlation_matrix
    elif "cossim" in metric:
        sims = metric["cossim"]
    O = sims.shape[0]
    remainder = int(O * (1-r) + 1e-4)
    permutation_matrix = torch.eye(O, O)#, device=sims.device)

    torch.diagonal(sims)[:] = -torch.inf

    num_models = int(1/(1 - r) + 0.5)
    Om = O // num_models

    original_model = torch.zeros(O, device=sims.device).long()
    for i in range(num_models):
        original_model[i*Om:(i+1)*Om] = i

    to_remove = permutation_matrix.shape[1] - remainder
    budget = torch.zeros(num_models, device=sims.device).long() + int((to_remove // num_models) * b + 1e-4)

    merge_value = []

    while permutation_matrix.shape[1] > remainder:
        best_idx = sims.reshape(-1).argmax()
        row_idx = best_idx % sims.shape[1]
        col_idx = best_idx // sims.shape[1]
        
        merge_value.append(permutation_matrix[row_idx, col_idx])

        if col_idx < row_idx:
            row_idx, col_idx = col_idx, row_idx
        
        row_origin = original_model[row_idx]
        col_origin = original_model[col_idx]
        
        permutation_matrix[:, row_idx] += permutation_matrix[:, col_idx]
        permutation_matrix = remove_col(permutation_matrix, col_idx)
        
        sims[:, row_idx] = torch.minimum(sims[:, row_idx], sims[:, col_idx])
        
        if 'magnitudes' in metric:
            metric['magnitudes'][row_idx] = torch.minimum(metric['magnitudes'][row_idx], metric['magnitudes'][col_idx])
            metric['magnitudes'] = remove_col(metric['magnitudes'][None], col_idx)[0]
        
        if a <= 0:
            sims[row_origin*Om:(row_origin+1)*Om, row_idx] = -torch.inf
            sims[col_origin*Om:(col_origin+1)*Om, row_idx] = -torch.inf
        else: sims[:, row_idx] *= a
        sims = remove_col(sims, col_idx)
        
        sims[row_idx, :] = torch.minimum(sims[row_idx, :], sims[col_idx, :])
        if a <= 0:
            sims[row_idx, row_origin*Om:(row_origin+1)*Om] = -torch.inf
            sims[row_idx, col_origin*Om:(col_origin+1)*Om] = -torch.inf
        else: sims[row_idx, :] *= a
        sims = remove_col(sims.T, col_idx).T

        row_origin, col_origin = original_model[row_idx], original_model[col_idx]
        original_model = remove_col(original_model[None, :], col_idx)[0]
        
        if row_origin == col_origin:
            origin = original_model[row_idx].item()
            budget[origin] -= 1

            if budget[origin] <= 0:
                # kill origin
                selector = original_model == origin
                sims[selector[:, None] & selector[None, :]] = -torch.inf
    

    unmerge = permutation_matrix

    merge = permutation_matrix / (permutation_matrix.sum(dim=0, keepdim=True) + 1e-5)

    merge = merge.to(sims.device)
    unmerge = unmerge.to(sims.device)
    
    return merge.T, unmerge


@click.command()
@click.argument("model1-ft", type=str, required=True)
@click.argument("model2-ft", type=str, required=True)
@click.option(
    "--out_path", required=True, type=str, help="Output path for metric tensors"
)
@click.option(
    "--metric",
    "-m",
    type=str,
    default="covariance",
    help="Metric to calculate (default: covariance)",
)
@click.option(
    "--device",
    "-d",
    type=str,
    default="cpu",
    help="Device to compute on (default: cpu)",
)
def main(model1_ft, model2_ft, out_path, metric, device):
    metric_classes = {"covariance": CovarianceMetric()}
    if metric not in metric_classes:
        raise ValueError(f"Unsupported metric: {metric}")

    model1_features = safetensors.torch.load_file(model1_ft)
    model2_features = safetensors.torch.load_file(model2_ft)

    model_1_attention_mask = model1_features.pop("attention_mask")
    model_2_attention_mask = model2_features.pop("attention_mask")

    merges = {}
    unmerges = {}

    for layer_name in model1_features.keys():

        print(f"Layer: {layer_name}")

        concatenated_feature = torch.cat(
            (model1_features[layer_name], model2_features[layer_name]), dim=-1
        )

        # Luke why do we have a four dimentional tensor ? 

        print(f" -> {concatenated_feature.shape}")

        concatenated_layer = concatenated_feature
        # TODO: one wonder about flattening the first two dims :thinking-face:

        print(f" -> {concatenated_layer.shape}")

        metric_instance = metric_classes[metric]

        final_metric = metric_instance.calculate(concatenated_layer)
        print(f" final_metric -> {final_metric.shape}")
        merge, unmerge, _, _ = match_tensors_permute(correlation_matrix=final_metric)
        # print merge, unmerge shape
        print(f" merge -> {merge.shape}")
        print(f" unmerge -> {unmerge.shape}")
        merges[layer_name] = merge
        unmerges[layer_name] = unmerge

    os.makedirs(out_path, exist_ok=True)

    # Saving the metrics results as SafeTensors
    for identifier, tensor in merges.items():
        safetensors.torch.save_file(
            {identifier: tensor.contiguous()},
            f"{out_path}/{identifier}_merge.safetensor",
        )
    for identifier, tensor in unmerges.items():
        safetensors.torch.save_file(
            {identifier: tensor.contiguous()},
            f"{out_path}/{identifier}_unmerge.safetensor",
        )


if __name__ == "__main__":
    main()

# python scripts/dump_m_and_u.py ./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v0.6_features.bin ./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v1.0_features.bin --out_path ./m_v_out
