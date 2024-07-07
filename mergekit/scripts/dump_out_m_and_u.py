import os
import pdb
import sys
from collections import defaultdict
from typing import List, Optional

import click
import numpy as np
import safetensors.torch
import scipy
import torch

from mergekit.architecture import _template_substitution, get_architecture_info
from mergekit.common import ModelReference


def calc_correlation_matrix(feats):
    feats = feats.view(-1, feats.shape[-1])

    return torch.corrcoef(feats.T)


def match_tensors_permute(
    r=0.5,
    no_absval=True,
    correlation_matrix=None,
):
    """
    This function is adapted from ZipIt! (https://github.com/gstoica27/ZipIt)

    Matches arbitrary models by permuting all to the spaces of the first in your graph list.
    Mimics Rebasin methods.
    """

    O = correlation_matrix.shape[0]
    N = int(1 / (1 - r) + 0.5)
    Om = O // N
    device = correlation_matrix.device

    mats = [torch.eye(Om, device=device)]
    cost = 0
    for i in range(1, N):
        try:
            corr_submatrix = (
                correlation_matrix[:Om, Om * i : Om * (i + 1)].cpu().numpy()
            )
            if no_absval == False:
                corr_submatrix = np.absolute(corr_submatrix)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                corr_submatrix, maximize=True
            )
            cost = corr_submatrix[row_ind, col_ind].sum()
            # correlation subset is is [0:4096, 4096:8192]
        except Exception as e:
            print(e)
            pdb.set_trace()

        new_mat = torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)]
        mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)

    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge, None, cost / merge.shape[0]


def match_tensors_permute_MHA(
    n_heads=32,
    r=0.5,
    no_absval=True,
    correlation_matrix=None,
    number_of_repeats=8,
):
    """
    Handles different head permutations in attention
    """
    correlation = correlation_matrix

    O = correlation.shape[0]

    N = int(1 / (1 - r) + 0.5)  # num models
    Om = O // N  # matrix dimension
    device = correlation.device
    query_size = Om // n_heads

    mats = [torch.eye(Om, device=device)]
    head_perms = []

    costs = np.ones((n_heads, n_heads)) * -sys.maxsize

    cost = 0
    col_inds_storage = defaultdict(lambda: defaultdict(int))

    for i in range(1, N):  # just once if 2 models
        for j in range(n_heads):  # outer loop through all heads
            for k in range(n_heads):  # inner loop through heads >= current head j
                head1_idx = [query_size * j, query_size * (j + 1)]
                head2_idx = [query_size * k, query_size * (k + 1)]

                # take abs value of submatrix of correlations
                corr_submatrix = (
                    correlation[
                        head1_idx[0] : head1_idx[1],
                        (Om + head2_idx[0]) : (Om + head2_idx[1]),
                    ]
                    .cpu()
                    .numpy()
                )
                if no_absval == False:
                    corr_submatrix = np.absolute(corr_submatrix)

                # compute perm for head j & head k
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    corr_submatrix, maximize=True
                )

                # store cost (cost is maximized here)
                costs[j, k] = corr_submatrix[row_ind, col_ind].sum()

                # store perm so we don't have to recompute it later
                col_inds_storage[j][k] = col_ind

    outer_row_ind, outer_col_ind = scipy.optimize.linear_sum_assignment(
        costs, maximize=True
    )  # get assignment with lowest cost
    cost += costs[outer_row_ind, outer_col_ind].sum()

    for j in range(n_heads):
        head_1 = outer_row_ind[j]  # these are in order, outer_row_ind[j] = j
        head_2 = outer_col_ind[j]

        head_perm = col_inds_storage[head_1][head_2]
        head_perms.append(torch.tensor(head_perm + query_size * head_2))

    new_mat = torch.eye(Om, device=device)[
        torch.cat(head_perms).clone().detach().long().to(device)
    ]
    mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    return merge.T, unmerge, None, None


def match_tensors_permute_MHA_GQA(
    n_heads=32,
    r=0.5,
    no_absval=False,
    correlation_matrix=None,
    number_of_repeats=8,
):
    """
    Handles different head permutations in attention
    """
    correlation = correlation_matrix

    O = correlation.shape[0]

    N = int(1 / (1 - r) + 0.5)  # num models
    Om = O // N  # matrix dimension
    device = correlation.device
    query_size = Om // n_heads

    mats = [torch.eye(Om, device=device)]
    mats_2 = [torch.eye(Om // number_of_repeats, device=device)]
    head_perms = []
    head_perms_2 = []

    # in this case it is 256 x 256 i.e 2048 / 8 x 2048 / 8
    _compressed_correlation = torch.zeros(
        Om // number_of_repeats, Om // number_of_repeats, device=device
    )

    # compress matrix down
    for i in range(n_heads // number_of_repeats):
        for j in range(n_heads // number_of_repeats):
            _c = torch.zeros(query_size, query_size, device=device)
            for _i in range(number_of_repeats):
                for _j in range(number_of_repeats):
                    _c += correlation[
                        query_size * number_of_repeats * i
                        + _i * query_size : query_size * number_of_repeats * i
                        + (_i + 1) * query_size,
                        Om
                        + query_size * number_of_repeats * j
                        + _j * query_size : Om
                        + query_size * number_of_repeats * j
                        + (_j + 1) * query_size,
                    ]
            _compressed_correlation[
                i * query_size : (i + 1) * query_size,
                j * query_size : (j + 1) * query_size,
            ] = _c

    costs = (
        np.ones((n_heads // number_of_repeats, n_heads // number_of_repeats))
        * -sys.maxsize
    )

    cost = 0
    col_inds_storage = defaultdict(lambda: defaultdict(int))

    for i in range(1, N):  # just once if 2 models
        for j in range(n_heads // number_of_repeats):  # outer loop through all heads
            for k in range(
                n_heads // number_of_repeats
            ):  # inner loop through heads >= current head j
                head1_idx = [query_size * j, query_size * (j + 1)]
                head2_idx = [query_size * k, query_size * (k + 1)]

                # take abs value of submatrix of correlations
                corr_submatrix = (
                    _compressed_correlation[
                        head1_idx[0] : head1_idx[1], head2_idx[0] : head2_idx[1]
                    ]
                    .cpu()
                    .numpy()
                )
                if no_absval == False:
                    corr_submatrix = np.absolute(corr_submatrix)

                # compute perm for head j & head k
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    corr_submatrix, maximize=True
                )

                # store cost (cost is maximized here)
                costs[j, k] = corr_submatrix[row_ind, col_ind].sum()

                # store perm so we don't have to recompute it later
                col_inds_storage[j][k] = col_ind

    outer_row_ind, outer_col_ind = scipy.optimize.linear_sum_assignment(
        costs, maximize=True
    )  # get assignment with lowest cost
    cost += costs[outer_row_ind, outer_col_ind].sum()

    for j in range(n_heads // number_of_repeats):
        head_1 = outer_row_ind[j]  # these are in order, outer_row_ind[j] = j
        head_2 = outer_col_ind[j]

        head_perm = col_inds_storage[head_1][head_2]
        head_perms_2.append(torch.tensor(head_perm + query_size * head_2))

        for k in range(number_of_repeats):
            head_perms.append(
                torch.tensor(
                    head_perm + query_size * head_2 * number_of_repeats + k * query_size
                )
            )

    new_mat = torch.eye(Om, device=device)[
        torch.cat(head_perms).clone().detach().long().to(device)
    ]
    mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    # -------- smaller version ----------------------
    new_mat_2 = torch.eye(Om // number_of_repeats, device=device)[
        torch.tensor(torch.cat(head_perms_2)).long().to(device)
    ]
    mats_2.append(new_mat_2.T)

    unmerge_mats_2 = mats_2

    unmerge_2 = torch.cat(unmerge_mats_2, dim=0)
    merge_2 = torch.cat(mats_2, dim=0)
    merge_2 = merge_2 / (merge_2.sum(dim=0, keepdim=True) + 1e-5)
    return merge.T, unmerge, merge_2.T, unmerge_2


def match_tensors_permute_MHA_GQA_rev(
    n_heads=32,
    r=0.5,
    no_absval=False,
    correlation_matrix=None,
    number_of_repeats=8,
):
    """
    Handles different head permutations in attention
    """
    correlation = correlation_matrix

    O = correlation.shape[0]

    N = int(1 / (1 - r) + 0.5)  # num models
    Om = O // N  # matrix dimension
    device = correlation.device
    query_size = 64  # Om // number_of_repeats

    mats = [torch.eye(Om * number_of_repeats, device=device)]
    mats_2 = [torch.eye(Om, device=device)]
    head_perms = []
    head_perms_2 = []

    costs = (
        np.ones((n_heads // number_of_repeats, n_heads // number_of_repeats))
        * -sys.maxsize
    )

    cost = 0
    col_inds_storage = defaultdict(lambda: defaultdict(int))

    for i in range(1, N):  # just once if 2 models
        for j in range(n_heads // number_of_repeats):  # outer loop through all heads
            for k in range(
                n_heads // number_of_repeats
            ):  # inner loop through heads >= current head j
                head1_idx = [query_size * j, query_size * (j + 1)]
                head2_idx = [Om + query_size * k, Om + query_size * (k + 1)]

                # take abs value of submatrix of correlations
                corr_submatrix = (
                    correlation[
                        head1_idx[0] : head1_idx[1], head2_idx[0] : head2_idx[1]
                    ]
                    .cpu()
                    .numpy()
                )

                if no_absval == False:
                    corr_submatrix = np.absolute(corr_submatrix)

                # compute perm for head j & head k
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                    corr_submatrix, maximize=True
                )

                # store cost (cost is maximized here)
                costs[j, k] = corr_submatrix[row_ind, col_ind].sum()

                # store perm so we don't have to recompute it later
                col_inds_storage[j][k] = col_ind

    outer_row_ind, outer_col_ind = scipy.optimize.linear_sum_assignment(
        costs, maximize=True
    )  # get assignment with lowest cost
    cost += costs[outer_row_ind, outer_col_ind].sum()

    for j in range(n_heads // number_of_repeats):
        head_1 = outer_row_ind[j]  # these are in order, outer_row_ind[j] = j
        head_2 = outer_col_ind[j]

        head_perm = col_inds_storage[head_1][head_2]
        head_perms_2.append(torch.tensor(head_perm + query_size * head_2))

        for k in range(number_of_repeats):
            head_perms.append(
                torch.tensor(
                    head_perm + number_of_repeats * query_size * head_2 + k * query_size
                )
            )

    new_mat = torch.eye(Om * number_of_repeats, device=device)[
        torch.tensor(torch.cat(head_perms)).long().to(device)
    ]
    mats.append(new_mat.T)

    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)

    # -------- smaller version ----------------------
    new_mat_2 = torch.eye(Om, device=device)[
        torch.tensor(torch.cat(head_perms_2)).long().to(device)
    ]
    mats_2.append(new_mat_2.T)

    unmerge_mats_2 = mats_2

    unmerge_2 = torch.cat(unmerge_mats_2, dim=0)
    merge_2 = torch.cat(mats_2, dim=0)
    merge_2 = merge_2 / (merge_2.sum(dim=0, keepdim=True) + 1e-5)
    return merge.T, unmerge, merge_2.T, unmerge_2


@click.command()
@click.argument("model1-ft", type=str, required=True)
@click.argument("model2-ft", type=str, required=True)
@click.option("--model_path", type=str, required=True, help="Model information")
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
@click.option(
    "--key-shrink",
    type=bool,
    default=False,
    help="Shrink the GQA matrix to obtain key space permutation. Default is False, the opposite is to grow query related permutation matrix",
)
def main(model1_ft, model2_ft, model_path, out_path, metric, device, key_shrink):
    model = ModelReference.model_validate(model_path)

    model_config = model.config()

    has_gqa = hasattr(model_config, "num_key_value_heads") and getattr(
        model_config, "num_key_value_heads", 0
    ) < getattr(model_config, "num_attention_heads", 0)

    model_arch_info = get_architecture_info(model_config)

    _json = model_arch_info.definition

    residual_space = None  # probably don't need this
    kq_space = None  # We do need this
    v_space = None

    # extract the residual, attention related spaces
    for weight in _json.layer_templates.weights:
        if weight.is_kq:
            kq_space = weight.output_space
            residual_space = weight.input_space
            continue

        # assuming order is observed
        if (
            not weight.is_kq
            and weight.head_split
            and (weight.input_space == residual_space)
        ):
            v_space = weight.output_space
            continue

    num_layers = model_arch_info.num_layers(model_config)

    kq_spaces = []
    v_spaces = []
    for j in range(num_layers):
        kq_spaces.append(
            _template_substitution(kq_space, num_layers=num_layers, layer_idx=j)
        )
        v_spaces.append(
            _template_substitution(v_space, num_layers=num_layers, layer_idx=j)
        )

    model1_features = safetensors.torch.load_file(model1_ft)
    model2_features = safetensors.torch.load_file(model2_ft)

    model_1_attention_mask = model1_features.pop("attention_mask")
    model_2_attention_mask = model2_features.pop("attention_mask")

    merges = {}
    unmerges = {}

    for feature_space in model1_features.keys():
        # model1_feature = model1_features[layer_name].float().to(device)
        # model2_feature = model2_features[layer_name].float().to(device)

        # model1_filtered_feature = remove_pads(model_1_attention_mask, model1_feature)
        # model2_filtered_feature = remove_pads(model_2_attention_mask, model1_feature)

        concatenated_feature = torch.cat(
            (model1_features[feature_space], model2_features[feature_space]), dim=-1
        )

        correlation_matrix = calc_correlation_matrix(concatenated_feature)

        if feature_space in (kq_spaces + v_spaces):
            if not has_gqa:
                f = match_tensors_permute_MHA
            elif not key_shrink:
                f = match_tensors_permute_MHA_GQA
            else:
                f = match_tensors_permute_MHA_GQA_rev
            merge, unmerge, a_merge, a_unmerge = f(
                correlation_matrix=correlation_matrix,
                n_heads=model_config.num_attention_heads,
                number_of_repeats=8,
            )

            # print merge, unmerge shape
            merges[feature_space] = merge
            unmerges[feature_space] = unmerge

            if has_gqa:
                merges[feature_space + "_alternate"] = a_merge
                unmerges[feature_space + "_alternate"] = a_unmerge

        else:
            merge, unmerge, _, _ = match_tensors_permute(
                correlation_matrix=correlation_matrix
            )
            merges[feature_space] = merge
            unmerges[feature_space] = unmerge

    os.makedirs(out_path, exist_ok=True)

    qkv_spaces = []
    # TODO: figure out a better way to do this
    qkv_space = "attn_qkv_${layer_index}"
    for j in range(num_layers):
        qkv_spaces.append(
            _template_substitution(qkv_space, num_layers=num_layers, layer_idx=j)
        )

    # NOTE: making sure the attention space merge/unmerge is shared
    for v_space, kq_space, qkv_space in zip(v_spaces, kq_spaces, qkv_spaces):
        # merges[v_space], unmerges[v_space] = merges[kq_space], unmerges[kq_space]

        merges[qkv_space], unmerges[qkv_space] = merges[v_space], unmerges[v_space]

        if has_gqa:
            # store the alternative
            merges[v_space + "_alternate"], unmerges[v_space + "_alternate"] = (
                merges[kq_space + "_alternate"],
                unmerges[kq_space + "_alternate"],
            )
            merges[qkv_space + "_alternate"], unmerges[qkv_space + "_alternate"] = (
                merges[kq_space + "_alternate"],
                unmerges[kq_space + "_alternate"],
            )

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

# python scripts/dump_out_m_and_u.py ./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v0.6_features.bin ./dump_output/TinyLlama_TinyLlama-1.1B-Chat-v1.0_features.bin --model_path TinyLlama/TinyLlama-1.1B-Chat-v0.6 --out_path ./m_v_out
