import os
import pdb
from typing import List, Optional

import click
import numpy as np
import safetensors.torch
import scipy
import torch

from mergekit.scripts.zipit_utils import CovarianceMetric, remove_pads


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
        # model1_feature = model1_features[layer_name].float().to(device)
        # model2_feature = model2_features[layer_name].float().to(device)

        # model1_filtered_feature = remove_pads(model_1_attention_mask, model1_feature)
        # model2_filtered_feature = remove_pads(model_2_attention_mask, model1_feature)

        # print(model2_filtered_feature.shape)
        # exit()

        print(f"Layer: {layer_name}")

        concatenated_feature = torch.cat(
            (model1_features[layer_name], model2_features[layer_name]), dim=-1
        )

        print(f" -> {concatenated_feature.shape}")

        # if "attention" in layer_name:
        #    # how to compute the metrics here?
        #    print(layer_name)
        # else:
        #    print(layer_name)

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
