# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from typing import Any, Dict, List, Optional

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.io.tasks import GatherTensors
from mergekit.metric_methods.base import MetricMethod
from mergekit.merge_methods.base import ConfigParameterDef
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

import torch.nn.functional as F

def pca_components_for_variance(X, variance_threshold=0.99, rescale=True):
    """
    Compute the number of principal components required to explain
    at least `variance_threshold` of the total variance in the dataset X using PyTorch.

    Args:
        X (torch.Tensor): The data matrix. Rows are samples and columns are features.
        variance_threshold (float): The fraction of total variance that we want to capture.

    Returns:
        int: The number of principal components required to capture the specified variance threshold.
    """
    # Standardize the data (mean 0 and variance 1)
    X_mean = torch.mean(X, dim=0)
    X_std = torch.std(X, dim=0, unbiased=False)
    X = X - X_mean

    if rescale:
        X = X / X_std

    # Compute the covariance matrix
    covariance_matrix = torch.mm(X.T, X) / (X.shape[0] - 1)

    # Perform SVD on the covariance matrix
    U, S, V = torch.svd(covariance_matrix)

    # Calculate explained variance ratios
    explained_variance_ratio = S / torch.sum(S)
    cumsum_variance = torch.cumsum(explained_variance_ratio, dim=0)

    # Determine the number of components needed to surpass the variance threshold
    num_components = torch.where(cumsum_variance >= variance_threshold)[0][0] + 1

    return num_components.item()


class PCA_RankTask(Task[torch.Tensor]):
    gather_tensors: GatherTensors
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    normalize: bool
    weight_info: WeightInfo

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs
    ) -> torch.Tensor:
        keys = list(tensors.keys())

        tensors = [tensors[key] for key in keys]


        unique_shapes = set(t.shape for t in tensors)
        if len(unique_shapes) != 1:
            raise RuntimeError(
                f"Tensor size mismatch for {self.weight_info.name}, sizes: {list(unique_shapes)}"
            )
        if len(tensors) != 1:
            raise RuntimeError(f"Expected 1 tensors, got {len(tensors)}")

        if 'mlp' not in self.weight_info.name:
            return

        res = {}
        X = tensors[0]

        res['num_components_99'] = pca_components_for_variance(X, variance_threshold=0.99, rescale=True)
        res['num_components_95'] = pca_components_for_variance(X, variance_threshold=0.95, rescale=True)
        return res


    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class PCA_RankMetric(MetricMethod):

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: GatherTensors,
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return PCA_RankTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            weight_info=output_weight,
        )
