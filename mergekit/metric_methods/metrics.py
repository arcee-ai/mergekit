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

from mergekit.common import ModelReference
from mergekit.metric_methods.base import MeanStd, Heatmap, Histogram, Metric
import torch
import torch.nn.functional as F
import numpy as np
from typing import List

# Helper functions

def compute_histogram(tensor: torch.Tensor, n_bins: int) -> List[np.ndarray]:
    bin_counts, bin_edges = np.histogram(tensor.cpu().numpy(), bins=n_bins)
    bin_widths = np.diff(bin_edges)
    return bin_counts, bin_edges, bin_widths

def cosine_similarity_heatmap(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Normalize the rows of both matrices
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    
    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(A_norm, B_norm.t())
    
    return similarity_matrix


# Tensor Comparisons (Require exactly 2 tensors)

def smape(
    tensors: List[torch.Tensor], **_kwargs
) -> Metric:
    """Symmetric Mean Absolute Percentage Error (smape)."""

    numerator = torch.abs(tensors[0] - tensors[1])
    denominator = (torch.abs(tensors[0]) + torch.abs(tensors[1]))
    smape = torch.mean(torch.div(numerator, denominator), dim=1) 
    
    hist_info = compute_histogram(smape, 100)

    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=smape.mean().item(), std=smape.std().item())
    )

def cosine_similarity(
    tensors: List[torch.Tensor], return_heatmap=False, **_kwargs
) -> Metric:
    """Cosine similarity"""
    cosine_similarity = F.cosine_similarity(tensors[0], tensors[1], dim=1)

    if return_heatmap:
        heatmap = cosine_similarity_heatmap(tensors[0], tensors[1])

    assert torch.isclose(cosine_similarity, cosine_similarity, atol=1e-6).all(), "NaNs in cosine similarity"
    assert torch.isclose(cosine_similarity, cosine_similarity_heatmap(tensors[0], tensors[1]).diagonal(), atol=1e-2).all(), "Diagonal elements of cosine similarity matrix do not match"

    hist_info = compute_histogram(cosine_similarity, 100)
    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=cosine_similarity.mean().item(), std=cosine_similarity.std().item()),
        heatmap=Heatmap(data=heatmap) if return_heatmap else None
    )

def scale(
    tensors: List[torch.Tensor], return_heatmap=False, **_kwargs
) -> Metric:
    """
    Scale difference: ratio of absolute difference to average scale.
    Complementary to cosine similarity, which measures the angle between two vectors and is invariant to scale.

    values close to 0 indicate that the scales of the two vectors are similar
    """

    norm_0 = torch.norm(tensors[0], dim=1)
    norm_1 = torch.norm(tensors[1], dim=1)

    scale_diff = torch.abs(norm_0 - norm_1) / ((norm_0 + norm_1) / 2)

    if return_heatmap:
        norm_0 = norm_0.unsqueeze(1)  # shape becomes [num_heads, 1]
        norm_1 = norm_1.unsqueeze(0)  # shape becomes [1, num_heads]

        # Compute the scale difference between each pair of heads by broadcasting
        heatmap = torch.abs(norm_0 - norm_1) / ((norm_0 + norm_1 + 1e-10) / 2)

        assert torch.isclose(scale_diff, heatmap.diagonal(), atol=1e-4).all(), "Diagonal elements of scale difference matrix do not match"
        
    hist_info = compute_histogram(scale_diff, 100)

    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=scale_diff.mean().item(), std=scale_diff.std().item()),
        heatmap=Heatmap(data=heatmap) if return_heatmap else None
    )

def mse(
    tensors: List[torch.Tensor], return_heatmap: bool =False, **_kwargs
) -> Metric:
    """Mean squared error (MSE)."""
    if return_heatmap:
        # Expand dimensions for broadcasting
        tensors_0_exp = tensors[0].unsqueeze(1)  # shape becomes [num_heads, 1, ...]
        tensors_1_exp = tensors[1].unsqueeze(0)  # shape becomes [1, num_heads, ...]

        # Compute squared differences
        diffs = (tensors_0_exp - tensors_1_exp) ** 2

        # Compute mean over all dimensions except the first two
        heatmap = diffs.mean(dim=tuple(range(2, diffs.dim()))).cpu().numpy()

    squared_diff = (tensors[0] - tensors[1]) ** 2
    mse_per_neuron = torch.mean(squared_diff, dim=1)

    hist_info = compute_histogram(mse_per_neuron, 100)

    return Metric(
        histogram=Histogram(count=hist_info[0], edges=hist_info[1], widths=hist_info[2]),
        mean_std=MeanStd(mean=mse_per_neuron.mean().item(), std=mse_per_neuron.std().item()),
        heatmap=Heatmap(data=heatmap) if return_heatmap else None
    )

# Tensor Analysis (number of tensors can vary)

def weight_magnitude(tensor: torch.Tensor) -> Metric:
    weight_magnitudes = torch.abs(tensor.flatten())
    hist_info = compute_histogram(weight_magnitudes, 100)
    return Metric(
        histogram=Histogram(count=hist_info[0], 
                            edges=hist_info[1], 
                            widths=hist_info[2]
                            ),
        mean_std=MeanStd(mean=weight_magnitudes.mean().item(),
                            std=weight_magnitudes.std().item()),
        )

def numerical_rank(tensor: torch.Tensor, epsilon: float = 1e-5) -> Metric:
    """
    Computes the numerical rank of the representations matrix X based on the singular values
    of its sample covariance matrix. The rank is determined as the number of singular values
    above a threshold. The threshold is defined as the highest singular value times a given epsilon.

    Parameters:
    - X : torch.Tensor
        The representations matrix from which the sample covariance matrix will be computed.
    - epsilon : float, optional
        The factor to multiply with the highest singular value to set the threshold (default is 1e-3).
    - flip : bool, optional - allows transpose for efficient computation. False only used in testing
    Returns:
    - int
        The numerical rank of the matrix.

    Implemented according to description in the paper:
        The Tunnel Effect: Building Data Representations in Deep Neural Networks
        https://arxiv.org/pdf/2305.19753.pdf

    """
        
    # Center the data by subtracting the mean
    X_centered = tensor - torch.mean(tensor, dim=0)
    X_std = torch.std(X_centered, dim=0, unbiased=False)
    X_centered /= X_std

    # Compute the sample covariance matrix
    covariance_matrix = X_centered.t() @ X_centered / (tensor.shape[0] - 1)
    # Compute singular values using SVD on the covariance matrix
    U, singular_values, V = torch.svd(covariance_matrix.cpu())
    # Determine the threshold
    threshold = singular_values[0] * epsilon
    # Count singular values greater than the threshold
    num_rank = torch.sum(singular_values > threshold).item()

    value = int(num_rank)

    return Metric(
            mean_std=MeanStd(
                mean=value, 
                std=None), 
            )

