# zip_it_metrics.py
from abc import ABC, abstractmethod

import pbd
import torch


class MetricCalculator(ABC):
    @abstractmethod
    def update(self, feats):
        pass

    @abstractmethod
    def finalize(self, eps=1e-4):
        pass


# This returns the correlation matrix
class CovarianceMetric:
    name = "covariance"

    # expects [batch, seq_len, hidden_dim]
    def calculate(self, feats):
        dataset_size = feats.shape[0]
        feats = feats.view(-1, feats.shape[-1])

        # compute the covariance matrix
        return torch.corrcoef(feats.T)


def remove_pads(attention_mask, feature_vector):
    batch_size, seq_length = attention_mask.shape
    if (
        len(feature_vector.shape) == 3
    ):  # Hidden states: (batch_size, seq_length, embedding_dim)
        # Expand mask to match the feature_vector dimensions and apply it
        expanded_mask = attention_mask.unsqueeze(-1)
        filtered_feature_vector = feature_vector * expanded_mask
    elif (
        len(feature_vector.shape) == 4
    ):  # Attention outputs: (batch_size, num_attention_heads, seq_length, seq_length)
        # Expand mask for application to attention outputs
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
        # Apply mask to the "keys" dimension of the attention scores
        filtered_feature_vector = feature_vector * expanded_mask
        # Apply mask to the "queries" dimension of the attention scores (transpose mask application)
        expanded_mask_transposed = attention_mask.unsqueeze(1).unsqueeze(2)
        filtered_feature_vector = filtered_feature_vector * expanded_mask_transposed
    else:
        raise ValueError("Unsupported feature vector shape.")

    return filtered_feature_vector
