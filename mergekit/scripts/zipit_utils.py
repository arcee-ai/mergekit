# zip_it_metrics.py
import torch
from abc import ABC, abstractmethod

class MetricCalculator(ABC):
    
    @abstractmethod
    def update(self, feats):
        pass
    
    @abstractmethod
    def finalize(self, eps=1e-4):
        pass

class CovarianceMetric(MetricCalculator):
    name = 'covariance'
    
    def __init__(self):
        super().__init__()
        self.mean = None
        self.outer = None
        self.dataset_size = 0  # Keep track of the total number of samples
    
    def update(self, feats):

        self.dataset_size = feats.shape[0]

        std = feats.std(dim=1)
        mean = feats.mean(dim=0, keepdim=True)
        centered_feats = feats - mean
        outer = centered_feats.T @ centered_feats 
        
        if self.mean is None: 
            self.mean = torch.zeros_like(mean)
        if self.outer is None: 
            self.outer = torch.zeros_like(outer)
    
    def finalize(self, eps=1e-4):
        mean = self.mean.squeeze()  # Remove singleton dimensions to make mean 1-D
        cov = self.outer - torch.outer(mean, mean) / self.dataset_size
        return cov

def remove_pads(attention_mask, feature_vector):
    batch_size, seq_length = attention_mask.shape
    if len(feature_vector.shape) == 3:  # Hidden states: (batch_size, seq_length, embedding_dim)
        # Expand mask to match the feature_vector dimensions and apply it
        expanded_mask = attention_mask.unsqueeze(-1)
        filtered_feature_vector = feature_vector * expanded_mask
    elif len(feature_vector.shape) == 4:  # Attention outputs: (batch_size, num_attention_heads, seq_length, seq_length)
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
