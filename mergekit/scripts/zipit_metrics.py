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