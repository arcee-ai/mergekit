import torch
from abc import ABC, abstractmethod
import safetensors.torch import load_file, save_file

class MetricCalculator(ABC):
    
    @abstractmethod
    def update(self, batch_size, *feats, **aux_params):
        pass
    
    @abstractmethod
    def finalize(self, numel, eps=1e-4):
        pass

class CovarianceMetric(MetricCalculator):
    name = 'covariance'
    
    def __init__(self):
        self.std = None
        self.mean = None
        self.outer = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        feats = torch.nan_to_num(feats)
        
        mean = feats.mean(dim=1, keepdim=True)
        centered_feats = feats - mean
        outer = centered_feats @ centered_feats.T / feats.shape[1]
        
        if self.mean is None: self.mean = torch.zeros_like(mean)
        if self.outer is None: self.outer = torch.zeros_like(outer)
            
        self.mean += mean.squeeze() * batch_size
        self.outer += outer * batch_size
    
    def finalize(self, numel, eps=1e-4):
        cov = self.outer / numel - torch.outer(self.mean, self.mean) / numel**2
        return cov

class MeanMetric(MetricCalculator):
    name = 'mean'
    
    def __init__(self):
        self.mean = None
    
    def update(self, batch_size, *feats, **aux_params):
        feats = torch.cat(feats, dim=0)
        mean = feats.mean(dim=1)
        
        if self.mean is None: self.mean = torch.zeros_like(mean)
        self.mean += mean * batch_size
    
    def finalize(self, numel, eps=1e-4):
        return self.mean / numel

model1_layers = load_file('model1_safetensor.pth')
model2_layers = load_file('model2_safetensor.pth')

metric_classes = {'covariance': CovarianceMetric, 'mean': MeanMetric}
metrics = {layer_name: {metric_name: metric() for metric_name, metric in metric_classes.items()}
           for layer_name in model1_layers.keys()}

for layer_name in model1_layers.keys():
    layer1_data = model1_layers[layer_name].float()
    layer2_data = model2_layers[layer_name].float()
    
    if layer1_data.dim() == 2:
        layer1_data = layer1_data.unsqueeze(0)
    if layer2_data.dim() == 2:
        layer2_data = layer2_data.unsqueeze(0)
    
    concatenated_layers = torch.cat((layer1_data, layer2_data), dim=0)
    
    for metric in metrics[layer_name].values():
        metric.update(concatenated_layers.shape[0], concatenated_layers)
        
metric_results = {}
for layer_name, layer_metrics in metrics.items():
    metric_results[layer_name] = {metric_name: metric.finalize(concatenated_layers.numel()).numpy()
                                  for metric_name, metric in layer_metrics.items()}

# Saving the metrics as a SafeTensor
save_file(metric_results, 'nn_layer_metrics.st')

# Inform the user
print("Metrics have been saved as SafeTensors to 'nn_layer_metrics.st'.")