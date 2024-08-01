import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch.nn.functional as F
import math

from mergekit.metric_methods.base import MeanStd, Heatmap, Histogram, Metric, Results, Layer, ScatterPlot
from mergekit.metric_methods.metrics import compute_histogram

from mergekit.architecture import WeightInfo
from sklearn.manifold import TSNE

import enum

class ModelAnalysisType(enum.Enum):
    INDIVIDUAL = "individual"
    COMPARISON = "comparison"

class LayerComparisonType(enum.Enum):
    SINGLE = "single" # Analyse Layer i
    BLOCK = "block" # Compare Layer i in model 1 with layer i+(block size) in model 1
    CORRESPONDING = "corresponding" # Compare Layer i in model 1 with layer i in model 2
    ALL = "all_layers" # Compare Layer i in model 1 with Layer j in model (1 or 2)

class MetricAggregator():
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.valid_for = {
            LayerComparisonType.SINGLE.value: False,
            LayerComparisonType.BLOCK.value: False,
            LayerComparisonType.CORRESPONDING.value: False,
            LayerComparisonType.ALL.value: False
        }

    def process_batch(self, batch_a: torch.Tensor, batch_b: Optional[torch.Tensor]) -> None:
        raise NotImplementedError

    def aggregate(self) -> Metric:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

class Cosine_Similarity(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.cosine_similarities = torch.tensor([], device=self.device)
        self.valid_for.update({
            LayerComparisonType.BLOCK.value: True,
            LayerComparisonType.CORRESPONDING.value: True,
            LayerComparisonType.ALL.value: True
        })

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        batch_similarities = F.cosine_similarity(batch_a, batch_b, dim=1)
        self.cosine_similarities = torch.cat((self.cosine_similarities, batch_similarities))

    def aggregate(self) -> Metric:
        hist = compute_histogram(self.cosine_similarities, 100)        
        mean_std=MeanStd(
                mean=self.cosine_similarities.mean().item(), 
                std=self.cosine_similarities.std().item()
                )
        histogram=Histogram(
                count=hist[0],
                edges=hist[1],
                widths=hist[2]
            )
        self.__init__()
        return Metric(
            histogram=histogram,
            mean_std=mean_std
        )

    def clear(self) -> None:
        self.cosine_similarities = torch.tensor([], device=self.device)

class MSE(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.square_errors = torch.tensor([], device=self.device)
        self.valid_for.update({
            LayerComparisonType.BLOCK.value: True,
            LayerComparisonType.CORRESPONDING.value: True,
        })

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        batch_square_errors = torch.square(batch_a - batch_b).flatten()
        self.square_errors = torch.cat((self.square_errors, batch_square_errors))

    def aggregate(self) -> Metric:
        hist = compute_histogram(self.square_errors, 100)
        mean_std=MeanStd(
                mean=self.square_errors.mean().item(),
                std=self.square_errors.std().item()
            )
        histogram=Histogram(
                count=hist[0],
                edges=hist[1],
                widths=hist[2]
            )
        self.__init__()
        return Metric(
            histogram=histogram,
            mean_std=mean_std
        )

class Linearity_Score(MetricAggregator):
    def __init__(self, device: str = "cpu"):

        super().__init__(device=device)
        self.iterations = 0
        self.max_iterations = 5
        self.A = None
        self.optimiser = None
        self.initialised = False
        self.done = False
        self.losses = []

        self.absolute_square_sum = 0
        self.num_elements = 0

        self.valid_for.update({
            LayerComparisonType.BLOCK.value: True
        })

    def _first_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        batch_size, dimension = batch_a.size()

        self.A = torch.empty(dimension, dimension, device=self.device)
        torch.nn.init.normal_(self.A)
        self.A = torch.nn.Parameter(self.A)

        self.optimiser = torch.optim.SGD([self.A], lr=0.0001)  
        self.initialised = True

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        batch_a = batch_a / torch.norm(batch_a, dim=1, keepdim=True)
        batch_b = batch_b / torch.norm(batch_b, dim=1, keepdim=True) # Check dimensionality (X)
        if not self.initialised:
            self._first_batch(batch_a, batch_b)
        if self.done: # stop training A and evaluate
            residuals = batch_a @ self.A - batch_b
            self.absolute_square_sum += torch.abs(residuals).sum().item()
            self.num_elements += residuals.numel()

        else:

            loss = torch.norm(batch_a @ self.A - batch_b) ** 2
            loss.backward()
            self.losses.append(loss.item())
            print(f'Loss: {loss.item()}')
            self.optimiser.step()

            self.iterations += 1

            if self.iterations >= self.max_iterations:
                self.done = True
    
    def aggregate(self) -> Metric:
        linearity_score = 1 - self.absolute_square_sum / self.num_elements
        self.__init__()
        return Metric(mean_std=MeanStd(mean=linearity_score)) 

    def clear(self) -> None:
        pass

class _CKA(object):
    # Class from https://github.com/jayroxis/CKA-similarity/blob/main/CKA.py
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(
            self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma))
        )

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

class CKA(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.cka = _CKA()
        self.batches_a = []
        self.batches_b = []
        self.stop = False
        self.max_batches = 10

        self.valid_for.update({
            LayerComparisonType.BLOCK.value: True,
            LayerComparisonType.CORRESPONDING.value: True,
            LayerComparisonType.ALL.value: True
        })

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        if not self.stop:
            self.batches_a.append(batch_a.cpu().numpy())
            self.batches_b.append(batch_b.cpu().numpy())
        
            if len(self.batches_a) >= self.max_batches:
                self.stop = True 
        
    def aggregate(self) -> Metric:
        self.result = self.cka.linear_CKA(np.concatenate(self.batches_a), 
                                          np.concatenate(self.batches_b))
        return Metric(mean_std=MeanStd(mean=self.result))

class t_SNE(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.tsne = TSNE(n_components=2, random_state=42)
        self.batches = []
        self.max_batches = 5
        self.stop = False

        self.valid_for.update({
            LayerComparisonType.SINGLE.value: True,
        })

    def process_batch(self, batch: torch.Tensor) -> None:
        if not self.stop:
            self.batches.append(batch.cpu().numpy())
        
            if len(self.batches) >= self.max_batches:
                self.stop = True 

    def aggregate(self) -> Metric:
        data = np.concatenate(self.batches)
        self.result = self.tsne.fit_transform(data)

        metric = Metric(
            scatter_plot=ScatterPlot(
                x=self.result[:, 0],
                y=self.result[:, 1],
            )
        )
        self.__init__(self.device) # Reset ready for next layer
        return metric
     
METRICS_TABLE = {
    'cosine_similarity': Cosine_Similarity,
    'mse': MSE,
    'linearity_score': Linearity_Score,
    'cka': CKA,
    't-sne': t_SNE
    }