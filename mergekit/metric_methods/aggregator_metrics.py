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
        self.max_iterations = 25
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
    def __init__(self):
        self.kernel_functions = {
            'inner_product': self.inner_product,
            'rbf': self.rbf
        }
    
    def inner_product(self, X):
        return X @ X.T
    
    def rbf(X, sigma=None):
        GX = torch.mm(X, X.t())
        diag_GX = torch.diag(GX).unsqueeze(1)
        KX = diag_GX - GX + (diag_GX - GX).t()
        
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
            
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        
        return KX
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n]).to(K.device)
        I = torch.eye(n).to(K.device)
        H = I - unit / n
        return H @ K @ H
    
    def hsic(self, K_x, K_y):
        """
        Hilbert-Schmidt Independence Criterion
        Input: K_x, K_y: *Centered* Kernel matrices

        Returns: HSIC(K_x, K_y)
        """
        return torch.trace(K_x.T @ K_y) / ((K_x.shape[0]-1) ** 2)

    def cka(self, X, Y, kernel_function='inner_product'):
        K_x = self.kernel_functions[kernel_function](X)
        K_y = self.kernel_functions[kernel_function](Y)

        K_x = self.centering(K_x)
        K_y = self.centering(K_y)

        hsic_xy = self.hsic(K_x, K_y)
        hsic_xx = self.hsic(K_x, K_x)
        hsic_yy = self.hsic(K_y, K_y)

        return hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    
    def align(self, X, Y, knn_x, knn_y):
        """
        Input: X, Y: Centered Kernel matrices
        """
        assert X.shape == Y.shape
        num_rows, num_cols = X.shape
        rows, cols = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing='ij')
        
        # Check if each element in the meshgrid is a mutual nearest neighbor
        mutual_nn_mask = torch.isin(rows, knn_x.indices[cols]) & \
                        torch.isin(cols, knn_y.indices[rows])

        trace_xy = torch.trace(X.T @ Y)
        return mutual_nn_mask * trace_xy

    def cknna(self, X, Y, kernel_function='inner_product', k=5):
        K_x = self.kernel_functions[kernel_function](X)
        K_y = self.kernel_functions[kernel_function](Y)

        K_x = self.centering(K_x)
        K_y = self.centering(K_y)

        k_nearest_neighbors_x = torch.topk(K_x, k=k, dim=1, largest=True, sorted=False)
        k_nearest_neighbors_y = torch.topk(K_y, k=k, dim=1, largest=True, sorted=False)

        align_xy = self.align(K_x, K_y, k_nearest_neighbors_x, k_nearest_neighbors_y)
        align_xx = self.align(K_x, K_x, k_nearest_neighbors_x, k_nearest_neighbors_x)
        align_yy = self.align(K_y, K_y, k_nearest_neighbors_y, k_nearest_neighbors_y)

        return align_xy / torch.sqrt(align_xx * align_yy)

class CKA(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.cka = _CKA()
        self.batches_a = []
        self.batches_b = []
        self.stop = False
        self.max_batches = 20

        self.valid_for.update({
            LayerComparisonType.BLOCK.value: True,
            LayerComparisonType.CORRESPONDING.value: True,
            LayerComparisonType.ALL.value: True
        })

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        if not self.stop:
            self.batches_a.append(batch_a)
            self.batches_b.append(batch_b)
        
            if len(self.batches_a) >= self.max_batches:
                self.stop = True 
        
    def aggregate(self) -> Metric:
        result = self.cka.cka(torch.concat(self.batches_a), 
                                          torch.concat(self.batches_b))
        
        self.__init__(self.device) # Reset ready for next layer
        return Metric(mean_std=MeanStd(mean=result))
    
class CNNKA(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.cka = _CKA()
        self.batches_a = []
        self.batches_b = []
        self.stop = False
        self.max_batches = 20

        self.valid_for.update({
            LayerComparisonType.BLOCK.value: True,
            LayerComparisonType.CORRESPONDING.value: True,
            LayerComparisonType.ALL.value: True
        })

    def process_batch(self, batch_a: torch.Tensor, batch_b: torch.Tensor) -> None:
        if not self.stop:
            self.batches_a.append(batch_a)
            self.batches_b.append(batch_b)
        
            if len(self.batches_a) >= self.max_batches:
                self.stop = True 
        
    def aggregate(self) -> Metric:
        result = self.cka.cknna(torch.concat(self.batches_a), 
                                          torch.concat(self.batches_b))
        
        self.__init__(self.device) # Reset ready for next layer
        return Metric(mean_std=MeanStd(mean=result))

class t_SNE(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.tsne = TSNE(n_components=2, random_state=42)
        self.batches = []
        self.max_batches = 20
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

class PCA_Projection(MetricAggregator):
    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)
        self.batches = []
        self.max_batches = 20
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
        data = torch.cat(self.batches, dim=0)
        mean = torch.mean(data, dim=0)
        data -= mean
        U, S, V = torch.pca_lowrank(data, q=2)
        result = torch.matmul(data, V[:, :2])
        result = result.cpu().numpy() 
        metric = Metric(
            scatter_plot=ScatterPlot(
                x=result[:, 0],
                y=result[:, 1],
            )
        )
        self.__init__(self.device) # Reset ready for next layer
        return metric
     
METRICS_TABLE = {
    'cosine_similarity': Cosine_Similarity,
    'mse': MSE,
    'linearity_score': Linearity_Score,
    'cka': CKA,
    'cknna': CNNKA,
    't-sne': t_SNE,
    'pca_projection': PCA_Projection
    }