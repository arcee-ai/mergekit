# Creating Custom Merge Methods

This guide explains how to create custom model merging algorithms using mergekit's decorator-based API.

## Simple Example: Average Merge

```python
from mergekit.merge_methods import merge_method
import torch

@merge_method(
    name="average",
    pretty_name="Simple Average",
    reference_url="https://example.com/mean-merge"
)
def average_merge(
    tensors: list[torch.Tensor],  # Input tensors to merge
    weights: list[float],         # Per-model weights (tensor parameter)
    normalize: bool = True        # Scalar parameter
) -> torch.Tensor:
    if normalize:
        total = sum(weights)
        weights = [w/total for w in weights]
    
    return sum(t * w for t, w in zip(tensors, weights))
```

This would enable merge configurations like:
```yaml
merge_method: average
parameters:
  normalize: true
tensor_parameters:
  weights: [0.3, 0.7]
```

## Parameter Types

### Scalar Parameters
- Defined as `float`, `int`, or `bool` types
- Configured in top-level `parameters` section
- Example: `normalize` in above example

### Tensor Parameters
- Defined as `list[float]` or `list[int]`
- Configured per-model in `parameters` section
- Automatically collected in model order
- Example: `weights` in above example

## Advanced Usage

### Base Model Handling
```python
def merge_with_base(
    tensors: list[torch.Tensor],
    base_tensor: torch.Tensor,  # Optional base model tensor
    strength: float = 0.5
) -> torch.Tensor:
    return base_tensor * strength + sum(tensors) * (1-strength)
```

Two approaches:
1. Base model first in tensor list (default)
2. Explicit `base_tensor` parameter

### Input Validation
```python
def safe_merge(
    tensors: list[torch.Tensor],
    weights: list[float]
) -> torch.Tensor:
    if len(tensors) != len(weights):
        raise ValueError("Weights/tensors count mismatch")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")
    
    return sum(t*w for t,w in zip(tensors, weights))
```

## Built-in Examples

1. **Linear Merge** (`mergekit.merge_methods.linear`)
   - Basic weighted sum of tensors
   - Handles parameter normalization

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`) 
   - Hypersphere interpolation
   - Uses exponential map projection

## How It Works

The `@merge_method` decorator:
1. Analyzes function signature
2. Generates configuration classes
3. Handles tensor loading/permutation
4. Manages parameter validation
5. Registers method in mergekit system

All boilerplate is handled automatically - focus on the core merge logic!

> **Note:** Input tensors will have matching shapes when your function is called. Mergekit handles dimension mismatches for embeddings and output layers automatically.
