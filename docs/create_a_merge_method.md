# Creating Custom Merge Methods

This guide explains two approaches to creating custom model merging algorithms in mergekit:

## 1. Decorator-based API (Recommended)

For most use cases, the `@merge_method` decorator provides a concise way to define merge logic.

### Example: Average Merge

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

This enables merge configurations like:
```yaml
merge_method: average
parameters:
  normalize: true
tensor_parameters:
  weights: [0.3, 0.7]
```

### Key Features:
- **Automatic parameter handling** - Type annotations define config options
- **Base model support** - Optional `base_tensor` parameter
- **Validation** - Built-in type checking and error reporting

## 2. Class-based API (Advanced)

For complex merges requiring explicit control, implement `MergeMethod` and `Task`:

### Example: Linear Merge

```python
from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.graph import Task

class LinearMergeTask(Task[torch.Tensor]):
    def execute(self, tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
        return sum(t * w for t,w in zip(tensors, weights))

class LinearMerge(MergeMethod):
    def parameters(self) -> list[ConfigParameterDef]:
        return [ConfigParameterDef("normalize", required=False, default=True)]
        
    def make_task(self, tensors: list[torch.Tensor], parameters: dict) -> Task:
        return LinearMergeTask(tensors, parameters)
```

### When to Use This Approach:
- Need direct control over tensor loading/permutation
- Require custom task dependencies
- Implementing complex weight calculations
- Handling non-standard model architectures

## Parameter Types

| Type          | Decorator Annotation | Class-based Equivalent       |
|---------------|----------------------|------------------------------|
| Scalar        | `float`/`int`/`bool` | `ConfigParameterDef`         |
| Tensor        | `list[float]`        | Per-model `tensor_parameters`|
| Base Model    | `base_tensor` param  | `base_model` reference       |

## Key Implementation Details

For class-based merges:
1. Subclass `MergeMethod` and implement:
   - `make_task()` - Create computation tasks
   - `parameters()` - Define config options
   
2. Create `Task` subclass(es) implementing:
   - `execute()` - Core merge logic
   - `arguments()` - Task dependencies
   - `group_label()` - Execution grouping

For decorator-based merges:
- Input tensors are automatically normalized
- Base model handling is automatic
- Parameter validation happens at config load
- Tensor permutation handled by framework

## Built-in Examples

1. **Linear Merge** (`mergekit.merge_methods.linear`)
   - Class-based implementation
   - Weighted sum with normalization

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`)  
   - Decorator-based hypersphere interpolation
   - Exponential map projection

## Choosing an Approach

|                      | Decorator | Class-based |
|----------------------|-----------|-------------|
| Learning Curve       | Easy      | Moderate    |
| Flexibility          | Moderate  | High        |
| Boilerplate          | None      | Some        |
| Access to Low-Levels | Limited   | Full        |

> **Note:** Mergekit automatically handles tensor shape matching for both approaches. Implementations can assume consistent dimensions except when merging embeddings/output layers.
