# Creating Custom Merge Methods

This guide explains two approaches to creating custom model merging algorithms in mergekit:

## 1. Decorator-based API (Recommended)

For most use cases, the `@merge_method` decorator provides a concise way to define merge logic.

### Example: Average Merge

```python
from mergekit.merge_methods import merge_method
import torch

@merge_method(
    name="simple_average",
    pretty_name="Simple Average",
    reference_url="https://example.com/mean-merge"
)
def average_merge(
    tensors: list[torch.Tensor],  # Input tensors to merge
    weight: list[float],          # Per-model weights (tensor parameter)
    normalize: bool = True        # Scalar parameter
) -> torch.Tensor:
    if normalize:
        total = sum(weight)
        weight = [w/total for w in weight]
    
    return sum(t * w for t, w in zip(tensors, weight))
```

This enables merge configurations like:
```yaml
merge_method: simple_average
models:
  - model: model1
    parameters:
      weight: 0.3
  - model: model2
    parameters:
      weight: 0.7
parameters:
  normalize: true
```

### Key Features:
- **Automatic parameter handling** - Type annotations define config options
- **Base model support** - Optional `base_tensor` parameter
- **Validation** - Built-in type checking and error reporting

## 2. Class-based API (Advanced)

For complex merges requiring explicit control, implement `MergeMethod` and `Task` subclasses:

### Example: Linear Merge

```python
from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import MergeTensorInput
from mergekit.architecture import WeightInfo

class LinearMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    normalize: bool
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # Implementation using weight info and tensor parameters
        ...

class LinearMerge(MergeMethod):
    def name(self) -> str:
        return "linear"
        
    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("normalize", required=False, default=True)]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("weight", required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task:
        return LinearMergeTask(
            gather_tensors=tensors,
            tensor_parameters=tensor_parameters,
            normalize=parameters["normalize"],
            weight_info=output_weight,
        )
```

### Key Implementation Details
For class-based merges:
1. Subclass `MergeMethod` and implement:
   - `name()` - Return unique method identifier
   - `make_task()` - Create computation tasks with proper typing
   - `parameters()` - Define config options
   - `tensor_parameters()` - Define per-model tensor parameters
   
2. Create `Task` subclass implementing:
   - `execute()` - Core merge logic with type annotations
   - `arguments()` - Declare task dependencies
   - `group_label()` - (Optional) Task grouping for execution
   - `uses_accelerator()` - Indicate GPU acceleration support

3. Handle tensor parameters through `tensor_parameters` argument to `make_task`

Note on tensor sizes: Implementations can assume consistent dimensions except when merging embeddings/output layers. Use `rectify_embed_sizes` helper when merging embedding layers.

## Parameter Types

| Type       | Decorator Annotation | Class-based Equivalent        |
| ---------- | -------------------- | ----------------------------- |
| Scalar     | `float`/`int`/`bool` | `ConfigParameterDef`          |
| Tensor     | `list[float]`        | Per-model `tensor_parameters` |
| Base Model | `base_tensor` param  | `base_model` reference        |

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
| -------------------- | --------- | ----------- |
| Learning Curve       | Easy      | Moderate    |
| Flexibility          | Moderate  | High        |
| Boilerplate          | None      | Some        |
| Access to Low-Levels | Limited   | Full        |

> **Implementation Notes:**
> - All tensor operations should preserve device/dtype attributes
> - The first dimension is typically batch dimension and should not be averaged/summed
> - Use `rectify_embed_sizes` from `mergekit.merge_methods.rectify_embed` when merging embedding layers
> - Return tensors should match base model dtype unless explicitly converting
