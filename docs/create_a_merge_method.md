# Extending MergeKit with Custom Merge Methods

This guide explains how to create custom model merging algorithms for MergeKit. You'll learn two distinct approaches - choose based on your needs:

## Choosing an Implementation Approach

| Consideration       | Decorator API          | Class-based API       |
|---------------------|------------------------|-----------------------|
| Complexity          | Simple                 | Moderate              |
| Flexibility         | Limited                | Full control          |
| Access to Base Model| Optional `base_tensor` | Direct via reference  | 
| Parameter Types     | Scalar/Vector          | Any config structure  |
| Best For            | Most merges            | Research/Complex ops  |

## 1. Quick Implementation with Decorator API

### When to Use This Approach
- You want maximum simplicity
- Your merge logic can be expressed as weighted arithmetic
- You don't need low-level control over tensor loading
- You want automatic parameter validation

### Implementation Steps
1. Define your merge function with type annotations
2. Add `@merge_method` decorator with metadata
3. Handle base model optionally via `base_tensor` param
4. Return merged tensor

### Example: Weighted Average

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

### Key Features
- **Type-Driven Config** - Annotations auto-create config schema
- **Base Model Handling** - First model is base by default
- **Parameter Validation** - Built-in type checking
- **Device Management** - Automatic GPU/CPU placement

### Advanced Parameter Handling
```python
@merge_method(name="complex_example")
def complex_merge(
    tensors: List[torch.Tensor],
    layer_weight: List[float],  # Per-layer weights
    temperature: float = 1.0,   # Global scalar 
    base_tensor: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # Mix base tensor (if provided) with others
    if base_tensor is not None:
        tensors = [base_tensor] + tensors
    
    # Apply temperature scaling
    weights = torch.softmax(torch.tensor(layer_weight) / temperature, dim=0)
    return (torch.stack(tensors) * weights.view(-1, 1, 1)).sum(dim=0)
```

This supports config like:
```yaml
merge_method: complex_example
parameters:
  temperature: 0.5
tensor_parameters:
  layer_weight: [0.2, 0.3, 0.5] 
```

>> Parameter Types:
- **Scalar**: Single value (float/int/bool) 
- **Vector**: Per-layer weights (interpolated automatically)
- **Base Model**: Optional first input via `base_tensor`

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

Note on tensor sizes: Implementations can assume consistent dimensions.

## Parameter Types

| Type       | Decorator Annotation | Class-based Equivalent        |
| ---------- | -------------------- | ----------------------------- |
| Scalar     | `float`/`int`/`bool` | `ConfigParameterDef`          |
| Tensor     | `list[float]`        | Per-model `tensor_parameters` |
| Base Model | `base_tensor` param  | `base_model` reference        |

### Implementation Steps

1. Subclass `MergeMethod`:
```python
class CustomMerge(MergeMethod):
    def name(self) -> str: return "custom"
    
    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("threshold", float, required=True)]
    
    def make_task(self, output_weight, tensors, parameters, **kwargs) -> Task:
        return CustomTask(
            output_weight=output_weight,
            tensors=tensors,
            threshold=parameters["threshold"]
        )
```

2. Create Task subclass:
```python
class CustomTask(Task[torch.Tensor]):
    threshold: float
    output_weight: WeightInfo
    
    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}
    
    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # Merge logic here
        merged = ...
        return merged.clamp(-self.threshold, self.threshold)
    
    def group_label(self) -> str:
        return self.output_weight.name
```

### Key Methods to Implement

For `MergeMethod`:
- `name()`: Return unique identifier
- `parameters()`: List of config parameters
- `make_task()`: Factory for task instances

For `Task`:
- `arguments()`: Declare input dependencies
- `execute()`: Core computation logic
- `group_label()`: Task grouping for execution
- `uses_accelerator()`: Indicate GPU usage

### Tensor Considerations
1. Shapes: May differ for embeddings - use `rectify_embed_sizes()`
2. Devices: Tensors may be on CPU/GPU - use `.to()` if needed
3. DTypes: Preserve original types unless explicitly converting
4. Memory: Use `clone()` judiciously for large tensors

## Testing & Debugging Tips

1. Validate with Small Models:
```bash
mergekit-yaml config.yml output --cuda --low-cpu
```

2. Profile Execution:
```bash
PYTORCH_CUDA_ALLOC_CONF=backend python -m cProfile -o profile.stats your_script.py
```

3. Inspect Intermediate Tensors:
```python
class DebugTask(Task):
    def execute(self, **inputs):
        print(f"Merging {self.weight_info.name}")
        for k,v in inputs.items():
            print(f"{k}: {v.shape} {v.dtype}")
        return super().execute(**inputs)
```

## Reference Implementations

1. **Linear Merge** (`mergekit.merge_methods.linear`):
   - Simple weighted average
   - Good starter example

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`):
   - Hypersphere interpolation
   - Complex decorator usage

3. **Generalized Task Arithmetic** (`mergekit.merge_methods.generalized_task_arithmetic`):
   - Advanced class-based example
   - Implements TIES/Magnitude pruning

## Next Steps

1. Study existing merge methods
2. Start with decorator API for simple merges
3. Use class-based API for research-level techniques
4. Submit PRs for generally useful methods!
