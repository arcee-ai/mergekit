# Extending MergeKit with Custom Merge Methods

This guide explains how to create custom model merging algorithms for MergeKit. You'll learn two distinct approaches - choose based on your needs:

## Choosing an Implementation Approach

Choose between the two approaches based on your needs:

**Use the Decorator API if you need:**
* Simple weighted combinations or mathematical operations
* A single logical step in the merge process
* Automatic parameter validation and device management
* Quick implementation without complex dependencies

**Choose the Class-based API when you require:**
* Full control over the execution graph and tensor routing
* Multi-stage merge operations with intermediate steps
* Custom parameter types or complex validation logic
* Access to weight metadata or model architecture details
* Specialized hardware handling or accelerator usage

## 1. Quick Implementation with Decorator API

### When to Use This Approach
- You want maximum simplicity
- Your merge logic can be expressed as a single PyTorch function
- You don't need low-level control over the underlying computational graph
- You want automatic parameter validation

### Implementation Steps
1. Define your merge function with type annotations
2. Add `@merge_method` decorator with metadata
3. Handle base model optionally via `base_tensor` param
4. Return merged tensor

### Example: Average Merge

```python
from mergekit.merge_methods.easy_define import merge_method
from typing import List
import torch

@merge_method(
    name="simple_average",
    pretty_name="Simple Average",
    reference_url="https://example.com/mean-merge"
)
def average_merge(
    tensors: List[torch.Tensor],  # Input tensors to merge
    weight: List[float],          # Per-model weights (tensor parameter)
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
- **Parameter Validation** - Built-in type checking
- **Device Management** - Automatic GPU/CPU placement
- **Base Model Handling** - Presence of optional `base_tensor`/`base_model` parameters determine support for a base model

### Advanced Parameter Handling

The decorator supports three parameter types through type annotations:

1. **Scalar Parameters** (`bool|float|int`):
   - Defined in top-level `parameters` section
   - Can be optional or have default values
   - Example: `normalize: bool = True`

2. **Vector Parameters** (`list[float]|list[int]`):
   - Configured per-model in their `parameters` section
   - Will be passed as a list of floats or integers with the same length as `models`
   - Can be optional or have default values, like scalars
     - Note that default values must be a single value, not a list!
   - Example: `weight: List[float]` or `threshold: List[float] = 0.5`

3. **Base Model Handling** (`torch.Tensor`):
   - Automatic when method has `base_tensor` parameter:
     * `torch.Tensor` annotation: Requires `base_model` in config
     * `Optional[torch.Tensor]`: Base model is optional
   - Without `base_tensor`: Base model is supported only if `base_model` parameter is present, and if so the associated tensor will be placed first in the `tensors` list

Key requirements for decorated functions:
- Must have `tensors: List[torch.Tensor]` as first parameter
- Must return a single `torch.Tensor`
- All parameters must have type annotations

Implementation notes:
- Base model tensor (if used) is excluded from `tensors` list
- Decorated functions get automatic:
  * Parameter validation
  * Tensor shape alignment
  * Device placement

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

    def execute(self, tensors: Dict[ModelReference, torch.Tensor], **_kwargs) -> torch.Tensor:
        # Implementation using weight info and tensor parameters
        ...

class LinearMerge(MergeMethod):
    def name(self) -> str:
        return "linear"
        
    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("normalize", required=False, default_value=True)]

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

Note on tensor sizes: Implementations can assume consistent dimensions. Should the input tensors have different shapes, the user is doing something profane and will get what they deserve. (Or, you can check if the tensors are embeddings and truncate them to the smallest size. But justice can only be delayed, not denied.)

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
    tensors: MergeTensorInput
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

## Reference Implementations

1. **Linear Merge** (`mergekit.merge_methods.linear`):
   - Simple weighted average
   - Good starter example for class-based API

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`):
   - Hypersphere interpolation
   - Complex decorator usage

3. **Generalized Task Arithmetic** (`mergekit.merge_methods.generalized_task_arithmetic`):
   - Advanced class-based example
   - Implements TIES/Magnitude pruning
