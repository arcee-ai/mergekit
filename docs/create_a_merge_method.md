# Extending MergeKit with Custom Merge Methods

## Overview

MergeKit offers two different paths for implementing custom merge methods:

|                        | Decorator API         | Class-based API             |
| ---------------------- | --------------------- | --------------------------- |
| **Complexity**         | Simple function-based | Full class implementation   |
| **Abstraction Level**  | Higher-level          | Lower-level                 |
| **Parameter Handling** | Automatic validation  | Manual configuration        |
| **Execution Flow**     | Single-step           | Arbitrary computation graph |
| **Best For**           | Simple tensor ops     | Complex merge strategies    |

Either approach benefits from MergeKit's underlying task system for resource management and execution control. The question of which to use largely depends on the complexity of the merge operation and the level of control needed.

### Core Task System Features

MergeKit's computational graph infrastructure provides sophisticated resource management that all merge methods inherit:

- **Smart Memory Management**
  - Automatic return value lifecycle tracking
  - Early value eviction when no longer needed
  - Optimized shard loading based on task groups

- **Device Management**
  - Automatic tensor movement between compute and storage devices
  - Support for both CPU and GPU execution

- **Task Scheduling**
  - Tasks grouped by tensor shard to minimize memory usage
  - Loads deferred until last possible moment (via priority system)
  - Execution ordered to optimize shard residency

### Decorator API
Best for straightforward merge operations that can be expressed as a single tensor transformation. Features:
- Parameter validation and type checking
- Configuration schema generation
- Simplified base model handling
- Default GPU acceleration opt-in

### Class-based API
Choose when you need:
- Multi-stage merge operations
- Custom computation graphs
- Direct access to weight metadata
- Complex parameter types
- Fine-grained control over execution

## Decorator API Implementation

### Basic Workflow
1. Define a type-annotated Python function with your merge logic
2. Add the `@merge_method` decorator with configuration
3. Register by importing in `mergekit/merge_methods/__init__.py`

### Example: Weighted Average

```python
from mergekit.merge_methods.easy_define import merge_method
from typing import List
import torch

@merge_method(
    name="weighted_average",
    pretty_name="Weighted Average",            # Optional: human-readable name
    reference_url="https://example.com/docs",  # Optional: documentation link
)
def average_merge(
    tensors: List[torch.Tensor],  # Required: input tensors
    weight: List[float],          # Vector parameter (per-model)
    normalize: bool = True,       # Scalar parameter with default
) -> torch.Tensor:
    if normalize:
        total = sum(weight)
        weight = [w / total for w in weight]

    return sum(t * w for t, w in zip(tensors, weight))
```

This enables configurations like:
```yaml
merge_method: weighted_average
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

### Parameter Types and Handling

The decorator supports three parameter categories:

1. **Scalar Parameters**
   - Types: `bool`, `float`, or `int`
   - Defined in top-level `parameters` section
   - Without defaults they become required parameters
   - Example: `normalize: bool = True`

2. **Vector Parameters**
   - Types: `List[float]` or `List[int]` only
   - Configured per-model in their `parameters` section
   - Default values must be single numbers, not lists, as they are broadcasted
   - Example: `weights: List[float]`

3. **Base Model Integration**
   - Via `base_tensor` parameter annotation:
     * `torch.Tensor`: Base model required
     * `Optional[torch.Tensor]`: Base model optional
   - Without `base_tensor`: Base model tensor goes first in `tensors` list if present

## Class-based API

For complex merges requiring granular control, implement `MergeMethod` and `Task` classes:

### Example Implementation

```python
from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from typing import Any, Dict, List


class CustomMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    parameters: ImmutableMap[str, Any]
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def priority(self) -> int:
        return 1  # Optional: higher priority = earlier execution

    def group_label(self) -> str:
        return self.weight_info.name  # Optional: modify task grouping

    def uses_accelerator(self) -> bool:
        return True  # Enable GPU acceleration

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # Implementation using weight info and parameters
        result = ...
        return result


class CustomMerge(MergeMethod):
    def name(self) -> str:
        return "custom_merge"

    def pretty_name(self) -> str:
        return "Custom Merge"

    def reference_url(self) -> str:
        return "https://example.com/custom"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef("threshold", float, required=False, default_value=0.5)
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef("weight", float, required=True)]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task:
        return CustomMergeTask(
            gather_tensors=tensors,
            parameters=parameters,
            weight_info=output_weight,
        )
```

### Task Scheduling System

The class-based API provides fine-grained control over execution:

- **Priority Control**: Override `priority()` to influence execution order within groups
- **Task Grouping**: Use `group_label()` to batch similar operations
- **Resource Management**:
  - Automatic tensor lifecycle tracking
  - Memory optimization via early tensor eviction
  - Smart device placement for computation vs storage
- **Computation Graph**: Build complex flows by connecting multiple tasks

### Implementation Requirements

1. Task Class:
   - Must implement `execute()` with proper type annotations
   - Must implement `arguments()` to declare dependencies
   - Optionally override `priority()`, `group_label()`, `uses_accelerator()`

2. Method Class:
   - Must implement core methods: `name()`, `make_task()`
   - Optional methods: `pretty_name()`, `reference_url()`
   - Define parameters via `parameters()` and `tensor_parameters()`

### Registration

Add class-based methods to `STATIC_MERGE_METHODS` in `mergekit/merge_methods/registry.py`:

```python
from mergekit.merge_methods.my_module import CustomMerge

STATIC_MERGE_METHODS: List[MergeMethod] = [
    CustomMerge(),
    # other methods...
]
```

## Reference Implementations

1. **Linear Merge** (`mergekit.merge_methods.linear`):
   - Basic weighted averaging
   - Good example of class-based implementation

2. **Multi-SLERP** (`mergekit.merge_methods.multislerp`):
   - Hypersphere interpolation
   - Complex decorator usage example

3. **Task Arithmetic** (`mergekit.merge_methods.task_arithmetic`):
   - Advanced graph-based implementation
   - TIES/Magnitude pruning example
