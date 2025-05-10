# Extending MergeKit with Custom Merge Methods

## Overview

MergeKit offers two different paths for implementing custom merge methods:

|                        | Decorator API         | Class-based API                                |
| ---------------------- | --------------------- | ---------------------------------------------- |
| **Complexity**         | Simple function-based | Full class implementation                      |
| **Abstraction Level**  | Higher-level          | Lower-level                                    |
| **Parameter Handling** | Automatic validation  | Manual configuration                           |
| **Execution Flow**     | Single function       | Arbitrary computation graph                    |
| **Best For**           | Most merge methods    | Complex multi-stage, multi-input strategies    |

Either approach benefits from MergeKit's underlying task system for resource management and execution control. The question of which to use largely depends on the complexity of the merge operation and the level of control needed.

**Note on Parameter Configuration:** MergeKit uses a hierarchical YAML-based configuration system. Parameters for your custom merge methods (both scalar and per-model) can be defined at various levels (e.g., globally, per-model, per-slice). The values your merge function or task receives are resolved by MergeKit based on this hierarchy and context. For full details on configuration structure and parameter precedence, please refer to the [Merge Configuration](../README.md#merge-configuration) section of the README.

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

- Parameter validation, type checking, and value resolution
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
3. Ensure the module containing your function is imported
   - For MergeKit to discover your decorated merge method, the Python module containing it must be imported during MergeKit's initialization. Add an import statement for your module in `mergekit/merge_methods/__init__.py`. Once the module is imported, the `@merge_method` decorator handles the registration of the method with MergeKit.

### Example: Weighted Average

```python
from mergekit.merge_methods.easy_define import merge_method
from typing import List
import torch

@merge_method(
    name="weighted_average",
    pretty_name="Weighted Average",            # Optional: human-readable name
    reference_url="https://example.com/docs",  # Optional: documentation or paper link
)
def average_merge(
    tensors: List[torch.Tensor],  # Required: input tensors
    weight: List[float],          # Vector parameter (one float per model)
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
parameters: # Global parameters
  normalize: true
```

### Parameter Types and Handling

The decorator supports three parameter categories:

1. **Scalar Parameters**
   - Types: `bool`, `float`, or `int`
   - Single value for all models
   - Without defaults they become required parameters
   - Example: `normalize: bool = True`

2. **Vector Parameters**
   - Types: `List[float]` or `List[int]` only
   - Configured per-model
   - Default values must be single numbers, not lists, as they are broadcasted
   - Example: `weights: List[float]`

3. **Base Model Integration**
    The `tensors: List[torch.Tensor]` argument and an optional `base_tensor` argument in your function signature interact with the `base_model` specified in the YAML configuration as follows:

    - **If your function includes a `base_tensor` parameter (e.g., `base_tensor: torch.Tensor` or `base_tensor: Optional[torch.Tensor]`):**
        - The `base_tensor` argument will receive the tensor from the `base_model` specified in the YAML. If annotated as `Optional` and no `base_model` is configured, it will be `None`.
        - The `tensors: List[torch.Tensor]` argument will *only* contain tensors from the models specified under the `models:` key in the YAML, in order. It will *not* include the base model's tensor.
    - **If your function does *not* include a `base_tensor` parameter:**
        - If a `base_model` is specified in the YAML, its tensor will be the *first element* in the `tensors: List[torch.Tensor]` list (i.e., `tensors[0]`).
        - Subsequent elements (`tensors[1:]`) will correspond to the models listed under the `models:` key in the YAML, in order.
        - If no `base_model` is specified, the `tensors` list will directly correspond to the models listed under the `models:` key.

4. **Special Auto-Populated Parameters**
    Certain parameter names in your function signature have special meaning and are auto-populated by MergeKit if present. You do not configure these directly in the YAML `parameters` sections for your method; MergeKit provides them.
    - `output_weight: WeightInfo`: If your function accepts an argument named `output_weight` annotated with `WeightInfo`, MergeKit will pass metadata about the specific weight tensor being computed.
    - `base_model: ModelReference` (or `Optional[ModelReference]`): If your function accepts `base_model` annotated with `ModelReference`, MergeKit will pass a reference to the base model if one is used in the configuration for this merge operation.

## Class-based API Implementation

For complex merges requiring granular control, implement `MergeMethod` and `Task` classes:

### Example Implementation

```python
from mergekit.merge_methods.base import MergeMethod, ConfigParameterDef
from mergekit.common import ImmutableMap, ModelReference, WeightInfo
from mergekit.graph import Task
from typing import Any, Dict, List
import torch


class CustomDependencyTask(Task[float]):
    totally_real_parameter: str

    # Example of a task that computes a dependency for the merge
    def execute(self) -> float:
        # Custom logic to compute a dependency
        return 42.0

class CustomMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    parameters: ImmutableMap[str, Any]
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {
            "tensors": self.gather_tensors,
            "dependency": CustomDependencyTask(totally_real_parameter="example"),
        }

    def priority(self) -> int:
        return 1  # Optional: higher priority = earlier execution

    def group_label(self) -> str:
        return self.weight_info.name  # Optional: modify task grouping

    def uses_accelerator(self) -> bool:
        return True  # Enable GPU acceleration

    def execute(
        self, tensors: Dict[ModelReference, torch.Tensor], dependency: float
    ) -> torch.Tensor:
        # Implementation using self.weight_info, self.parameters, and self.tensor_parameters
        # Access global parameters via self.parameters["param_name"]
        # Access per-model tensor parameters via self.tensor_parameters[model_ref]["tensor_param_name"]
        # These values are pre-resolved by MergeKit's configuration system.

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
        output_weight: WeightInfo, # Metadata about the weight being computed
        tensors: MergeTensorInput, # Internal Task that fetches input tensors
        parameters: ImmutableMap[str, Any], # Global parameters
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]], # Per-model parameters
        **kwargs, # Other context like base_model: Optional[ModelReference]
    ) -> Task:
        return CustomMergeTask(
            gather_tensors=tensors,
            parameters=parameters,
            tensor_parameters=tensor_parameters,
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
