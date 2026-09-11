# Defining Merge Methods

Merge methods are ordinary, synchronous tensor operations. They do not depend on
MergeKit's configuration format, model loaders, or computation graph. Those systems
adapt their inputs to the same callable method interface used by in-memory callers.

## A minimal method

```python
import torch

from mergekit.merge_methods import PerInput, Shared, TensorGroup
from mergekit.merge_methods.easy_define import merge_method


@merge_method(name="weighted_average", pretty_name="Weighted Average")
def weighted_average(
    group: TensorGroup,
    weight: PerInput[float],
    normalize: Shared[bool] = True,
) -> torch.Tensor:
    tensors = [entry.tensor for entry in group.entries]
    weights = weight.values_for(group.entries)
    weight_tensor = torch.tensor(
        weights, dtype=tensors[0].dtype, device=tensors[0].device
    )
    while weight_tensor.dim() <= tensors[0].dim():
        weight_tensor.unsqueeze_(-1)
    result = (torch.stack(tensors) * weight_tensor).sum(0)
    if normalize:
        result /= weight_tensor.sum(0)
    return result
```

The function signature is the single source of truth for parameter names, types,
scopes, defaults, and requiredness:

- `Shared[T]` is one value shared by the inputs to an output tensor.
- `PerInput[T]` is one value for each input. At runtime it is an ordered, immutable
  mapping keyed by `TensorEntry.id`.
- `PerNonBase[T]` is the same, but only binds values for non-base inputs.
- A parameter without a default is required.

Scopes do not encode the YAML hierarchy. For example, a shared parameter may still be
overridden per module or slice and may vary between output tensors through filters or
gradients. Scope describes the input axis along which the resolved value is bound.

Unsupported or ambiguous annotations are rejected when the method is defined. Resolved
values are validated against their annotations before any tensor math runs.

## Input contracts

Structural requirements belong to the method definition:

```python
from mergekit.merge_methods import BasePolicy, InputContract


@merge_method(
    name="base_interpolation",
    contract=InputContract(
        base=BasePolicy.REQUIRED,
        min_inputs=2,
        max_inputs=2,
        min_non_base=1,
        max_non_base=1,
    ),
)
def base_interpolation(
    group: TensorGroup,
    t: Shared[float],
) -> torch.Tensor:
    return (1 - t) * group.base.tensor + t * group.non_base[0].tensor
```

The available base policies are `IGNORED`, `OPTIONAL`, `REQUIRED`, and `FORBIDDEN`.
Input and non-base arities can be bounded independently. MergeKit validates contracts
while planning a configured merge and validates every group in a direct batch before
executing any group.

## Calling a method directly

The decorator returns the registered callable method rather than the undecorated
implementation:

```python
from mergekit import merge_methods
from mergekit.merge_methods import MergeBatch

batch = MergeBatch.from_tensors(
    [tensor_a, tensor_b],
    ids=["model_a", "model_b"],
)

result = merge_methods.get("linear")(
    batch,
    weight={"model_a": 0.25, "model_b": 0.75},
)
merged_tensor = result.one()
```

Sequences may be used for per-input values when their order matches the tensor entries.
Mappings are preferable whenever the inputs already have stable IDs. Scalar per-input
defaults and scalar arguments are broadcast.

Modules and state dictionaries also have a convenience adapter:

```python
from mergekit.merge_methods import merge_state_dicts

merged = merge_state_dicts(
    {"a": model_a, "b": model_b},
    "linear",
    parameters={"weight": {"a": 0.25, "b": 0.75}},
)
```

By default this validates that all state dictionaries have the same keys before
executing the batch. Pass `strict=False` to merge their intersection.

## Batches

A `TensorGroup` contains the inputs for one logical output tensor. A `MergeBatch`
contains one or more groups, and invoking a method returns a `MergedBatch` with one
output per group.

Most method implementations are group kernels and are lifted over a batch by the
registered method object. This keeps heterogeneous tensor shapes possible and gives
the interface room for chunked state-dict and in-memory model consumers.

Ordinary parameters broadcast across groups. Use `PerGroupValues` to explicitly vary
a parameter along the outer batch axis; the wrapper avoids ambiguity with shared
list-valued parameters.

## Metadata and base inputs

`TensorEntry.id` is an opaque hashable identifier. The core method API does not require
`ModelReference`; direct callers can use strings or integer positions.

`TensorGroup.metadata` carries lightweight output information such as its name and
whether it is an embedding tensor. A base is represented by an entry with
`is_base=True`, accessible through `group.base` and excluded from `group.non_base`.

## Registration

Decorator-based methods register when their module is imported. Add the module import
to `mergekit/merge_methods/__init__.py` for built-in methods. Static families such as
TIES/DARE may construct `MergeMethodSpec` objects programmatically when their available
parameters depend on a registered variant profile.

## Execution adapters

The computation graph uses a generic `ExecuteMergeMethodTask`; individual methods do
not define tasks. The YAML planner resolves configured values and builds that adapter.
Other consumers can construct `MergeBatch` directly without importing graph or config
types.
