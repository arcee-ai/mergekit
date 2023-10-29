# Copyright (C) 2023 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

"""
Computational graph execution for tensor operations

This module provides a mechanism for constructing and executing a computational
graph for operations on tensors. The tensors are computed lazily,
being loaded and operated upon as per the defined computation graph
and execution strategy.

The primary class, `Executor`, uses a `RuleSet` to build a computation graph,
organizes an execution order which minimizes tensor resource requirements, and
executes the operations, handling tensor loading and storage automatically.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx
import torch
import tqdm
from pydantic import BaseModel
from typing_extensions import Protocol

from mergekit.common import ModelReference
from mergekit.lazy_tensors import LazyTensorLoader, TensorWriter


class TensorReference(BaseModel):
    """
    A reference to a tensor, optionally associated with a specific model.

    Attributes:
    - model: An optional reference to a language model.
    - key: A string identifier for the tensor.
    """

    model: Optional[ModelReference]
    key: str

    def __str__(self) -> str:
        if self.model is not None:
            namespace = str(self.model)
        else:
            namespace = "_"
        return namespace + ":" + self.key

    class Config:
        frozen = True


class Operation(BaseModel):
    """
    Defines a node in a computational graph, representing an operation on tensors.

    Attributes:
    - function: A string identifier for the operation to be performed.
    - inputs: A list of tensor inputs for this operation.
    - kwargs: Optional keyword arguments for the operation.
    """

    function: str
    inputs: List[TensorReference]
    kwargs: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True


class ProceduralRule(ABC):
    """
    Abstract base class for procedural rules. Procedural rules define a method for
    dynamically generating `Operation` instances that can produce a given
    `TensorReference`.
    """

    @abstractmethod
    def can_generate_rule(self, component: TensorReference) -> bool:
        ...

    @abstractmethod
    def generate_rule(self, component: TensorReference) -> Optional[Operation]:
        ...


class LoadTensorRule(ProceduralRule):
    """Rule for loading tensors from input models."""

    model: ModelReference
    tensor_paths: Dict[str, str]

    def __init__(
        self,
        model: ModelReference,
        tensor_paths: Dict[str, str],
        dtype: Optional[str],
    ):
        self.model = model
        self.tensor_paths = tensor_paths
        self.dtype = dtype

    def can_generate_rule(self, component: TensorReference) -> bool:
        return (
            isinstance(component, TensorReference)
            and component.model == self.model
            and component.key in self.tensor_paths
        )

    def generate_rule(self, component: TensorReference) -> Operation:
        if not self.can_generate_rule(component):
            return None
        return Operation(
            function="load_tensor",
            inputs=[],
            kwargs={
                "model": component.model,
                "key": component.key,
                "dtype": self.dtype,
            },
        )


class RuleSet:
    """
    A mapping from TensorReference instances to specific Operations to produce them.

    Can contain both statically defined rules and procedural rules for dynamic
    operation generation.
    """

    static: Dict[TensorReference, Operation]
    procedural: List[ProceduralRule]

    def __init__(
        self,
        static: Optional[Dict[TensorReference, Operation]] = None,
        procedural: Optional[List[ProceduralRule]] = None,
    ):
        self.static = static or {}
        self.procedural = procedural or []

    def get(self, tensor: TensorReference) -> Optional[Operation]:
        """
        Retrieve an operation to produce the specified tensor.

        First checks if a static operation exists for the given tensor reference.
        If not, iterates over procedural rules to find a match.
        """
        if tensor in self.static:
            return self.static[tensor]

        for proc_rule in self.procedural:
            if proc_rule.can_generate_rule(tensor):
                operation = proc_rule.generate_rule(tensor)
                if operation:
                    return operation
        return None


class OperationProtocol(Protocol):
    """Protocol for operation implementations."""

    def __call__(
        self, tensors: Dict[TensorReference, torch.Tensor], **kwargs
    ) -> Optional[torch.Tensor]:
        ...


def _normalized_shard_name(path: str) -> int:
    name, _ext = os.path.splitext(os.path.basename(path))
    return name.lower().replace("pytorch_model", "model")


class Executor:
    """
    The primary computation manager, organizing and executing tensor
    operations in a structured and resource-minimized manner.

    `Executor` takes in models, target tensor references, rules, and
    operation definitions to create and execute a computation graph.
    """

    rules: RuleSet
    loaders: Dict[ModelReference, LazyTensorLoader]
    targets: List[TensorReference]
    operations: Dict[str, OperationProtocol]
    low_cpu_memory: bool = False

    def __init__(
        self,
        models: List[ModelReference],
        targets: List[TensorReference],
        rules: RuleSet,
        operations: Optional[Dict[str, OperationProtocol]] = None,
        transformers_cache_dir: Optional[str] = None,
        lora_cache_dir: Optional[str] = None,
        dtype: Optional[str] = None,
        cuda: bool = False,
        low_cpu_memory: bool = False,
    ):
        if lora_cache_dir is None and transformers_cache_dir is not None:
            lora_cache_dir = transformers_cache_dir

        self.targets = targets
        self.loaders = {
            ref: LazyTensorLoader(
                ref.merged(cache_dir=lora_cache_dir).tensor_index(
                    cache_dir=transformers_cache_dir
                )
            )
            for ref in models
        }
        for model, loader in self.loaders.items():
            rules.procedural.append(
                LoadTensorRule(model, loader.index.tensor_paths, dtype=dtype)
            )

        if operations is None:
            operations = {}
        self.operations = operations
        self.rules = rules
        self.cuda = cuda
        self.low_cpu_memory = low_cpu_memory

    def run(self, out_path: str, max_shard_size: int, clone_tensors: bool = False):
        """
        Execute the computation graph and save results to disk.

        This method will generate the tensors as per the computation graph and save each
        tensor to the disk. Tensor computations are scheduled to minimize memory usage.

        Args:
            out_path (str): The path to the directory where the computed tensors will be saved.
            max_shard_size (int): The maximum size of each saved shard.
        """
        writer = TensorWriter(out_path, max_shard_size=max_shard_size)
        for ref, tensor in tqdm.tqdm(self.generate_tensors(), total=len(self.targets)):
            if not self.low_cpu_memory:
                tensor = tensor.cpu()

            writer.save_tensor(ref.key, tensor, clone=clone_tensors)

        writer.finalize()

    def generate_tensors(self) -> Iterator[Tuple[TensorReference, torch.Tensor]]:
        """
        Generate the specified target tensors.

        Builds the computational graph, schedules execution, then computes all tensors
        and yields each target tensor along with its reference. Tensors are kept or
        evicted from memory based on the last usage to optimize memory usage.

        Yields:
            Tuple[TensorReference, torch.Tensor]: A tensor reference and the corresponding computed tensor.
        """
        schedule = self._schedule_ops()

        # determine last usage of each tensor, so they can be evicted afterwards
        last_use = {}
        for idx, (component, _) in enumerate(schedule):
            for j in range(len(schedule) - 1, idx, -1):
                if component in schedule[j][1].inputs:
                    break
            last_use[component] = j

        tensors: Dict[TensorReference, torch.Tensor] = {}
        for idx, (component, operation) in enumerate(schedule):
            tensor_args = {}
            for ref in operation.inputs:
                value = tensors[ref]
                if self.cuda and value.device.type != "cuda":
                    value = value.cuda()
                tensor_args[ref] = value

            res = self._perform_operation(operation, tensor_args)
            del tensor_args

            if res is not None:
                tensors[component] = res

            if component in self.targets:
                yield (component, res)

            # evict unreferenced tensors
            expired = []
            for key in tensors:
                if idx >= last_use[key]:
                    expired.append(key)

            for key in expired:
                del tensors[key]

    def _perform_operation(
        self, operation: Operation, tensor_args: Dict[TensorReference, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Execute the given operation with the provided tensor arguments.

        Args:
            operation: The operation to execute.
            tensor_args: Mapping of tensor references to their actual values.
        """
        if operation.function == "load_tensor":
            return self._load_tensor(operation)

        elif operation.function in self.operations:
            return self.operations[operation.function](
                input_tensors=tensor_args, **operation.kwargs
            )

        else:
            raise RuntimeError(f"Unimplemented function {operation.function}")

    def _load_tensor(self, operation: Operation):
        """Load a tensor from an input model."""
        assert operation.function == "load_tensor"

        res = self.loaders[operation.kwargs["model"]].get_tensor(
            operation.kwargs["key"]
        )
        if operation.kwargs["dtype"]:
            res = res.to(dtype=operation.kwargs["dtype"])

        if self.cuda and self.low_cpu_memory:
            # immediately move to gpu so inputs don't accumulate in RAM
            res = res.cuda()
        return res

    def _compare_key(self, ref: TensorReference):
        """
        Generate a key for ordering computations.

        Aims to minimize the number of shards that must be resident in memory
        at any given time.
        """
        if ref.model:
            shard_key = _normalized_shard_name(
                self.loaders[ref.model].index.tensor_paths[ref.key]
            )
        else:
            shard_key = ""

        out_key = "" if ref in self.targets else "input"
        return (out_key, shard_key, ref.key)

    def _schedule_ops(self) -> List[Tuple[TensorReference, Operation]]:
        """
        Generate a schedule for executing tensor operations.

        Builds dependency graph for tensor computations and orders them in a manner
        that satisfies all dependencies while also minimizing memory usage.
        """
        dependencies, ops = self._build_dependencies()

        edge_tups = []
        for node in dependencies:
            for dependency in dependencies[node]:
                edge_tups.append((dependency, node))

        graph = networkx.DiGraph(edge_tups)
        res = list(
            networkx.lexicographical_topological_sort(graph, key=self._compare_key)
        )
        return [(r, ops[r]) for r in res]

    def _build_dependencies(
        self,
    ) -> Tuple[
        Dict[TensorReference, Set[TensorReference]], Dict[TensorReference, Operation]
    ]:
        """Build a dependency graph for the computation and select rules to
        produce each tensor."""
        dependencies: Dict[TensorReference, Set[TensorReference]] = {}
        ops: Dict[TensorReference, Operation] = {}

        def _visit(node: TensorReference):
            if node in ops:
                return

            operation = self.rules.get(node)
            if not operation:
                raise RuntimeError(f"No rule to produce {node}")
            ops[node] = operation

            dependencies[node] = set()
            for dependency in operation.inputs:
                dependencies[node].add(dependency)

            for dependency in operation.inputs:
                _visit(dependency)

        for target in self.targets:
            _visit(target)
        return dependencies, ops
