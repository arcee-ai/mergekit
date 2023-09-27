from abc import ABC, abstractmethod
import os.path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from pydantic import BaseModel
from typing_extensions import Protocol, TypeAlias

import torch
from common import ModelReference
from collections import defaultdict
import re
import functools
import graphlib
import networkx
import safetensors.torch

from lazy_tensors import LazyTensorLoader


class TensorReference(BaseModel):
    model: Optional[ModelReference]
    key: str

    def __str__(self) -> str:
        if self.model is not None:
            ns = str(self.model)
        else:
            ns = "_"
        return ns + ":" + self.key

    class Config:
        frozen = True


class InputShard(BaseModel):
    path: str

    class Config:
        frozen = True


ModelComponent: TypeAlias = Union[TensorReference, InputShard]


class Operation(BaseModel):
    function: str
    inputs: List[ModelComponent]
    kwargs: Optional[Dict[str, Any]] = None

    class Config:
        frozen = True


class ProceduralRule(ABC):
    @abstractmethod
    def can_generate_rule(self, component: ModelComponent) -> bool:
        ...

    @abstractmethod
    def generate_rule(self, component: ModelComponent) -> Optional[Operation]:
        ...


class LoadTensorRule(ProceduralRule):
    model: ModelReference
    tensor_paths: Dict[str, str]

    def __init__(self, model: ModelReference, tensor_paths: Dict[str, str]):
        self.model = model
        self.tensor_paths = tensor_paths

    def can_generate_rule(self, component: ModelComponent) -> bool:
        return (
            isinstance(component, TensorReference)
            and component.model == self.model
            and component.key in self.tensor_paths
        )

    def generate_rule(self, component: ModelComponent) -> Operation:
        if not self.can_generate_rule(component):
            return None
        return Operation(
            function="load_tensor",
            inputs=[],
            kwargs={"model": component.model, "key": component.key},
        )


class RuleSet:
    static: Dict[ModelComponent, Operation]
    procedural: List[ProceduralRule]

    def get(self, component: ModelComponent) -> Optional[Operation]:
        if component in self.static:
            return self.static[component]

        for p in self.procedural:
            if p.can_generate_rule(component):
                op = p.generate_rule(component)
                if op:
                    return op
        return None


def compare_component_key(a: ModelComponent):
    if isinstance(a, TensorReference):
        return (True, str(a))
    return (False, a.path)


class OperationProtocol(Protocol):
    def __call__(
        self, tensors: Dict[TensorReference, torch.Tensor], **kwargs
    ) -> Optional[torch.Tensor]:
        ...


class Executor:
    rules: RuleSet
    inputs: List[ModelComponent]
    loaders: Dict[ModelReference, LazyTensorLoader]
    targets: List[ModelComponent]
    operations: Dict[str, OperationProtocol]

    def __init__(
        self,
        models: List[ModelReference],
        targets: List[ModelComponent],
        rules: RuleSet,
        operations: Optional[Dict[str, OperationProtocol]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.rules = rules
        self.targets = targets
        self.loaders = {
            ref: LazyTensorLoader(ref.tensor_index(cache_dir=cache_dir))
            for ref in models
        }
        self.inputs = []
        for model, loader in self.loaders.items():
            for shard in loader.index.shards:
                filepath = os.path.join(loader.index.base_path, shard.filename)
                self.inputs.append(InputShard(path=filepath))

            rules.procedural.append(LoadTensorRule(model, loader.index.tensor_paths))

        if operations is None:
            operations = {}
        self.operations = operations

    def generate_tensors(self) -> Iterator[Tuple[TensorReference, torch.Tensor]]:
        schedule = self._schedule_ops()
        last_use = {}
        for idx, (component, op) in enumerate(schedule):
            for j in range(len(schedule) - 1, idx, -1):
                if component in schedule[j][1].inputs:
                    break
            last_use[component] = j

        tensors: Dict[ModelComponent, torch.Tensor] = {}
        for idx, (component, op) in enumerate(schedule):
            tensor_args = {}
            for ref in op.inputs:
                if isinstance(ref, InputShard):
                    continue

                tensor_args[ref] = tensors[ref]

            res = self._perform_operation(op, tensor_args)
            if res:
                tensors[component] = res
                if component in self.targets:
                    yield (component, res)

            # expire unreferenced tensors
            expired = []
            for key in tensors:
                if idx > last_use[key]:
                    expired.append(key)

            for key in expired:
                del tensors[key]

    def _perform_operation(
        self, operation: Operation, tensor_args: Dict[TensorReference, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if operation.function == "load_tensor":
            return self.loaders[operation.kwargs["model"]].get_tensor(
                operation.kwargs["key"]
            )
        elif operation.function == "pack_shard":
            safetensors.torch.save_file(
                {key.key: value for (key, value) in tensor_args.items()},
                operation.kwargs["path"],
                metadata={"format": "pt"},
            )
        elif operation.function in self.operations:
            return self.operations[operation.function](
                tensors=tensor_args, **operation.kwargs
            )
        else:
            raise RuntimeError(f"Unimplemented function {operation.function}")

    def _schedule_ops(self):
        dependencies, ops = self._build_dependencies()

        edge_tups = []
        for a in dependencies:
            for b in dependencies[a]:
                edge_tups.append((a, b))

        graph = networkx.DiGraph(edge_tups)
        res = list(
            networkx.lexicographical_topological_sort(graph, key=compare_component_key)
        )
        return [(r, ops[r]) for r in res]

    def _build_dependencies(self):
        dependencies: DefaultDict[ModelComponent, Set[ModelComponent]] = defaultdict(
            default_factory=set
        )
        ops: Dict[ModelComponent, Operation] = {}

        def _visit(node: ModelComponent):
            if (node in ops) or (node in self.inputs):
                return

            operation = self.rules.get(node)
            if not operation:
                raise RuntimeError(f"No rule to produce {node}")
            ops[node] = operation

            for dependency in operation.inputs:
                dependencies[node].add(dependency)

            for dependency in operation.inputs:
                _visit(dependency)

        for t in self.targets:
            _visit(t)
        return dependencies, ops


class Executor:
    loaders: Dict[ModelReference, LazyTensorLoader]
