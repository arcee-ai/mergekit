# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1
"""
Module for computational graph execution.

Classes:
    Task: Abstract base class representing a computational task.
    Executor: Class for scheduling and executing directed acyclic task graphs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import networkx
import torch
import tqdm
from pydantic import BaseModel
from typing_extensions import Generic, TypeVar

ValueT = TypeVar("ValueT")


class Task(ABC, BaseModel, Generic[ValueT], frozen=True):
    """
    Abstract base class representing a task in a computational graph.

    This class should be extended to define specific tasks. Each task can have arguments (dependencies) and a defined execution strategy.

    Attributes:
        Generic[ValueT] (TypeVar): The type of the value that the task returns upon execution.

    Methods:
        arguments: Abstract method to define task arguments (dependencies).
        execute: Abstract method to execute the task.
        priority: Returns the priority of the task for scheduling purposes.
        group_label: Returns an optional label for task grouping.
    """

    @abstractmethod
    def arguments(self) -> Dict[str, "Task"]:
        """
        Returns a dictionary of arguments required for this task. The keys of the dictionary
        are argument names, and the values are Task instances. These keys correspond to the
        keyword argument names expected by the execute method.

        For example, if this method returns {'input1': taskA, 'input2': taskB}, the execute
        method should expect to be called as execute(input1=valueA, input2=valueB), where
        valueA and valueB are the outputs of taskA and taskB respectively.

        Returns:
            Dict[str, "Task"]: A dictionary mapping argument names to Task instances.
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ValueT:
        """
        Executes the task using the results of its dependencies.

        The keyword arguments (**kwargs) for this method are dynamically determined based on
        the dictionary returned by the 'arguments' method. Each key in the 'arguments' method's
        return dictionary becomes a keyword argument in this method, with its value being
        the result of the corresponding task's execution.

        Returns:
            ValueT: The result of the task execution.
        """
        ...

    def priority(self) -> int:
        """
        Returns the priority of the task for scheduling.

        Higher numbers indicate higher priority. Default is 0.

        Returns:
            int: The priority of the task.
        """
        return 0

    def group_label(self) -> Optional[str]:
        """
        Returns an optional label used for grouping tasks together.

        Returns:
            Optional[str]: The group label of the task, if any.
        """
        return None

    def uses_accelerator(self) -> bool:
        """
        Returns True if the task can take advantage of matrix operation
        acceleration (such as on a GPU).

        Tasks that perform heavy matrix operations should return True here
        so they can be scheduled on appropriate devices.

        Returns:
            bool: True if the task benefits from acceleration, False otherwise
        """
        return False

    def main_thread_only(self) -> bool:
        """
        Returns True if the task should only be executed on the main thread.

        Returns:
            bool: True if the task must run on the main thread, False otherwise
        """
        return False

    def duplicate_per_gpu(self) -> bool:
        """
        Returns True if the task should be duplicated for each GPU.

        Tasks that are faster to execute than to transfer between devices
        or are common dependencies of otherwise independent tasks should
        return True here to maximize parallelism.

        Returns:
            bool: True if the task should be duplicated per GPU, False otherwise
        """
        return False


class TaskUniverse:
    """
    Container for tasks and their relationships.

    Maintains a registry of tasks and their dependencies, allowing efficient
    lookup and traversal of the task graph.

    Attributes:
        tasks: List of all tasks in this universe
        task_to_index: Mapping from task instances to their indices
        task_arguments: Mapping from task indices to their argument dependencies
        _type_id_to_index: Quick lookup for seen task instances
    """

    tasks: List[Task]
    task_to_index: Dict[Task, int]
    task_arguments: Dict[int, Dict[str, int]]
    _type_id_to_index: Dict[Tuple[type, int], int]

    def __init__(self, tasks: Optional[Iterable[Task]] = None):
        self.tasks = []
        self.task_to_index = {}
        self.task_arguments = {}
        self._type_id_to_index = {}
        if tasks is not None:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Task, recursive: bool = True) -> "TaskHandle":
        """
        Add a task to the universe and return a handle to it.

        If the task already exists in the universe, returns a handle to the existing instance.

        Args:
            task: The task to add
            recursive: If True, also add all dependent tasks recursively

        Returns:
            TaskHandle: A handle to the added task
        """
        _ti_key = (type(task), id(task))
        if _ti_key in self._type_id_to_index:
            index = self._type_id_to_index[_ti_key]
            assert (
                self.tasks[index] == task
            ), "Task modified after being added to universe"
            return TaskHandle(self, index)

        index = self.task_to_index.setdefault(task, len(self.tasks))
        if index < len(self.tasks):
            return TaskHandle(self, index)
        self.tasks.append(task)
        self._type_id_to_index[_ti_key] = index

        if recursive:
            self.task_arguments[index] = {}
            for k, v in task.arguments().items():
                self.task_arguments[index][k] = self.add_task(v, recursive=True)._index
        return TaskHandle(self, index)

    def get_handle(self, task: Task) -> Optional["TaskHandle"]:
        """
        Get a TaskHandle for an existing task, if it exists in this universe.

        Args:
            task: The task to look up

        Returns:
            Optional[TaskHandle]: A handle to the task, or None if not found
        """
        if task not in self.task_to_index:
            return None
        return TaskHandle(self, self.task_to_index[task])


class TaskHandle:
    """
    A reference to a task within a specific TaskUniverse.

    TaskHandle provides a lightweight way to refer to tasks without directly
    holding the task instances themselves. Particularly useful for putting
    tasks in sets or as keys in dictionaries. Much faster to compare and hash
    than full Task instances.

    Attributes:
        _universe: The TaskUniverse containing the referenced task
        _index: The index of the task within the universe
    """

    __slots__ = ["_universe", "_index"]
    _universe: TaskUniverse
    _index: int

    def __init__(self, universe: TaskUniverse, index: int):
        """
        Initialize a TaskHandle.

        Args:
            universe: The TaskUniverse containing the task
            index: The index of the task within the universe
        """
        self._universe = universe
        self._index = index

    def task(self) -> Task:
        """
        Get the actual Task instance referenced by this handle.

        Returns:
            Task: The referenced task
        """
        return self._universe.tasks[self._index]

    def arguments(self) -> Dict[str, "TaskHandle"]:
        """
        Get handles to all argument tasks (dependencies) of this task.

        Returns:
            Dict[str, TaskHandle]: Mapping from argument names to task handles
        """
        return {
            k: TaskHandle(self._universe, v)
            for k, v in self._universe.task_arguments[self._index].items()
        }

    def __eq__(self, other):
        """
        Check if two TaskHandles refer to the same task in the same universe.

        Args:
            other: Another object to compare with

        Returns:
            bool: True if equal, False otherwise
        """
        if not isinstance(other, TaskHandle):
            return False
        if self._index != other._index:
            return False
        if self._universe is not other._universe:
            return False
        return True

    def __hash__(self):
        return self._index

    def __str__(self):
        return f"TaskHandle({type(self.task()).__name__}, {self._index})"

    __repr__ = __str__


class ExecutionSchedule:
    """
    Represents an ordered schedule of tasks for execution and their lifecycle information.

    Tracks when each task's result can be discarded to optimize memory usage.

    Attributes:
        tasks: Ordered list of tasks to execute
        last_use_index: Maps each task to the index in the schedule where its result is last used
    """

    tasks: List[TaskHandle]
    last_use_index: Dict[TaskHandle, int]

    def __init__(self, tasks: List[TaskHandle], last_use_index: Dict[TaskHandle, int]):
        """
        Initialize an execution schedule.

        Args:
            tasks: Ordered list of tasks to execute
            last_use_index: Dictionary mapping tasks to their last use index in the schedule
        """
        self.tasks = tasks
        self.last_use_index = last_use_index


def build_schedule(
    targets: List[TaskHandle], cached_values: Dict[TaskHandle, Any]
) -> ExecutionSchedule:
    """
    Build an execution schedule for the given target tasks.

    Creates a topologically sorted schedule that respects task dependencies and
    tracks when each task's result can be discarded to optimize memory usage.

    Args:
        targets: List of target tasks that need to be executed
        cached_values: Dictionary of task results that are already available

    Returns:
        ExecutionSchedule: A schedule containing tasks to execute and their lifecycle info
    """
    if not targets:
        return ExecutionSchedule(tasks=[], last_use_index={})

    universe = targets[0]._universe
    assert all(
        t._universe is universe for t in targets
    ), "All tasks must be from the same universe"

    dummy_handle = TaskHandle(universe, -1)
    edge_tups: List[Tuple[TaskHandle, TaskHandle]] = []

    # build a directed graph of dependencies
    explored = set()
    to_explore = set(targets)
    while to_explore:
        task = to_explore.pop()
        if task in explored:
            continue
        explored.add(task)
        if task in (cached_values or {}):
            continue
        for dep in task.arguments().values():
            to_explore.add(dep)
            edge_tups.append((dep, task))

    # add edges from a dummy node to each target to guarantee
    # they will be included in the final schedule
    for target in targets:
        edge_tups.append((dummy_handle, target))

    def _compare_key(node: TaskHandle) -> Tuple[str, int]:
        if node._index < 0:
            return ("", 0)
        task = node.task()
        return (
            task.group_label() or "",
            -task.priority(),
        )

    graph = networkx.DiGraph(edge_tups)
    schedule: List[TaskHandle] = [
        node
        for node in networkx.lexicographical_topological_sort(graph, key=_compare_key)
        if (node != dummy_handle) and node not in (cached_values or {})
    ]

    # Calculate last use indices for memory optimization
    last_use_index = {}
    for idx, task in reversed(list(enumerate(schedule))):
        for dep in task.arguments().values():
            if dep not in last_use_index:
                last_use_index[dep] = idx
        if task not in last_use_index:
            last_use_index[task] = idx
    for task in cached_values or {}:
        if task not in last_use_index:
            last_use_index[task] = len(schedule) + 1

    return ExecutionSchedule(
        tasks=schedule,
        last_use_index=last_use_index,
    )


class Executor:
    """
    Schedules and executes a set of tasks and their dependencies.

    Handles scheduling, execution, the movement of data between devices, and the lifecycle of intermediate results.

    Attributes:
        math_device (torch.device): Device used for tensor computations.
        storage_device (torch.device): Device used for storing intermediate results.
        targets (List[Task]): List of target tasks to be executed.
        schedule (List[Task]): Calculated execution schedule of tasks.
        dependencies (Dict[Task, Set[Task]]): Dependencies of each task.
        cached_values (Optional[Dict[Task, Any]]): Cached values for tasks that have been executed before in a different context.
    """

    math_device: torch.device
    storage_device: torch.device
    universe: TaskUniverse
    targets: List[TaskHandle]
    schedule: ExecutionSchedule
    cached_values: Optional[Dict[TaskHandle, Any]]

    def __init__(
        self,
        targets: Union[List[Task], List[TaskHandle]],
        math_device: torch.device = torch.device("cpu"),
        storage_device: torch.device = torch.device("cpu"),
        cached_values: Optional[Dict[TaskHandle, Any]] = None,
    ):
        """
        Initializes the Executor with a list of tasks and device configurations.

        Args:
            tasks (List[Task]): The list of tasks to be executed.
            math_device (torch.device, optional): The device for tensor computations. Defaults to CPU.
            storage_device (torch.device, optional): The device for storing results. Defaults to CPU.
        """
        self.cached_values = cached_values
        if isinstance(math_device, str):
            math_device = torch.device(math_device)
        if isinstance(storage_device, str):
            storage_device = torch.device(storage_device)
        self.math_device = math_device
        self.storage_device = storage_device
        if targets and isinstance(targets[0], Task):
            universe = TaskUniverse(targets)
            targets = [universe.add_task(t) for t in targets]
        elif targets and isinstance(targets[0], TaskHandle):
            universe = targets[0]._universe
        elif not targets:
            universe = TaskUniverse()
        else:
            raise ValueError("Targets must be a list of Task or TaskHandle instances")
        self.universe = universe
        self.targets = targets
        self.schedule = build_schedule(targets, cached_values=cached_values)

    def _run(
        self,
        quiet: bool = False,
        desc: Optional[str] = None,
    ) -> Iterator[Tuple[TaskHandle, Any]]:
        """
        Execute the computed schedule and yield the target values.

        As opposed to the `run` method, this method yields task handles
        instead of actual `Task` instances.

        Yields:
            Iterator[Tuple[TaskHandle, Any]]: An iterator of taskhandle-result
            pairs.
        """
        last_use_index = self.schedule.last_use_index

        values: Dict[TaskHandle, Any] = {}
        if self.cached_values:
            for task, value in self.cached_values.items():
                values[task] = value
        for idx, task_handle in (
            pbar := tqdm.tqdm(
                list(enumerate(self.schedule.tasks)),
                disable=quiet,
                desc=desc or "Executing graph",
            )
        ):
            task = task_handle.task()
            use_math_device = task.uses_accelerator()

            arguments = {}
            for name, dep_handle in task_handle.arguments().items():
                value = values[dep_handle]

                # ensure any input tensors are on math device if task asks for it
                if use_math_device:
                    value = self._move_tensors(value, self.math_device)

                arguments[name] = value
                del value

            res = task.execute(**arguments)
            del arguments
            res = self._move_tensors(res, self.storage_device)

            values[task_handle] = res
            del res

            if task_handle in self.targets:
                yield (task_handle, values[task_handle])

            # evict unreferenced values
            expired = []
            for key in values:
                if idx >= last_use_index[key]:
                    expired.append(key)

            for key in expired:
                del values[key]

        del values
        del pbar

    def run(
        self,
        quiet: bool = False,
        desc: Optional[str] = None,
    ) -> Iterator[Tuple[Task, Any]]:
        """
        Execute the computed schedule and yield the target values.

        Yields:
            Iterator[Tuple[Task, Any]]: An iterator of task-result
            pairs.
        """
        for handle, value in self._run(quiet=quiet, desc=desc):
            yield (handle.task(), value)

    def execute(self, desc: Optional[str] = None) -> None:
        """
        Execute all tasks and discard results.
        """
        for _ in self.run(desc=desc):
            pass

    def _move_tensors(
        self, value: Any, device: torch.device, non_blocking: Optional[bool] = None
    ) -> Any:
        if non_blocking is None:
            non_blocking = device.type in ["cuda", "xpu"]
        if isinstance(value, torch.Tensor):
            if value.device == device:
                return value
            return value.to(device=device, non_blocking=non_blocking)
        elif isinstance(value, dict):
            return {
                k: self._move_tensors(v, device, non_blocking) for k, v in value.items()
            }
        elif isinstance(value, list):
            return [self._move_tensors(v, device, non_blocking) for v in value]
        elif isinstance(value, tuple):
            return tuple(self._move_tensors(v, device, non_blocking) for v in value)
        return value
