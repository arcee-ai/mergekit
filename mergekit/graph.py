# Copyright (C) 2024 Charles O. Goddard
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
Module for computational graph execution.

Classes:
    Task: Abstract base class representing a computational task.
    Executor: Class for scheduling and executing directed acyclic task graphs.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

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
        """
        return False


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
    """

    math_device: torch.device
    storage_device: torch.device
    targets: List[Task]
    schedule: List[Task]
    dependencies: Dict[Task, Set[Task]]

    def __init__(
        self,
        tasks: List[Task],
        math_device: torch.device = torch.device("cpu"),
        storage_device: torch.device = torch.device("cpu"),
    ):
        """
        Initializes the Executor with a list of tasks and device configurations.

        Args:
            tasks (List[Task]): The list of tasks to be executed.
            math_device (torch.device, optional): The device for tensor computations. Defaults to CPU.
            storage_device (torch.device, optional): The device for storing results. Defaults to CPU.
        """
        self.math_device = math_device
        self.storage_device = storage_device
        self.schedule = self._make_schedule(tasks)
        self.targets = tasks

    def run(self) -> Iterator[Tuple[Task, Any]]:
        """
        Execute the computed schedule and yield the target values.

        Yields:
            Iterator[Tuple[Task, Any]]: An iterator of task-result pairs.
        """
        # determine last usage of each value, so they can be evicted afterwards
        last_use_index = {}
        for idx, task in reversed(list(enumerate(self.schedule))):
            for t in self.dependencies[task]:
                if t not in last_use_index:
                    last_use_index[t] = idx
            if task not in last_use_index:
                last_use_index[task] = idx

        values: Dict[Task, Any] = {}
        for idx, task in tqdm.tqdm(enumerate(self.schedule), total=len(self.schedule)):
            use_math_device = task.uses_accelerator()

            arguments = {}
            for name, dep in task.arguments().items():
                value = values[dep]

                # ensure any input tensors are on math device if task asks for it
                if use_math_device:
                    if (
                        isinstance(value, torch.Tensor)
                        and value.device != self.math_device
                    ):
                        value = value.to(self.math_device)
                    elif isinstance(value, dict):
                        for key in value:
                            if (
                                isinstance(value[key], torch.Tensor)
                                and value[key].device != self.math_device
                            ):
                                value[key] = value[key].to(self.math_device)

                arguments[name] = value
                del value

            res = task.execute(**arguments)
            del arguments

            if isinstance(res, torch.Tensor) and res.device != self.storage_device:
                res = res.to(self.storage_device)

            values[task] = res
            del res

            if task in self.targets:
                yield (task, values[task])

            # evict unreferenced values
            expired = []
            for key in values:
                if idx >= last_use_index[key]:
                    expired.append(key)

            for key in expired:
                del values[key]

    def execute(self) -> None:
        """
        Execute all tasks and discard results.
        """
        for task, value in self.run():
            pass

    DUMMY_TASK_VALUE = "!!DUMMY!!"

    def _make_schedule(self, targets: List[Task]) -> List[Task]:
        self.schedule = []
        self.dependencies = self._build_dependencies(targets)

        edge_tups = []
        for node in self.dependencies:
            for dependency in self.dependencies[node]:
                edge_tups.append((dependency, node))

        for task in targets:
            # add edges from a dummy node to each target to guarantee
            # they will be included in the final schedule
            edge_tups.append((Executor.DUMMY_TASK_VALUE, task))

        def _compare_key(task: Union[Task, str]):
            if task == Executor.DUMMY_TASK_VALUE:
                return ("", 0)
            return (
                task.group_label() or "",
                -task.priority(),
            )

        graph = networkx.DiGraph(edge_tups)
        res = [
            t
            for t in networkx.lexicographical_topological_sort(graph, key=_compare_key)
            if t != Executor.DUMMY_TASK_VALUE
        ]
        return res

    def _build_dependencies(self, targets: List[Task]) -> Dict[Task, Set[Task]]:
        task_dependencies: Dict[Task, Set[Task]] = {}
        to_process = list(targets)
        while to_process:
            child = to_process.pop()
            if child in task_dependencies:
                continue

            task_dependencies[child] = set()
            for _, dep in child.arguments().items():
                task_dependencies[child].add(dep)
                to_process.append(dep)
        return task_dependencies
