# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1
"""
Implementation of multi-GPU parallel task execution.

Handles distribution of parallelizable tasks across multiple GPUs while respecting:
- Main-thread-only task requirements
- Task dependency graphs
- GPU assignment of connected task components
- Intermediate result storage locations
"""

import concurrent.futures
import logging
import queue
import threading
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import networkx as nx
import torch
import tqdm

from .graph import (
    Executor,
    Task,
    TaskHandle,
    TaskUniverse,
    build_schedule,
)

logger = logging.getLogger(__name__)


class MultiGPUExecutor:
    """
    Execute tasks across multiple GPUs.

    Attributes:
        num_gpus: Number of GPUs to utilize (None = all available)
        storage_device: Device for storing tensors between stages
        targets: Final output tasks to retain results for
    """

    def __init__(
        self,
        tasks: List[Task],
        num_gpus: Optional[int] = None,
        storage_device: Optional[torch.device] = None,
    ):
        """
        Initialize the executor with a list of tasks.

        Args:
            tasks: List of tasks to execute
            num_gpus: Number of GPUs to utilize (None = all available)
            storage_device: Device for storing tensors between stages
        """
        self.results: Dict[TaskHandle, Any] = {}
        self.storage_device = storage_device

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        logger.info(f"Using {num_gpus} GPUs for parallel execution")

        logger.debug(f"Buidling task universe with {len(tasks)} target tasks")
        self.universe = TaskUniverse(tasks)
        self.targets = set([self.universe.get_handle(t) for t in tasks])
        logger.debug("Building task schedule")
        preliminary_schedule = build_schedule(list(self.targets), {})
        ordered_handles = preliminary_schedule.tasks

        leading_tasks = self._find_leading_tasks(ordered_handles)
        trailing_tasks = self._find_trailing_tasks(ordered_handles)
        self.trailing_main_handles = [t for t in ordered_handles if t in trailing_tasks]
        self.leading_main_handles = [t for t in ordered_handles if t in leading_tasks]

        self.trailing_dependencies: Set[TaskHandle] = set()
        for task_handle in self.trailing_main_handles:
            self.trailing_dependencies.update(task_handle.arguments().values())

        parallel_handles = [
            t
            for t in ordered_handles
            if (t not in trailing_tasks and t not in leading_tasks)
        ]
        logger.info(
            f"Task breakdown: {len(self.leading_main_handles)} leading, "
            f"{len(parallel_handles)} parallel, "
            f"{len(self.trailing_main_handles)} trailing"
        )
        if any(t.task().main_thread_only() for t in parallel_handles):
            raise RuntimeError(
                "Main-thread-only tasks must be either leading or trailing"
            )
        self.gpu_assignments = self._assign_islands_to_gpus(parallel_handles, num_gpus)

        self.task_completion_queue = queue.Queue()
        self.done_event = threading.Event()

    def run(self, quiet: bool = False) -> Iterator[Tuple[Task, Any]]:
        """
        Execute all tasks and yield target results.

        Yields:
            Iterator[Tuple[Task, Any]]: Task and result pairs
        """
        with tqdm.tqdm(
            total=len(self.universe.tasks), disable=quiet, desc="Executing graph"
        ) as pbar:
            if self.leading_main_handles:
                exec = Executor(
                    self.leading_main_handles,
                    math_device=self.storage_device or torch.device("cpu"),
                    storage_device=self.storage_device or torch.device("cpu"),
                )
                for task_handle, result in exec._run(quiet=True):
                    pbar.update()
                    self.results[task_handle] = result

            def update_progress():
                while not self.done_event.is_set():
                    try:
                        task_handle, result = self.task_completion_queue.get(
                            timeout=0.1
                        )
                        self.results[task_handle] = result
                        pbar.update()
                    except queue.Empty:
                        continue

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            # Run parallel tasks
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for device, island_task_handles in self.gpu_assignments.items():
                    futures.append(
                        executor.submit(
                            self._device_worker,
                            task_list=island_task_handles,
                            cached_values=dict(self.results),
                            device=device,
                            quiet=True,
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    if ex := future.exception():
                        self.done_event.set()
                        executor.shutdown(wait=False)
                        raise ex

            self.done_event.set()
            progress_thread.join()

            # Run main thread tasks
            if self.trailing_main_handles:
                exec = Executor(
                    self.trailing_main_handles,
                    math_device=self.storage_device or torch.device("cpu"),
                    storage_device=self.storage_device or torch.device("cpu"),
                    cached_values=dict(self.results),
                )
                for task_handle, result in exec._run(quiet=True):
                    pbar.update()
                    if task_handle in self.targets:
                        self.results[task_handle] = result

        # Yield final results
        for task_handle, result in self.results.items():
            if task_handle in self.targets:
                yield task_handle.task(), result

    def execute(self) -> None:
        """Execute all tasks and discard results"""
        for _ in self.run(quiet=False):
            pass

    def _find_trailing_tasks(self, tasks: List[TaskHandle]) -> Set[TaskHandle]:
        """
        Identify tasks that must execute AFTER parallel GPU tasks complete.

        Trailing tasks must:
        - Require main thread execution
        - Not have non-trailing dependants
        """
        dependants = defaultdict(set)
        for task_idx, arg_indices in self.universe.task_arguments.items():
            for dep_idx in arg_indices.values():
                dependants[TaskHandle(self.universe, dep_idx)].add(
                    TaskHandle(self.universe, task_idx)
                )

        trailing_tasks = set()
        to_explore = set([t for t in tasks if not dependants[t]])
        while to_explore:
            task_handle = to_explore.pop()
            task = task_handle.task()
            if not task.main_thread_only():
                continue
            if all(d in trailing_tasks for d in dependants[task_handle]):
                trailing_tasks.add(task_handle)
                to_explore.update(task_handle.arguments().values())
        return trailing_tasks

    def _find_leading_tasks(self, tasks: List[TaskHandle]) -> Set[TaskHandle]:
        """Identify tasks that must execute BEFORE parallel GPU tasks.

        Leading tasks must:
        - Require main thread execution
        - Not have non-leading dependencies
        """
        leading_tasks = set()
        for task_handle in tasks:
            task = task_handle.task()
            if not task.main_thread_only():
                continue
            args = task_handle.arguments()
            if args and any(dep not in leading_tasks for dep in args.values()):
                continue
            leading_tasks.add(task_handle)
        return leading_tasks

    def _assign_islands_to_gpus(
        self, tasks: List[TaskHandle], num_gpus: int
    ) -> Dict[torch.device, List[TaskHandle]]:
        """
        Assign task islands to GPUs.

        Task islands (weakly connected components) are groups of tasks that
        can execute independently. This method identifies islands in the
        non-trailing, non-leading task graph and assigns them to devices.
        """
        task_set = set(tasks)

        edge_list = []
        # Add edges only between parallel tasks
        for task_handle in tasks:
            for dep_handle in task_handle.arguments().values():
                if dep_handle in task_set:
                    edge_list.append((dep_handle._index, task_handle._index))

        island_graph = nx.DiGraph()
        island_graph.add_edges_from(edge_list)
        islands: List[Set[int]] = list(nx.weakly_connected_components(island_graph))
        logger.info(f"Found {len(islands)} islands in parallel task graph")
        assignments: Dict[torch.device, List[int]] = {}
        for island in islands:
            if not island:
                continue
            # don't need to sort, inner executor will handle
            island_tasks = [self.universe.tasks[i] for i in island]
            # assign to GPU with fewest tasks
            device_idx = min(
                range(num_gpus),
                key=lambda i: len(assignments.get(torch.device(f"cuda:{i}"), [])),
            )
            device = torch.device(f"cuda:{device_idx}")
            assignments[device] = assignments.get(device, []) + island_tasks
        return assignments

    def _device_worker(
        self,
        task_list: List[TaskHandle],
        cached_values: Dict[TaskHandle, Any],
        device: torch.device,
        quiet: bool,
    ):
        """
        Execute a set of tasks on a single GPU.

        Args:
            island_tasks: List of tasks to execute
            cached_values: Values of previously-executed dependent tasks
            device: Device to execute tasks on
            quiet: Suppress progress bar output
        """
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            exec = Executor(
                targets=task_list,
                math_device=device,
                storage_device=self.storage_device or device,
                cached_values=cached_values,
            )
            count = 0
            for task_handle, result in exec._run(quiet=quiet):
                count += 1
                if not (
                    task_handle in self.targets
                    or task_handle in self.trailing_dependencies
                ):
                    result = None
                self.task_completion_queue.put((task_handle, result))
        torch.cuda.synchronize(device=device)
