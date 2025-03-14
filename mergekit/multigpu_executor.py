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

from .graph import Executor, Task

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
        self.results: Dict[Task, Any] = {}
        self.targets = set(tasks)
        self.storage_device = storage_device

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        # Create temp executor to get full schedule
        temp_exec = Executor(tasks)
        ordered_tasks = temp_exec._make_schedule(tasks)
        self.dependencies = temp_exec.dependencies
        self.total_tasks = len(ordered_tasks)

        leading_tasks = self._find_leading_tasks(ordered_tasks)
        trailing_tasks = self._find_trailing_tasks(ordered_tasks)
        self.trailing_main_tasks = [t for t in ordered_tasks if t in trailing_tasks]
        self.leading_main_tasks = [t for t in ordered_tasks if t in leading_tasks]

        self.trailing_dependencies = set()
        for task in self.trailing_main_tasks:
            self.trailing_dependencies.update(self.dependencies[task])

        parallel_tasks = [
            t
            for t in ordered_tasks
            if (t not in trailing_tasks and t not in leading_tasks)
        ]
        logger.info(
            f"Task breakdown: {len(self.leading_main_tasks)} leading, "
            f"{len(parallel_tasks)} parallel, "
            f"{len(self.trailing_main_tasks)} trailing"
        )
        if any(t.main_thread_only() for t in parallel_tasks):
            raise RuntimeError(
                "Main-thread-only tasks must be either leading or trailing"
            )
        self.gpu_assignments = self._assign_islands_to_gpus(parallel_tasks, num_gpus)

        self.task_completion_queue = queue.Queue()
        self.done_event = threading.Event()

    def run(self, quiet: bool = False) -> Iterator[Tuple[Task, Any]]:
        """
        Execute all tasks and yield target results.

        Yields:
            Iterator[Tuple[Task, Any]]: Task and result pairs
        """
        with tqdm.tqdm(
            total=self.total_tasks, disable=quiet, desc="Executing graph"
        ) as pbar:
            if self.leading_main_tasks:
                exec = Executor(
                    self.leading_main_tasks,
                    math_device=self.storage_device or torch.device("cpu"),
                    storage_device=self.storage_device or torch.device("cpu"),
                )
                for task, result in exec.run(quiet=True):
                    pbar.update()
                    self.results[task] = result

                logger.debug("Leading tasks complete, beginning parallel execution")

            def update_progress():
                while not self.done_event.is_set():
                    try:
                        task, result = self.task_completion_queue.get(timeout=0.1)
                        self.results[task] = result
                        pbar.update()
                    except queue.Empty:
                        continue

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            # Run parallel tasks
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for device, island_tasks in self.gpu_assignments.items():
                    futures.append(
                        executor.submit(
                            self._device_worker,
                            task_list=island_tasks,
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

            logger.debug("Parallel tasks complete")

            # Run main thread tasks
            if self.trailing_main_tasks:
                exec = Executor(
                    self.trailing_main_tasks,
                    math_device=self.storage_device or torch.device("cpu"),
                    storage_device=self.storage_device or torch.device("cpu"),
                    cached_values=dict(self.results),
                )
                for task, result in exec.run(quiet=True):
                    pbar.update()
                    if task in self.targets:
                        self.results[task] = result

        # Yield final results
        for task, result in self.results.items():
            if task in self.targets:
                yield task, result

    def execute(self) -> None:
        """Execute all tasks and discard results"""
        for _ in self.run(quiet=False):
            pass

    def _find_trailing_tasks(self, tasks: List[Task]) -> Set[Task]:
        """
        Identify tasks that must execute AFTER parallel GPU tasks complete.

        Trailing tasks must:
        - Require main thread execution
        - Not have non-trailing dependants
        """
        dependants = defaultdict(set)
        for task, deps in self.dependencies.items():
            for dep in deps:
                dependants[dep].add(task)

        trailing_tasks = set()
        to_explore = set([t for t in tasks if not dependants[t]])
        while to_explore:
            task = to_explore.pop()
            if not task.main_thread_only():
                continue
            if all(d in trailing_tasks for d in dependants[task]):
                trailing_tasks.add(task)
                to_explore.update(self.dependencies[task])
        return trailing_tasks

    def _find_leading_tasks(self, tasks: List[Task]) -> Set[Task]:
        """Identify tasks that must execute BEFORE parallel GPU tasks.

        Leading tasks must:
        - Require main thread execution
        - Not have non-leading dependencies
        """
        leading_tasks = set()
        for task in tasks:
            if not task.main_thread_only():
                continue
            if self.dependencies[task] and any(
                dep not in leading_tasks for dep in self.dependencies[task]
            ):
                continue
            leading_tasks.add(task)
        return leading_tasks

    def _assign_islands_to_gpus(
        self, tasks: List[Task], num_gpus: int
    ) -> Dict[torch.device, List[Task]]:
        """
        Assign task islands to GPUs.

        Task islands (weakly connected components) are groups of tasks that
        can execute independently. This method identifies islands in the
        non-trailing, non-leading task graph and assigns them to devices.
        """

        island_graph = nx.DiGraph()
        island_graph.add_nodes_from(tasks)

        # Add edges only between parallel tasks
        for task in tasks:
            for dep in self.dependencies[task]:
                if dep in tasks:
                    island_graph.add_edge(dep, task)

        islands = list(nx.weakly_connected_components(island_graph))
        logger.info(f"Found {len(islands)} islands in parallel task graph")
        assignments: Dict[torch.device, List[Task]] = {}
        for island in islands:
            # Borrow orderings from original task list
            island_tasks = [t for t in tasks if t in island]
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
        task_list: List[Task],
        cached_values: Dict[Task, Any],
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
                tasks=task_list,
                math_device=device,
                storage_device=self.storage_device or device,
                cached_values=cached_values,
            )
            count = 0
            for task, result in exec.run(quiet=quiet):
                count += 1
                if not (task in self.targets or task in self.trailing_dependencies):
                    result = None
                self.task_completion_queue.put((task, result))
        torch.cuda.synchronize(device=device)
