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

LOG = logging.getLogger(__name__)


class MultiGPUExecutor:
    """
    Execute computational tasks in parallel across multiple GPUs.

    This class analyzes the dependency structure of a task graph and distributes
    the workload across available GPUs while respecting:
    1. Tasks requiring main thread execution
    2. Tasks that need to be duplicated on each GPU
    3. Task dependencies and data locality
    4. Memory management for intermediate results

    It automatically partitions the task graph into leading tasks (main thread, pre-GPU),
    parallel tasks (distributed across GPUs), and trailing tasks (main thread, post-GPU).

    Attributes:
        num_gpus: Number of GPUs to utilize (None = all available)
        storage_device: Device for storing tensors between stages
        targets: Final output tasks to retain results for
    """

    def __init__(
        self,
        targets: List[Task],
        num_gpus: Optional[int] = None,
        storage_device: Optional[torch.device] = None,
    ):
        """
        Initialize the executor with a list of target tasks.

        This performs initial task graph analysis, including:
        - Finding tasks that must run on the main thread before parallel execution
        - Finding tasks that must run on the main thread after parallel execution
        - Partitioning parallel tasks into islands that can run independently
        - Assigning islands to GPUs using a load-balancing approach

        Args:
            targets: List of final target tasks to execute
            num_gpus: Number of GPUs to utilize (None = all available)
            storage_device: Device for storing intermediate results between execution stages
        """
        self.results: Dict[TaskHandle, Any] = {}
        self.storage_device = storage_device

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        LOG.info(f"Using {num_gpus} GPUs for parallel execution")

        self.universe = TaskUniverse(targets)
        self.targets = set([self.universe.get_handle(t) for t in targets])
        self.serial_schedule = build_schedule(list(self.targets), {})
        ordered_handles = self.serial_schedule.tasks

        self.per_gpu_tasks = set(
            [t for t in ordered_handles if t.task().duplicate_per_gpu()]
        )
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
            if (
                t not in trailing_tasks
                and t not in leading_tasks
                and t not in self.per_gpu_tasks
            )
        ]
        LOG.info(
            f"Task breakdown: {len(self.leading_main_handles)} leading, "
            f"{len(self.per_gpu_tasks)} duplicated per-GPU, "
            f"{len(parallel_handles)} parallel, "
            f"{len(self.trailing_main_handles)} trailing"
        )
        if any(t.task().main_thread_only() for t in parallel_handles):
            raise RuntimeError(
                "Main-thread-only tasks must be either leading or trailing"
            )
        if any(t.task().main_thread_only() for t in self.per_gpu_tasks):
            raise RuntimeError("Tasks can not be both per-GPU and main-thread-only")
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
            total=len(self.serial_schedule.tasks), disable=quiet, desc="Executing graph"
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

            results_snapshot = dict(self.results)

            def update_progress():
                while not self.done_event.is_set():
                    try:
                        task_idx, result = self.task_completion_queue.get(timeout=0.1)
                        task_handle = TaskHandle(self.universe, task_idx)
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
                            task_list=list(self.per_gpu_tasks) + island_task_handles,
                            cached_values=results_snapshot,
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

        This method finds tasks that need to run after parallel execution because they
        require the main thread and have dependencies on other tasks.

        A task is considered "trailing" if:
        - It requires main thread execution (task.main_thread_only() is True)
        - All tasks dependent on it are also trailing tasks (recursive condition)
        - OR it has no dependents (terminal task)

        Args:
            tasks: List of task handles to analyze

        Returns:
            Set[TaskHandle]: Set of tasks that should be executed after parallel processing
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
        """
        Identify tasks that must execute BEFORE parallel GPU tasks.

        This method finds tasks that need to run before parallel execution because they
        require the main thread and are dependencies for other tasks.

        A task is considered "leading" if:
        - It requires main thread execution (task.main_thread_only() is True)
        - It has no dependencies, or all its dependencies are also leading tasks

        Args:
            tasks: List of task handles to analyze

        Returns:
            Set[TaskHandle]: Set of tasks that should be executed before parallel processing
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
        Assign task islands to GPUs for parallel execution.

        This method partitions the parallel task graph into independent subgraphs
        (islands) that can be executed independently on different GPUs. It uses
        a load-balancing approach to distribute islands across available GPUs.

        Task islands are identified as weakly connected components in the task
        dependency graph, meaning groups of tasks that are connected through
        dependencies but don't have dependencies outside their group.

        Args:
            tasks: List of parallel tasks to assign to GPUs
            num_gpus: Number of available GPUs

        Returns:
            Dict[torch.device, List[TaskHandle]]: Mapping from GPU devices to assigned tasks
        """
        task_set = set(tasks)

        edge_list = []
        # Add edges only between parallel tasks
        for task_handle in tasks:
            for dep_handle in task_handle.arguments().values():
                if dep_handle in task_set:
                    edge_list.append((dep_handle._index, task_handle._index))

        island_graph = nx.DiGraph()
        island_graph.add_nodes_from([t._index for t in tasks])
        island_graph.add_edges_from(edge_list)
        islands: List[Set[int]] = list(nx.weakly_connected_components(island_graph))
        LOG.info(f"Found {len(islands)} islands in parallel task graph")
        assignments: Dict[torch.device, List[int]] = {}
        for island in islands:
            if not island:
                continue
            # don't need to sort, inner executor will handle
            island_tasks = [TaskHandle(self.universe, idx) for idx in island]
            # assign to GPU with fewest tasks (load balancing)
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

        This method runs as a thread worker for a specific GPU. It creates an execution
        stream on the assigned GPU, runs the tasks, and queues results back to the main
        thread. Only results needed for target tasks or trailing tasks are retained.

        Args:
            task_list: List of tasks to execute on this GPU
            cached_values: Values of previously-executed dependent tasks
            device: GPU device to execute tasks on
            quiet: Whether to suppress progress bar output
        """
        LOG.debug(f"Device {device} starting")
        with torch.device(device):
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
                    # Only keep results needed for target tasks or trailing tasks
                    if not (
                        task_handle in self.targets
                        or task_handle in self.trailing_dependencies
                    ):
                        result = None
                    self.task_completion_queue.put((task_handle._index, result))
        torch.cuda.synchronize(device=device)
        LOG.debug(f"Device {device} done")
