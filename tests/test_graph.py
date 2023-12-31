from typing import Any, Dict, Optional

import networkx
import pytest

from mergekit.common import ImmutableMap
from mergekit.graph import Executor, Task

EXECUTION_COUNTS: Dict[Task, int] = {}


class DummyTask(Task):
    result: Any
    dependencies: ImmutableMap[str, Task]
    name: str = "DummyTask"
    grouplabel: Optional[str] = None
    execution_count: int = 0

    def arguments(self):
        return self.dependencies

    def group_label(self) -> Optional[str]:
        return self.grouplabel

    def execute(self, **kwargs):
        EXECUTION_COUNTS[self] = EXECUTION_COUNTS.get(self, 0) + 1
        return self.result


def create_mock_task(name, result=None, dependencies=None, group_label=None):
    if dependencies is None:
        dependencies = {}
    return DummyTask(
        result=result,
        dependencies=ImmutableMap(data=dependencies),
        name=name,
        grouplabel=group_label,
    )


# Test cases for the Task implementation
class TestTaskClass:
    def test_task_execute(self):
        # Testing the execute method
        task = create_mock_task("task1", result=42)
        assert task.execute() == 42, "Task execution did not return expected result"

    def test_task_priority(self):
        task = create_mock_task("task1")
        assert task.priority() == 0, "Default priority should be 0"

    def test_task_group_label(self):
        task = create_mock_task("task1")
        assert task.group_label() is None, "Default group label should be None"


# Test cases for the Executor implementation
class TestExecutorClass:
    def test_executor_initialization(self):
        # Testing initialization with single task
        task = create_mock_task("task1")
        executor = Executor([task])
        assert executor.targets == [
            task
        ], "Executor did not initialize with correct targets"

    def test_executor_empty_list(self):
        list(Executor([]).run())

    def test_executor_scheduling(self):
        # Testing scheduling with dependencies
        task1 = create_mock_task("task1", result=1)
        task2 = create_mock_task("task2", result=2, dependencies={"task1": task1})
        executor = Executor([task2])
        assert (
            len(executor._make_schedule([task2])) == 2
        ), "Schedule should include two tasks"

    def test_executor_dependency_building(self):
        # Testing dependency building
        task1 = create_mock_task("task1")
        task2 = create_mock_task("task2", dependencies={"task1": task1})
        executor = Executor([task2])
        dependencies = executor._build_dependencies([task2])
        assert task1 in dependencies[task2], "Task1 should be a dependency of Task2"

    def test_executor_run(self):
        # Testing execution through the run method
        task1 = create_mock_task("task1", result=10)
        task2 = create_mock_task("task2", result=20, dependencies={"task1": task1})
        executor = Executor([task2])
        results = list(executor.run())
        assert (
            len(results) == 1 and results[0][1] == 20
        ), "Executor run did not yield correct results"

    def test_executor_execute(self):
        # Testing execute method for side effects
        task1 = create_mock_task("task1", result=10)
        executor = Executor([task1])
        # No assert needed; we're ensuring no exceptions are raised and method completes
        executor.execute()

    def test_dependency_ordering(self):
        # Testing the order of task execution respects dependencies
        task1 = create_mock_task("task1", result=1)
        task2 = create_mock_task("task2", result=2, dependencies={"task1": task1})
        task3 = create_mock_task("task3", result=3, dependencies={"task2": task2})
        executor = Executor([task3])

        schedule = executor._make_schedule([task3])
        assert schedule.index(task1) < schedule.index(
            task2
        ), "Task1 should be scheduled before Task2"
        assert schedule.index(task2) < schedule.index(
            task3
        ), "Task2 should be scheduled before Task3"


class TestExecutorGroupLabel:
    def test_group_label_scheduling(self):
        # Create tasks with group labels and dependencies
        task1 = create_mock_task("task1", group_label="group1")
        task2 = create_mock_task(
            "task2", dependencies={"task1": task1}, group_label="group1"
        )
        task3 = create_mock_task("task3", group_label="group2")
        task4 = create_mock_task(
            "task4", dependencies={"task2": task2, "task3": task3}, group_label="group1"
        )

        # Initialize Executor with the tasks
        executor = Executor([task4])

        # Get the scheduled tasks
        schedule = executor._make_schedule([task4])

        # Check if tasks with the same group label are scheduled consecutively when possible
        group_labels_in_order = [
            task.group_label() for task in schedule if task.group_label()
        ]
        assert group_labels_in_order == [
            "group1",
            "group1",
            "group2",
            "group1",
        ], "Tasks with same group label are not scheduled consecutively"

    def test_group_label_with_dependencies(self):
        # Creating tasks with dependencies and group labels
        task1 = create_mock_task("task1", result=1, group_label="group1")
        task2 = create_mock_task(
            "task2", result=2, dependencies={"task1": task1}, group_label="group2"
        )
        task3 = create_mock_task(
            "task3", result=3, dependencies={"task2": task2}, group_label="group1"
        )

        executor = Executor([task3])
        schedule = executor._make_schedule([task3])
        scheduled_labels = [
            task.group_label() for task in schedule if task.group_label()
        ]

        # Check if task3 is scheduled after task1 and task2 due to dependency, even though it has the same group label as task1
        group1_indices = [
            i for i, label in enumerate(scheduled_labels) if label == "group1"
        ]
        group2_index = scheduled_labels.index("group2")

        assert (
            group1_indices[-1] > group2_index
        ), "Task with the same group label but later dependency was not scheduled after different group label"


class TestExecutorSingleExecution:
    def test_single_execution_per_task(self):
        EXECUTION_COUNTS.clear()

        shared_task = create_mock_task("shared_task", result=100)
        task1 = create_mock_task("task1", dependencies={"shared": shared_task})
        task2 = create_mock_task("task2", dependencies={"shared": shared_task})
        task3 = create_mock_task("task3", dependencies={"task1": task1, "task2": task2})

        Executor([task3]).execute()

        assert shared_task in EXECUTION_COUNTS, "Dependency not executed"
        assert (
            EXECUTION_COUNTS[shared_task] == 1
        ), "Shared dependency should be executed exactly once"


class CircularTask(Task):
    def arguments(self) -> Dict[str, Task]:
        return {"its_a_me": self}

    def execute(self, **_kwargs) -> Any:
        assert False, "Task with circular dependency executed"


class TestExecutorCircularDependency:
    def test_circular_dependency(self):
        with pytest.raises(networkx.NetworkXUnfeasible):
            Executor([CircularTask()]).execute()


if __name__ == "__main__":
    pytest.main()
