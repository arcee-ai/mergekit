import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Executor, Task
from mergekit.merge_methods import PerInput, TensorGroup, easy_define, registry
from mergekit.merge_methods.task_adapter import (
    ExecuteMergeMethodTask,
    TensorDictWrapper,
)


class ScalarTensor(Task[torch.Tensor]):
    value: float

    def arguments(self):
        return {}

    def execute(self):
        return torch.tensor([self.value])


def test_decorated_merge_task_with_parameters(monkeypatch):
    monkeypatch.setattr(registry, "_METHODS", {})

    @easy_define.merge_method(name="scaled_sum")
    def scaled_sum(
        group: TensorGroup, weight: PerInput[float], scale: float = 1.0
    ) -> torch.Tensor:
        return (
            sum(
                entry.tensor * value
                for entry, value in zip(group.entries, weight.values_for(group.entries))
            )
            * scale
        )

    registry.register(scaled_sum)

    model_a = ModelReference.model_validate("model_a")
    model_b = ModelReference.model_validate("model_b")
    inputs = TensorDictWrapper(
        tensors=ImmutableMap(
            {model_a: ScalarTensor(value=2), model_b: ScalarTensor(value=4)}
        )
    )
    task = ExecuteMergeMethodTask(
        method_name="scaled_sum",
        model_order=(model_a, model_b),
        output_weight=WeightInfo(name="weight"),
        gather_tensors=inputs,
        parameters=ImmutableMap({"scale": 3.0}),
        input_parameters=ImmutableMap(
            {
                model_a: ImmutableMap({"weight": 0.25}),
                model_b: ImmutableMap({"weight": 0.75}),
            }
        ),
        base_model=None,
    )

    assert task.arguments() == {"tensors": inputs}
    assert task.group_label() == inputs.group_label()
    assert task.uses_accelerator()
    assert {"arguments", "execute", "group_label", "uses_accelerator"}.isdisjoint(
        type(task).model_fields
    )
    results = list(Executor([task]).run())
    assert len(results) == 1
    torch.testing.assert_close(results[0][1], torch.tensor([10.5]))
