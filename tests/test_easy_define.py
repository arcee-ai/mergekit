from typing import List

import torch

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Executor, Task
from mergekit.merge_methods import easy_define
from mergekit.merge_methods.base import TensorDictWrapper


class ScalarTensor(Task[torch.Tensor]):
    value: float

    def arguments(self):
        return {}

    def execute(self):
        return torch.tensor([self.value])


def test_decorated_merge_task_with_parameters(monkeypatch):
    registry = {}
    monkeypatch.setattr(easy_define, "REGISTERED_MERGE_METHODS", registry)

    @easy_define.merge_method(name="scaled_sum")
    def scaled_sum(
        tensors: List[torch.Tensor], weight: List[float], scale: float = 1.0
    ):
        return sum(t * w for t, w in zip(tensors, weight)) * scale

    model_a = ModelReference.model_validate("model_a")
    model_b = ModelReference.model_validate("model_b")
    inputs = TensorDictWrapper(
        tensors=ImmutableMap(
            {model_a: ScalarTensor(value=2), model_b: ScalarTensor(value=4)}
        )
    )
    task = registry["scaled_sum"].make_task(
        output_weight=WeightInfo(name="weight"),
        tensors=inputs,
        parameters=ImmutableMap({"scale": 3.0}),
        tensor_parameters=ImmutableMap(
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
