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

import shutil
import tempfile
from typing import List, Optional, Tuple

import click
import cma
import lm_eval
import lm_eval.models
import numpy as np
import ray
import torch
from pydantic import BaseModel, model_validator
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.io.tasks import LoaderCache
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, add_merge_options

METHOD_PARAM_MAPS = {
    "linear": ["weight"],
    "task_arithmetic": ["weight"],
    "ties": ["weight", "density"],
    "dare_ties": ["weight", "density"],
    "slerp": ["t"],
}


class ModelGenomeDefinition(BaseModel, frozen=True):
    models: List[ModelReference]
    merge_method: str
    base_model: Optional[ModelReference] = None
    tokenizer_source: Optional[str] = None
    layer_granularity: int = 1

    @model_validator(mode="after")
    def validate(self):
        assert self.merge_method in METHOD_PARAM_MAPS

        if self.merge_method in ["ties", "dare_ties", "task_arithmetic"]:
            assert self.base_model is not None, "base_model is required for this method"
        else:
            assert self.base_model is None, "base_model is not used for this method"

    def to_config(self, genotype: torch.Tensor) -> MergeConfiguration:
        (n_layer_groups, n_models, n_params) = genotype.shape
        assert n_models == len(self.models)
        assert n_params == len(METHOD_PARAM_MAPS[self.merge_method])

        slices = []
        for layer_idx in range(
            0, n_layer_groups * self.layer_granularity, self.layer_granularity
        ):
            s = {
                "sources": [
                    {
                        "model": self.models[i],
                        "layer_range": [layer_idx, layer_idx + self.layer_granularity],
                    }
                    for i in range(n_models)
                ]
            }

            if self.merge_method == "slerp":
                # Choose the two models with the highest weight and
                # calculate the interpolation parameter t
                chosen = torch.topk(
                    genotype[layer_idx // self.layer_granularity, :, 0], 2
                )
                t = torch.softmax(chosen.values, dim=-1)[1].item()
                s["parameters"] = {"t": t}
                s["base_model"] = self.models[chosen.indices[0].item()]
                s["sources"] = [
                    s["sources"][chosen.indices[0].item()],
                    s["sources"][chosen.indices[1].item()],
                ]
                if self.tokenizer_source:
                    s["sources"][0]["parameters"] = {"weight": 1 - t}
                    s["sources"][1]["parameters"] = {"weight": t}
            else:
                for model_idx in range(n_models):
                    params = {}
                    for param_idx, param in enumerate(
                        METHOD_PARAM_MAPS[self.merge_method]
                    ):
                        params[param] = genotype[
                            layer_idx // self.layer_granularity, model_idx, param_idx
                        ]
                        if param == "density":
                            # ensure density is in [0, 1]
                            params[param] = torch.abs(params[param]).clamp(0, 1).item()
                    s["sources"][model_idx]["parameters"] = params

            if self.base_model and (self.base_model not in self.models):
                s["sources"].append(
                    {
                        "model": self.base_model,
                        "layer_range": [layer_idx, layer_idx + self.layer_granularity],
                    }
                )

            slices.append(s)

        return MergeConfiguration.model_validate(
            {
                "merge_method": self.merge_method,
                "slices": slices,
                "parameters": {
                    "normalize": False,
                    "int8_mask": True,
                },
                "dtype": "bfloat16",
                "base_model": self.base_model,
                "tokenizer_source": self.tokenizer_source,
            }
        )


@ray.remote
def merge_model(
    genome: ModelGenomeDefinition, genotype: torch.Tensor, merge_options: MergeOptions
) -> str:
    cfg = genome.to_config(genotype)
    res = tempfile.mkdtemp(prefix="merged")
    run_merge(cfg, out_path=res, options=merge_options)
    return res


@ray.remote(num_gpus=0.9, num_cpus=0.25)
def evaluate_model(
    merged_path: str,
    task: str,
    shuffle_eval: bool = False,
    vllm: bool = False,
    **kwargs,
) -> float:
    if shuffle_eval:
        monkeypatch_lmeval_shuffle()

    model_args = {
        "pretrained": merged_path,
        "dtype": "bfloat16",
    }
    if vllm:
        model_args["gpu_memory_utilization"] = 0.8
        model_args["tensor_parallel_size"] = 1
        model_args["batch_size"] = "auto"
    else:
        model_args["use_cache"] = True

    results = lm_eval.evaluator.simple_evaluate(
        model="vllm" if vllm else "hf",
        model_args=model_args,
        tasks=[task],
        log_samples=False,
        verbosity="WARNING",
        **kwargs,
    )

    shutil.rmtree(merged_path)

    print(results["results"][task])
    return -results["results"][task]["acc,none"]


@ray.remote
def evaluate_genotype(
    genome: ModelGenomeDefinition,
    genotype: torch.Tensor,
    task: str,
    merge_options: MergeOptions,
    shuffle_eval: bool = False,
    vllm: bool = False,
    **kwargs,
) -> Tuple[float, ray.ObjectRef]:
    merge_num_gpus = 0.1 if merge_options.cuda else 0
    # placement group to ensure merge and eval are on the same physical node
    pg = placement_group(
        [{"CPU": 1.0, "GPU": 0.9}],
        strategy="STRICT_PACK",
    )

    ray.get(pg.ready())

    strat = PlacementGroupSchedulingStrategy(pg)
    merged_path = merge_model.options(
        scheduling_strategy=strat, num_gpus=merge_num_gpus
    ).remote(genome, genotype, merge_options)
    eval_task = evaluate_model.options(scheduling_strategy=strat).remote(
        merged_path, task, shuffle_eval=shuffle_eval, vllm=vllm, **kwargs
    )

    result = ray.get(eval_task)
    ray.util.remove_placement_group(pg)
    return result


def monkeypatch_lmeval_shuffle():
    """Monkeypatch lm_eval to shuffle the dataset after downloading."""
    import lm_eval.api.task

    if hasattr(lm_eval.api.task.Task, "_monkey_patched"):
        return

    _old_task_dl = lm_eval.api.task.Task.download

    def _dl_shuffled(self: lm_eval.api.task.Task, *args, **kwargs):
        _old_task_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.Task.download = _dl_shuffled

    _old_ct_dl = lm_eval.api.task.ConfigurableTask.download

    def _ct_dl_shuffled(self, *args, **kwargs):
        _old_ct_dl(self, *args, **kwargs)
        self.dataset = self.dataset.shuffle()

    lm_eval.api.task.ConfigurableTask.download = _ct_dl_shuffled

    lm_eval.api.task.Task._monkey_patched = True
    print("monkey has been patched")


@click.command("mergekit-evolve")
@click.option("--task", type=str, required=True)
@click.option(
    "--model", "-m", "models", type=ModelReference.model_validate, multiple=True
)
@click.option("--base-model", "-b", type=ModelReference.model_validate, default=None)
@click.option("--merge-method", "-M", type=str, required=True)
@click.option(
    "--num-fewshot", type=int, default=None, help="Number of few-shot examples"
)
@click.option("--tokenizer-source", type=str, default=None)
@click.option("--layer-granularity", type=int, default=8)
@click.option(
    "--limit", type=int, default=None, help="Maximum number of examples per eval"
)
@click.option("--shuffle", is_flag=True, default=False, help="Shuffle eval datasets")
@click.option(
    "--random-init",
    is_flag=True,
    default=False,
    help="Randomly initialize a seed genotype",
)
@click.option("--max-fevals", type=int, default=100)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@add_merge_options
def main(
    task: str,
    models: List[ModelReference],
    merge_method: str,
    base_model: Optional[ModelReference],
    num_fewshot: Optional[int],
    tokenizer_source: Optional[str],
    layer_granularity: int,
    limit: Optional[int],
    shuffle: bool,
    random_init: bool,
    max_fevals: int,
    vllm: bool,
    merge_options: MergeOptions,
):
    genome_def = ModelGenomeDefinition(
        models=models,
        merge_method=merge_method,
        base_model=base_model,
        tokenizer_source=tokenizer_source,
        layer_granularity=layer_granularity,
    )

    cfg_0 = models[0].config(trust_remote_code=merge_options.trust_remote_code)
    n_layers = cfg_0.num_hidden_layers
    if n_layers % layer_granularity != 0:
        raise ValueError(
            f"Number of layers ({n_layers}) must be a multiple of layer_granularity ({layer_granularity})"
        )
    n_layer_groups = n_layers // layer_granularity
    genotype_dim = n_layer_groups * len(models) * len(METHOD_PARAM_MAPS[merge_method])

    # fetch all models on the main process
    cache = LoaderCache()
    cache.setup(merge_options)
    for model in models:
        cache.get(model)

    if random_init:
        x0 = np.random.uniform(low=0, high=1, size=genotype_dim)
    else:
        x0_t = torch.zeros(
            n_layer_groups, len(models), len(METHOD_PARAM_MAPS[merge_method])
        )
        x0_t[:, :, 0] = 1 / len(models)
        if len(METHOD_PARAM_MAPS[merge_method]) > 1:
            x0_t[:, :, 1:] = 1
        x0 = x0_t.view(-1).numpy()

    xbest = x0
    xbest_cost = np.inf

    def parallel_eval(inputs: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of candidate genotypes. Parallelized with Ray."""
        return ray.get(
            [
                evaluate_genotype.remote(
                    genome_def,
                    torch.tensor(x).view(n_layer_groups, len(models), -1),
                    task,
                    merge_options,
                    shuffle,
                    limit=limit,
                    num_fewshot=num_fewshot,
                    vllm=vllm,
                )
                for x in inputs
            ]
        )

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_cost
        if es.result.fbest < xbest_cost:
            xbest = es.result.xbest
            xbest_cost = es.result.fbest
            print(f"New best cost: {xbest_cost:.4f}")
            print(
                genome_def.to_config(
                    torch.tensor(xbest).view(n_layer_groups, len(models), -1)
                ).to_yaml()
            )

    try:
        xbest, es = cma.fmin2(
            None,
            parallel_objective=parallel_eval,
            x0=x0,
            sigma0=0.5,
            options={"maxfevals": max_fevals},
            callback=progress_callback,
        )
        xbest_cost = es.result.fbest
    except KeyboardInterrupt:
        pass

    print("!!! OPTIMIZATION COMPLETE !!!")
    print(f"Best cost: {xbest_cost:.4f}")
    print()

    best_config = genome_def.to_config(
        torch.tensor(xbest).view(n_layer_groups, len(models), -1)
    )
    print("Best merge configuration:")
    print(best_config.to_yaml())


if __name__ == "__main__":
    main()
