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

import os
import shutil
import tempfile
from typing import List, Optional

import click
import cma
import lm_eval
import lm_eval.models
import numpy as np
import ray
import torch
import yaml
from pydantic import BaseModel, model_validator

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

        return self

    def initial_genotype(self, num_layers: int, random: bool = False) -> torch.Tensor:
        """Generate an initial genotype for the given number of layers."""
        assert (
            num_layers % self.layer_granularity == 0
        ), "Number of layers must be a multiple of layer_granularity"

        n_layer_groups = num_layers // self.layer_granularity
        n_models = len(self.models)
        n_params = len(METHOD_PARAM_MAPS[self.merge_method])

        if random:
            return torch.rand(n_layer_groups, n_models, n_params)
        else:
            x0_t = torch.zeros(n_layer_groups, n_models, n_params)
            # weight is always first
            x0_t[:, :, 0] = 1 / n_models
            if n_params > 1:
                # sometimes followed by density
                x0_t[:, :, 1:] = 1
            return x0_t

    def to_merge_config(self, genotype: torch.Tensor) -> MergeConfiguration:
        """Convert a genotype tensor to a mergekit configuration."""

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


class EvolMergeConfiguration(BaseModel, frozen=True):
    genome: ModelGenomeDefinition
    task: str
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    shuffle: bool = False
    random_init: bool = False


@ray.remote(num_cpus=1, num_gpus=1.0)
class MergeEvalActor:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
    ):
        self.config = config
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

    async def _merge_model(
        self,
        genotype: torch.Tensor,
    ) -> str:
        cfg = self.config.genome.to_merge_config(genotype)
        res = tempfile.mkdtemp(prefix="merged", dir=self.model_storage_path)
        run_merge(cfg, out_path=res, options=self.merge_options)
        return res

    async def _evaluate_model(
        self,
        merged_path: str,
    ) -> float:
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
        }
        if self.vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
        else:
            model_args["use_cache"] = True

        results = lm_eval.evaluator.simple_evaluate(
            model="vllm" if self.vllm else "hf",
            model_args=model_args,
            tasks=[self.config.task],
            log_samples=False,
            verbosity="WARNING",
        )

        shutil.rmtree(merged_path)

        print(results["results"][self.config.task])
        return -results["results"][self.config.task]["acc,none"]

    async def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        merged_path = await self._merge_model(genotype)
        return await self._evaluate_model(merged_path)


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
@click.argument("genome-config-path", type=str)
@click.option("--max-fevals", type=int, default=100)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--model-storage-path",
    type=str,
    help="Path to store merged models (should be accessible to all nodes)",
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@add_merge_options
def main(
    genome_config_path: str,
    max_fevals: int,
    vllm: bool,
    model_storage_path: Optional[str],
    num_gpus: Optional[int],
    merge_options: MergeOptions,
):
    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )
    print(config)

    genome_def = config.genome
    cfg_0 = genome_def.models[0].config(
        trust_remote_code=merge_options.trust_remote_code
    )
    n_layers = cfg_0.num_hidden_layers

    if model_storage_path:
        merge_options = merge_options.model_copy(
            update={
                "lora_merge_cache": os.path.join(model_storage_path, "lora"),
                "transformers_cache": os.path.join(model_storage_path, "transformers"),
            }
        )

    # fetch all models on the main process
    cache = LoaderCache()
    cache.setup(merge_options)
    for model in genome_def.models:
        cache.get(model)

    x0 = genome_def.initial_genotype(n_layers, random=config.random_init)
    n_layer_groups, n_models, n_params = x0.shape
    x0 = x0.view(-1).numpy()

    xbest = x0
    xbest_cost = np.inf

    actor_pool = ray.util.ActorPool(
        [
            MergeEvalActor.remote(
                config, merge_options, model_storage_path=model_storage_path, vllm=vllm
            )
            for _ in range(num_gpus or torch.cuda.device_count())
        ]
    )

    def parallel_eval(inputs: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of candidate genotypes. Parallelized with Ray."""
        return list(
            actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(
                    torch.tensor(x).view(n_layer_groups, n_models, n_params)
                ),
                inputs,
            )
        )

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_cost
        if es.result.fbest < xbest_cost:
            xbest = es.result.xbest
            xbest_cost = es.result.fbest
            print(f"New best cost: {xbest_cost:.4f}")
            print(
                genome_def.to_merge_config(
                    torch.tensor(xbest).view(n_layer_groups, n_models, -1)
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

    best_config = genome_def.to_merge_config(
        torch.tensor(xbest).view(n_layer_groups, n_models, -1)
    )
    print("Best merge configuration:")
    print(best_config.to_yaml())


if __name__ == "__main__":
    main()
