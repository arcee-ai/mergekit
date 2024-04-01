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

import tempfile
from typing import List, Optional

import accelerate
import click
import cma
import lm_eval
import lm_eval.models
import numpy as np
import ray
import torch
import tqdm
import transformers
from pydantic import BaseModel, model_validator

from mergekit.architecture import get_architecture_info
from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config, run_merge
from mergekit.options import MergeOptions, add_merge_options
from mergekit.plan import MergePlanner

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

    @model_validator(mode="after")
    def validate(self):
        assert self.merge_method in METHOD_PARAM_MAPS

        if self.merge_method in ["ties", "dare_ties", "task_arithmetic"]:
            assert self.base_model is not None, "base_model is required for this method"
        else:
            assert self.base_model is None, "base_model is not used for this method"

    def to_config(self, genotype: torch.Tensor) -> MergeConfiguration:
        (n_layers, n_models, n_params) = genotype.shape
        assert n_models == len(self.models)
        assert n_params == len(METHOD_PARAM_MAPS[self.merge_method])

        slices = []
        for layer_idx in range(n_layers):
            s = {
                "sources": [
                    {
                        "model": self.models[i],
                        "layer_range": [layer_idx, layer_idx + 1],
                    }
                    for i in range(n_models)
                ]
            }

            if self.merge_method == "slerp":
                # Choose the two models with the highest weight and
                # calculate the interpolation parameter t
                chosen = torch.topk(genotype[layer_idx, :, 0], 2)
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
                        params[param] = genotype[layer_idx, model_idx, param_idx]
                        if param == "density":
                            # ensure density is in [0, 1]
                            params[param] = torch.abs(params[param]).clamp(0, 1).item()
                    s["sources"][model_idx]["parameters"] = params

            if self.base_model and (self.base_model not in self.models):
                s["sources"].append(
                    {
                        "model": self.base_model,
                        "layer_range": [layer_idx, layer_idx + 1],
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


class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True


@ray.remote(num_gpus=1)
class MergeEvaluator:
    def __init__(
        self,
        genome_def: ModelGenomeDefinition,
        options: MergeOptions,
        exemplar_config: transformers.PretrainedConfig,
    ):
        self.genome_def = genome_def
        self.options = options
        self.cache = LoaderCache()
        self.cache.setup(options)

        self.arch_info = get_architecture_info(exemplar_config)
        with NoInit():
            self.model = (
                transformers.AutoModelForCausalLM.from_config(
                    exemplar_config,
                    trust_remote_code=options.trust_remote_code,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                )
                .bfloat16()
                .cuda()
                .eval()
                .requires_grad_(False)
            )
        print("Model initialized")

    def evaluate(self, genotype: torch.Tensor, task: str, **kwargs) -> float:
        config = self.genome_def.to_config(genotype)
        planner = MergePlanner(
            config,
            self.arch_info,
            self.options,
            _model_out_config(config, self.arch_info, self.options.trust_remote_code),
        )

        tasks = planner.plan_in_memory()
        executor = Executor(tasks, math_device="cuda")
        for tensor_task, value in executor.run(quiet=False):
            assert isinstance(tensor_task, ReturnTensor)
            if self.model.load_state_dict(
                {tensor_task.weight_info.name: value}, strict=False, assign=False
            ).unexpected_keys:
                raise RuntimeError(f"Unexpected keys in tensor {tensor_task}")

            del value

        model = lm_eval.models.huggingface.HFLM(pretrained=self.model)
        results = lm_eval.evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            log_samples=False,
            verbosity="WARNING",
            **kwargs,
        )

        return -results["results"][task]["acc,none"]


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
@click.option("--max-fevals", type=int, default=100)
@add_merge_options
def main(
    task: str,
    models: List[ModelReference],
    merge_method: str,
    base_model: Optional[ModelReference],
    num_fewshot: Optional[int],
    tokenizer_source: Optional[str],
    max_fevals: int,
    merge_options: MergeOptions,
):
    genome_def = ModelGenomeDefinition(
        models=models,
        merge_method=merge_method,
        base_model=base_model,
        tokenizer_source=tokenizer_source,
    )

    cfg_0 = models[0].config(trust_remote_code=merge_options.trust_remote_code)
    n_layers = cfg_0.num_hidden_layers
    genotype_dim = n_layers * len(models) * len(METHOD_PARAM_MAPS[merge_method])

    # fetch all models on the main process
    cache = LoaderCache()
    cache.setup(merge_options)
    for model in models:
        cache.get(model)

    x0 = np.random.uniform(low=0, high=1, size=genotype_dim)

    xbest = x0
    xbest_cost = np.inf

    exemplar_config = _model_out_config(
        genome_def.to_config(torch.tensor(x0).view(n_layers, len(models), -1)),
        get_architecture_info(cfg_0),
        merge_options.trust_remote_code,
    )

    actors = [
        MergeEvaluator.remote(genome_def, merge_options, exemplar_config)
        for _ in range(torch.cuda.device_count())
    ]
    pool = ray.util.ActorPool(actors)

    def parallel_eval(inputs: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of candidate genotypes. Parallelized with Ray."""
        return list(
            pool.map(
                lambda a, x: a.evaluate.remote(
                    torch.tensor(x).view(n_layers, len(models), -1),
                    task,
                    num_fewshot=num_fewshot,
                ),
                inputs,
            )
        )

    def single_eval(genome: np.ndarray) -> float:
        """Evaluate a single candidate genotype."""
        return parallel_eval([genome])[0]

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_cost
        if es.result.fbest < xbest_cost:
            xbest = es.result.xbest
            xbest_cost = es.result.fbest
            print(f"New best cost: {xbest_cost:.4f}")
            print(
                genome_def.to_config(
                    torch.tensor(xbest).view(n_layers, len(models), -1)
                ).to_yaml()
            )

    try:
        xbest, es = cma.fmin2(
            single_eval,
            x0=x0,
            sigma0=0.5,
            parallel_objective=parallel_eval,
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
        torch.tensor(xbest).view(n_layers, len(models), -1)
    )
    print("Best merge configuration:")
    print(best_config.to_yaml())


if __name__ == "__main__":
    main()
