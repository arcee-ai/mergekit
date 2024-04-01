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

import click
import cma
import lm_eval
import numpy as np
import ray
import torch
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
            }
        )


@ray.remote(num_gpus=0.25)
def evaluate_candidate(
    genome_def: ModelGenomeDefinition,
    genotype: torch.Tensor,
    options: MergeOptions,
    task: str,
    **kwargs,
) -> float:
    config = genome_def.to_config(genotype)

    with tempfile.TemporaryDirectory() as tmpdir:
        run_merge(config, out_path=tmpdir, options=options)

        results = lm_eval.evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={tmpdir},dtype=bfloat16",
            tasks=[task],
            device="cuda",
            log_samples=False,
            verbosity="WARNING",
            **kwargs,
        )

    return results["results"][task]["acc,none"]


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
@click.option("--max-fevals", type=int, default=100)
@add_merge_options
def main(
    task: str,
    models: List[ModelReference],
    merge_method: str,
    base_model: Optional[ModelReference],
    num_fewshot: Optional[int],
    max_fevals: int,
    merge_options: MergeOptions,
):
    genome_def = ModelGenomeDefinition(
        models=models, merge_method=merge_method, base_model=base_model
    )

    cfg_0 = models[0].config(trust_remote_code=merge_options.trust_remote_code)
    n_layers = cfg_0.num_hidden_layers
    genotype_dim = n_layers * len(models) * len(METHOD_PARAM_MAPS[merge_method])

    # fetch all models on the main process
    cache = LoaderCache()
    cache.setup(merge_options)
    for model in models:
        cache.get(model)

    def parallel_eval(inputs: List[np.ndarray]) -> List[float]:
        """Evaluate a batch of candidate genotypes. Parallelized with Ray."""
        return ray.get(
            [
                evaluate_candidate.remote(
                    genome_def,
                    torch.tensor(genome).view(n_layers, len(models), -1),
                    merge_options,
                    task,
                    num_fewshot=num_fewshot,
                )
                for genome in inputs
            ]
        )

    def single_eval(genome: np.ndarray) -> float:
        """Evaluate a single candidate genotype."""
        return parallel_eval([genome])[0]

    x0 = np.random.uniform(low=0, high=1, size=genotype_dim)
    xbest = x0
    xbest_score = -np.inf

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_score
        if es.result.fbest > xbest_score:
            xbest = es.result.xbest
            xbest_score = es.result.fbest
            print(f"New best score: {xbest_score:.4f}")
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
        xbest_score = es.result.fbest
    except KeyboardInterrupt:
        pass

    print("!!! OPTIMIZATION COMPLETE !!!")
    print(f"Best score: {xbest_score:.4f}")
    print()

    best_config = genome_def.to_config(
        torch.tensor(xbest).view(n_layers, len(models), -1)
    )
    print("Best merge configuration:")
    print(best_config.to_yaml())


if __name__ == "__main__":
    main()
