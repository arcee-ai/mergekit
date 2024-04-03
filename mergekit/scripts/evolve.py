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
from typing import List, Optional

import click
import cma
import numpy as np
import yaml

from mergekit.evo.actors import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
)
from mergekit.evo.genome import EvolMergeConfiguration, ModelGenome
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-evolve")
@click.argument("genome-config-path", type=str)
@click.option("--max-fevals", type=int, default=100)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--model-storage-path",
    type=str,
    help="Path to store downloaded and merged models",
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option("--in-memory/--no-in-memory", is_flag=True, default=False)
@click.option(
    "--schedule-strategy",
    "-s",
    type=click.Choice(["actor", "buffered"]),
    default="actor",
)
@add_merge_options
def main(
    genome_config_path: str,
    max_fevals: int,
    vllm: bool,
    model_storage_path: Optional[str],
    num_gpus: Optional[int],
    in_memory: bool,
    schedule_strategy: str,
    merge_options: MergeOptions,
):
    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )
    genome = ModelGenome(
        config.genome, trust_remote_code=merge_options.trust_remote_code
    )

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
    for model in config.genome.models:
        cache.get(model)

    if schedule_strategy == "actor":
        strat = ActorPoolEvaluationStrategy(
            config=config,
            genome=genome,
            merge_options=merge_options,
            model_storage_path=model_storage_path,
            vllm=vllm,
            in_memory=in_memory,
            num_gpus=num_gpus,
        )
    elif schedule_strategy == "buffered":
        if in_memory:
            raise ValueError("Buffered strategy does not support in-memory merging")
        strat = BufferedRayEvaluationStrategy(
            config=config,
            genome=genome,
            merge_options=merge_options,
            model_storage_path=model_storage_path,
            vllm=vllm,
            num_gpus=num_gpus,
        )
    else:
        raise ValueError(f"Invalid schedule strategy: {schedule_strategy}")

    x0 = genome.initial_genotype(random=config.random_init).view(-1).numpy()

    xbest = x0
    xbest_cost = np.inf

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_cost
        if es.result.fbest < xbest_cost:
            xbest = es.result.xbest
            xbest_cost = es.result.fbest
            print(f"New best cost: {xbest_cost:.4f}")
            print(genome.genotype_merge_config(xbest).to_yaml())

    def parallel_evaluate(x: List[np.ndarray]) -> List[float]:
        print(f"Received {len(x)} genotypes")
        res = strat.evaluate_genotypes(x)
        return [-x for x in res]  # maximize

    try:
        xbest, es = cma.fmin2(
            None,
            parallel_objective=parallel_evaluate,
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

    best_config = genome.genotype_merge_config(xbest)
    print("Best merge configuration:")
    print(best_config.to_yaml())


if __name__ == "__main__":
    main()
