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

import logging
import os
from typing import List, Optional

import click
import cma
import numpy as np
import torch
import tqdm
import transformers
import yaml

from mergekit.common import ModelReference
from mergekit.evo.actors import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
    SerialEvaluationStrategy,
)
from mergekit.evo.config import EvolMergeConfiguration, ModelGenomeDefinition
from mergekit.evo.genome import ModelGenome
from mergekit.options import MergeOptions


@click.command("mergekit-evolve")
@click.argument("genome-config-path", type=str)
@click.option("--max-fevals", type=int, default=100)
@click.option("--vllm/--no-vllm", is_flag=True, default=False, help="Use vLLM")
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(["pool", "buffered", "serial"]),
    default="pool",
    help="Evaluation scheduling strategy",
)
@click.option(
    "--in-memory/--no-in-memory",
    is_flag=True,
    default=False,
    help="Use in-memory merge & evaluation",
)
@click.option(
    "--storage-path",
    type=str,
    help="Path to storage accessible to all nodes for model storage",
    required=True,
)
@click.option("--num-gpus", type=int, help="Number of GPUs to use across all nodes")
@click.option("--merge-cuda/--no-merge-cuda", is_flag=True, default=True)
@click.option("--trust-remote-code/--no-trust-remote-code", is_flag=True, default=False)
@click.option("--allow-crimes/--no-allow-crimes", is_flag=True, default=False)
@click.option("--random-seed", type=int, default=0)
@click.option("--batch-size", type=int, default=None, help="Batch size for evaluation")
@click.option("--sigma0", type=float, default=1 / 6, help="Initial sigma for CMA-ES")
def main(
    genome_config_path: str,
    max_fevals: int,
    vllm: bool,
    strategy: str,
    in_memory: bool,
    storage_path: Optional[str],
    num_gpus: Optional[int],
    merge_cuda: bool,
    trust_remote_code: bool,
    allow_crimes: bool,
    random_seed: int,
    batch_size: Optional[int],
    sigma0: float,
):
    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=merge_cuda,
        low_cpu_memory=merge_cuda,
        out_shard_size=1_000_000_000_000,  # one trillion bytes!
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda,
        copy_tokenizer=True,
        tensorizer=False,
        safe_serialization=True,
    )

    # convert models to single-shard safetensors
    resharded_models = []
    resharded_base = None
    for model in tqdm.tqdm(config.genome.models, desc="Resharding models"):
        resharded_models.append(
            _reshard_model(
                model, storage_path, merge_options.lora_merge_cache, trust_remote_code
            )
        )
    if config.genome.base_model is not None:
        resharded_base = _reshard_model(
            config.genome.base_model,
            storage_path,
            merge_options.lora_merge_cache,
            trust_remote_code,
        )

    genome = ModelGenome(
        ModelGenomeDefinition.model_validate(
            {
                **config.genome.model_dump(
                    exclude=[
                        "models",
                        "base_model",
                    ]
                ),
                "models": resharded_models,
                "base_model": resharded_base,
            }
        ),
        trust_remote_code=trust_remote_code,
    )

    if strategy == "pool":
        strat_cls = ActorPoolEvaluationStrategy
    elif strategy == "buffered":
        strat_cls = BufferedRayEvaluationStrategy
    elif strategy == "serial":
        strat_cls = SerialEvaluationStrategy
    else:
        raise ValueError(f"Unknown strategy {strategy}")

    strat = strat_cls(
        config,
        genome,
        merge_options,
        num_gpus=num_gpus,
        vllm=vllm,
        in_memory=in_memory,
        model_storage_path=os.path.join(storage_path, "merged"),
        batch_size=batch_size,
    )

    x0 = genome.initial_genotype(random=config.random_init).view(-1).numpy()
    xbest = x0
    xbest_cost = np.inf

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_cost
        if es.result.fbest < xbest_cost:
            xbest = es.result.xbest
            xbest_cost = es.result.fbest
            print(f"New best cost: {xbest_cost:.4f}")
            best_yaml = genome.genotype_merge_config(xbest).to_yaml()
            with open(os.path.join(storage_path, "best_config.yaml"), "w") as f:
                f.write(best_yaml)
            print(f"Merge configuration:\n{best_yaml}")

    def parallel_evaluate(x: List[np.ndarray]) -> List[float]:
        print(f"Received {len(x)} genotypes")
        res = strat.evaluate_genotypes(x)
        return [-x for x in res]  # maximize

    try:
        xbest, es = cma.fmin2(
            None,
            parallel_objective=parallel_evaluate,
            x0=x0,
            sigma0=sigma0,
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


def _reshard_model(
    model: ModelReference, storage_path: str, merge_cache: str, trust_remote_code: bool
) -> ModelReference:
    merged = model.merged(
        cache_dir=merge_cache,
        trust_remote_code=trust_remote_code,
    )
    out_path = os.path.join(
        storage_path,
        "input_models",
        merged.model._unique_id(),
    )

    if os.path.exists(out_path):
        logging.info(f"Using existing resharded model at {out_path}")
        return ModelReference(model=out_path)

    model_hf = transformers.AutoModelForCausalLM.from_pretrained(
        merged.model.path,
        revision=merged.model.revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16,
    )
    model_hf.save_pretrained(
        out_path, safe_serialization=True, out_shard_size=1_000_000_000_000
    )
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model.model.path,
            revision=model.model.revision,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        tokenizer.save_pretrained(out_path)
    except Exception as e:
        logging.warning(f"Could not save tokenizer for {model.model}", exc_info=e)

    return ModelReference(model=out_path)
