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
import time
from typing import List, Optional

import click
import cma
import numpy as np
import pandas
import ray
import torch
import tqdm
import transformers
import yaml

try:
    import wandb
except ImportError:
    wandb = None


from mergekit.common import ModelReference
from mergekit.evo.config import (
    EvolMergeConfiguration,
    ModelGenomeDefinition,
    check_for_naughty_config,
)
from mergekit.evo.genome import ModelGenome
from mergekit.evo.strategy import (
    ActorPoolEvaluationStrategy,
    BufferedRayEvaluationStrategy,
    SerialEvaluationStrategy,
)
from mergekit.merge import run_merge
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
@click.option("use_wandb", "--wandb/--no-wandb", is_flag=True, default=False)
@click.option("--wandb-project", type=str, help="Wandb project name")
@click.option("--wandb-entity", type=str, help="Wandb entity name")
@click.option(
    "--task-search-path",
    type=str,
    multiple=True,
    help="Path to search for lmeval tasks",
)
@click.option(
    "--i-understand-the-depths-of-the-evils-i-am-unleashing",
    "allow_benchmark_tasks",
    is_flag=True,
    default=False,
    help="Allow benchmark tasks as objectives",
)
@click.option(
    "--save-final-model/--no-save-final-model",
    is_flag=True,
    default=True,
    help="Save the final merged model",
)
@click.option(
    "--reshard/--no-reshard",
    is_flag=True,
    default=True,
    help="Convert models to single-shard safetensors for faster merge",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Maximum time to run the optimization in seconds",
)
@click.option(
    "--force-population-size",
    type=int,
    default=None,
    help="Force a specific initial population size for CMA-ES",
)
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
    use_wandb: bool,
    wandb_project: Optional[str],
    wandb_entity: Optional[str],
    task_search_path: List[str],
    allow_benchmark_tasks: bool,
    save_final_model: bool,
    reshard: bool,
    timeout: Optional[float],
    force_population_size: Optional[int],
):
    config = EvolMergeConfiguration.model_validate(
        yaml.safe_load(open(genome_config_path, "r", encoding="utf-8"))
    )

    check_for_naughty_config(config, allow=allow_benchmark_tasks)

    if use_wandb:
        if not wandb:
            raise RuntimeError("wandb is not installed")
        run = wandb.init(
            project=wandb_project or "mergekit-evolve",
            entity=wandb_entity,
            config=config.model_dump(mode="json"),
        )
    else:
        run = None

    merge_options = MergeOptions(
        transformers_cache=os.path.join(storage_path, "transformers_cache"),
        lora_merge_cache=os.path.join(storage_path, "lora_merge_cache"),
        cuda=merge_cuda,
        low_cpu_memory=merge_cuda and not in_memory,
        out_shard_size=1_000_000_000_000,  # one trillion bytes!
        trust_remote_code=trust_remote_code,
        allow_crimes=allow_crimes,
        random_seed=random_seed,
        quiet=True,
        read_to_gpu=merge_cuda and not in_memory,
        copy_tokenizer=True,
        safe_serialization=True,
    )

    # convert models to single-shard safetensors
    if reshard:
        resharded_models = []
        resharded_base = None
        for model in tqdm.tqdm(config.genome.models, desc="Resharding models"):
            resharded_models.append(
                _reshard_model(
                    model,
                    storage_path,
                    merge_options.lora_merge_cache,
                    trust_remote_code,
                )
            )
        if config.genome.base_model is not None:
            resharded_base = _reshard_model(
                config.genome.base_model,
                storage_path,
                merge_options.lora_merge_cache,
                trust_remote_code,
            )
    else:
        resharded_models = config.genome.models
        resharded_base = config.genome.base_model

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
        task_search_path=task_search_path,
    )

    x0 = genome.initial_genotype(random=config.random_init).view(-1).numpy()
    xbest = x0
    xbest_cost = np.inf

    def progress_callback(es: cma.CMAEvolutionStrategy):
        nonlocal xbest, xbest_cost

        res = es.result
        if use_wandb:
            best_params = genome.genotype_to_param_arrays(res.xbest)
            mean_params = genome.genotype_to_param_arrays(res.xfavorite)
            run.log(
                {
                    "best_score": -res.fbest,
                    "best_genome": wandb.Table(data=pandas.DataFrame(best_params)),
                    "mean_genome": wandb.Table(data=pandas.DataFrame(mean_params)),
                    "mean_std": genome.genotype_to_param_arrays(res.stds),
                    "evaluations": res.evaluations,
                },
                commit=True,
                step=res.evaluations,
            )

        if res.fbest < xbest_cost:
            xbest = res.xbest
            xbest_cost = res.fbest
            print(f"New best score: {-xbest_cost:.4f}")
            best_yaml = genome.genotype_merge_config(xbest).to_yaml()
            with open(os.path.join(storage_path, "best_config.yaml"), "w") as f:
                f.write(best_yaml)
            print(f"Merge configuration:\n{best_yaml}")

            if use_wandb:
                art = wandb.Artifact("best_config", type="merge_config")
                art.add_file(os.path.join(storage_path, "best_config.yaml"))
                run.log_artifact(art)

    def parallel_evaluate(x: List[np.ndarray]) -> List[float]:
        print(f"Received {len(x)} genotypes")
        res = strat.evaluate_genotypes(x)

        if use_wandb:
            res = list(res)
            score_mean = np.mean([r["score"] for r in res])
            score_std = np.std([r["score"] for r in res])
            run.log(
                {
                    "population/score_mean": score_mean,
                    "population/score_std": score_std,
                },
                commit=False,
            )
            for task in res[0]["results"]:
                for metric in res[0]["results"][task]:
                    values = [r["results"][task][metric] for r in res]
                    values = [v for v in values if v is not None]
                    if not values or all(isinstance(v, str) for v in values):
                        continue

                    mean = np.mean(values)
                    max_val = max(values)
                    min_val = min(values)

                    metric_pretty = metric.replace(",none", "")
                    if metric_pretty.endswith("_stderr"):
                        # don't log stats for stderr that's just silly
                        continue

                    run.log(
                        {
                            f"population/{task}_{metric_pretty}_mean": mean,
                            f"population/{task}_{metric_pretty}_max": max_val,
                            f"population/{task}_{metric_pretty}_min": min_val,
                        },
                        commit=False,
                    )

        return [-x["score"] for x in res]  # maximize

    try:
        cma_opts = {"maxfevals": max_fevals, "timeout": timeout}
        if force_population_size is not None:
            cma_opts["popsize"] = force_population_size
        xbest, es = cma.fmin2(
            None,
            parallel_objective=parallel_evaluate,
            x0=x0,
            sigma0=sigma0,
            options=cma_opts,
            callback=progress_callback,
        )
        xbest_cost = es.result.fbest
    except KeyboardInterrupt:
        ray.shutdown()

    print("!!! OPTIMIZATION COMPLETE !!!")
    print(f"Best cost: {xbest_cost:.4f}")
    print()

    # pause for a bit to let any CUDA-using processes clean up
    time.sleep(1.0)

    # save the best merge configuration using original model references
    genome_pretty = ModelGenome(config.genome, trust_remote_code=trust_remote_code)
    best_config = genome_pretty.genotype_merge_config(xbest)
    print("Best merge configuration:")
    print(best_config.to_yaml())

    if save_final_model:
        print("Saving final model...")
        run_merge(best_config, os.path.join(storage_path, "final_model"), merge_options)


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
        cache_dir=os.path.join(storage_path, "transformers_cache"),
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


if __name__ == "__main__":
    main()
