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

import asyncio
import gc
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import lm_eval
import lm_eval.api.model
import lm_eval.models.huggingface
import numpy as np
import ray
import ray.util.queue
import ray.util.scheduling_strategies
import torch
import transformers

from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info
from mergekit.config import MergeConfiguration
from mergekit.evo.config import EvolMergeConfiguration, TaskConfiguration
from mergekit.evo.genome import ModelGenome
from mergekit.evo.monkeypatch import NoInit, monkeypatch_lmeval_shuffle
from mergekit.graph import Executor
from mergekit.io.tasks import LoaderCache, ReturnTensor
from mergekit.merge import _model_out_config, run_merge
from mergekit.options import MergeOptions
from mergekit.plan import MergePlanner


class EvaluationStrategyBase(ABC):
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.batch_size = batch_size

    @abstractmethod
    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        pass

    @abstractmethod
    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        pass


def _eval_model(
    model: Union[str, lm_eval.api.model.LM],
    tasks: List[TaskConfiguration],
    model_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> float:
    results = lm_eval.evaluator.simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=list(set([task.name for task in tasks])),
        log_samples=False,
        verbosity="WARNING",
        **kwargs,
    )

    logging.info(results["results"])
    res = 0
    for task in tasks:
        res += results["results"][task.name][task.metric] * task.weight
    return res


def evaluate_model(
    merged_path: str,
    tasks: List[TaskConfiguration],
    num_fewshot: Optional[int],
    limit: Optional[int],
    vllm: bool,
    batch_size: Optional[int] = None,
) -> float:
    # monkeypatch_tqdm()
    try:
        model_args = {
            "pretrained": merged_path,
            "dtype": "bfloat16",
        }
        if vllm:
            model_args["gpu_memory_utilization"] = 0.8
            model_args["tensor_parallel_size"] = 1
            model_args["batch_size"] = "auto"
            model_args["max_model_len"] = 4096
        else:
            model_args["use_cache"] = True

        res = _eval_model(
            "vllm" if vllm else "huggingface",
            tasks,
            model_args,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=batch_size,
        )
        return res
    finally:
        shutil.rmtree(merged_path)


evaluate_model_ray = ray.remote(num_cpus=1, num_gpus=1.0)(evaluate_model)


def merge_model(
    genotype: torch.Tensor,
    genome: ModelGenome,
    model_storage_path: str,
    merge_options: MergeOptions,
) -> str:
    # monkeypatch_tqdm()
    cfg = genome.genotype_merge_config(genotype)
    os.makedirs(model_storage_path, exist_ok=True)
    res = tempfile.mkdtemp(prefix="merged", dir=model_storage_path)
    run_merge(cfg, out_path=res, options=merge_options)
    return res


merge_model_ray = ray.remote(
    num_cpus=1,
    num_gpus=1,
    max_retries=3,
    retry_exceptions=[ConnectionError],
)(merge_model)


class MergeActorBase:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        batch_size: Optional[int] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.cache = LoaderCache()
        self.cache.setup(merge_options)
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.batch_size = batch_size

        if config.shuffle:
            monkeypatch_lmeval_shuffle()

        # monkeypatch_tqdm()


@ray.remote(num_cpus=1, num_gpus=1.0)
class OnDiskMergeEvaluator(MergeActorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Merging model")
        merged_path = merge_model(
            genotype, self.genome, self.model_storage_path, self.merge_options
        )
        logging.info(f"Model merged to {merged_path}")
        return evaluate_model(
            merged_path,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
            vllm=self.vllm,
            batch_size=self.batch_size,
        )


@ray.remote(num_cpus=1, num_gpus=1)
class InMemoryMergeEvaluator(MergeActorBase):
    model: Union[
        lm_eval.models.huggingface.HFLM, lm_eval.models.vllm_causallms.VLLM, None
    ] = None
    arch_info: Optional[ConfiguredArchitectureInfo] = None

    def __init__(
        self,
        *args,
        vllm: bool = False,
        **kwargs,
    ):
        # assert not vllm, "VLLM is not supported for in-memory merging"
        super().__init__(*args, vllm=vllm, **kwargs)

    def _maybe_init_model(self, config: MergeConfiguration):
        ai = get_architecture_info(self.genome._input_config_example)
        cfg_out = _model_out_config(
            config,
            ai,
            trust_remote_code=self.merge_options.trust_remote_code,
        )
        cfg_out.use_cache = True
        cfg_out.torch_dtype = torch.bfloat16

        if self.arch_info is not None:
            different = False
            for key in cfg_out.to_diff_dict():
                if key in ["architectures", "model_type"]:
                    # to get to here we must have --allow-crimes set, so let it ride
                    continue
                elif key in ["use_cache", "torch_dtype"]:
                    continue
                elif key.endswith("_token_id"):
                    # update our config but don't fail if it's different
                    setattr(self.arch_info.config, key, getattr(cfg_out, key, None))
                    continue

                if getattr(cfg_out, key) != getattr(self.arch_info.config, key, None):
                    logging.warn(f"Config key {key} changed, reinitializing model")
                    different = True
                    break

            if not different:
                return

        with NoInit():
            inner_model = (
                transformers.AutoModelForCausalLM.from_config(
                    cfg_out,
                    trust_remote_code=self.merge_options.trust_remote_code,
                    attn_implementation="flash_attention_2",
                    torch_dtype=torch.bfloat16,
                )
                .bfloat16()
                .cuda()
                .eval()
                .requires_grad_(False)
            )

        if self.vllm:
            self.model = lm_eval.models.vllm_causallms.VLLM(
                pretrained=inner_model,
                batch_size=self.batch_size or "auto",
                max_model_len=4096,
            )
        else:
            self.model = lm_eval.models.huggingface.HFLM(pretrained=inner_model)
        self.arch_info = ConfiguredArchitectureInfo(info=ai, config=cfg_out)
        logging.info("Model initialized")

    def evaluate(self, genotype: torch.Tensor) -> float:
        config = self.genome.genotype_merge_config(genotype)
        self._maybe_init_model(config)

        planner = MergePlanner(
            config,
            self.arch_info.info,
            self.merge_options,
            self.arch_info.config,
        )

        tasks = planner.plan_in_memory()

        param_dict = dict(self.model.model.named_parameters())

        stacked_mapping = {
            ".q_proj.": (".qkv_proj.", 0),
            ".k_proj.": (".qkv_proj.", 1),
            ".v_proj.": (".qkv_proj.", 2),
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
        }

        executor = Executor(tasks, math_device="cuda", storage_device="cuda")
        for tensor_task, value in executor.run(quiet=True):
            assert isinstance(tensor_task, ReturnTensor)
            name = tensor_task.weight_info.name

            if name in param_dict:
                param_dict[name].data.copy_(value, non_blocking=True)
            else:
                stacked = False
                for needle, (replacement, idx) in stacked_mapping.items():
                    if needle in name:
                        target = name.replace(needle, replacement)
                        param_dict[target].data[
                            idx * value.shape[0] : (idx + 1) * value.shape[0]
                        ] = value
                        stacked = True
                        break

                if not stacked:
                    raise ValueError(f"Unknown parameter {name}")

            del value

        return _eval_model(
            self.model,
            self.config.tasks,
            num_fewshot=self.config.num_fewshot,
            limit=self.config.limit,
        )

    def evaluate_genotype(
        self,
        genotype: torch.Tensor,
    ) -> float:
        return self.evaluate(genotype)


class ActorPoolEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        in_memory: bool = False,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if in_memory:
            self.actor_cls = InMemoryMergeEvaluator
        else:
            self.actor_cls = OnDiskMergeEvaluator

        self.actor_pool = ray.util.ActorPool(
            [
                self.actor_cls.remote(
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=model_storage_path,
                    vllm=vllm,
                )
                for _ in range(self.num_gpus)
            ]
        )

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        return list(
            self.actor_pool.map(
                lambda a, x: a.evaluate_genotype.remote(x),
                genotypes,
            )
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        return self.evaluate_genotypes([genotype])[0]


@ray.remote
class BufferedRayEvaluationStrategyActor:
    def __init__(
        self,
        config: EvolMergeConfiguration,
        genome: ModelGenome,
        merge_options: MergeOptions,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        num_gpus: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.config = config
        self.genome = genome
        self.merge_options = merge_options
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.input_queue = []
        self.batch_size = batch_size
        self._shutdown = False

    async def evaluate_genotype(self, genotype: np.ndarray):
        future_result = asyncio.Future()
        self.input_queue.append((genotype, future_result))
        return await future_result

    async def process_queue(self):
        merging: Dict[ray.ObjectRef, asyncio.Future] = {}
        merged: List[Tuple[asyncio.Future, ray.ObjectRef]] = []
        evaluating: Dict[ray.ObjectRef, asyncio.Future] = {}

        logging.info("Starting processing loop")

        try:
            while not self._shutdown:
                while self.input_queue and (len(merging) + len(merged) < self.num_gpus):
                    genotype, future_result = self.input_queue.pop(0)
                    merging[
                        merge_model_ray.remote(
                            genotype,
                            self.genome,
                            self.model_storage_path,
                            self.merge_options,
                        )
                    ] = future_result

                while merged and len(evaluating) < self.num_gpus:
                    future_result, merged_path = merged.pop()
                    evaluating[
                        evaluate_model_ray.remote(
                            merged_path,
                            self.config.tasks,
                            num_fewshot=self.config.num_fewshot,
                            limit=self.config.limit,
                            vllm=self.vllm,
                            batch_size=self.batch_size,
                        )
                    ] = future_result

                ready, _ = ray.wait(
                    list(merging.keys()) + list(evaluating.keys()),
                    num_returns=1,
                    fetch_local=False,
                    timeout=1,
                )
                for r in ready:
                    if r in merging:
                        future_result = merging.pop(r)
                        merged.append((future_result, r))
                    elif r in evaluating:
                        future_result = evaluating.pop(r)
                        future_result.set_result(await r)

                if (
                    not self.input_queue
                    and not merging
                    and not merged
                    and not evaluating
                ):
                    await asyncio.sleep(1)
        except Exception as e:
            logging.error("Error in processing loop", exc_info=e)
            raise

    async def shutdown(self):
        self._shutdown = True


class BufferedRayEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for buffered mode")

        super().__init__(*args, **kwargs)
        self.actor = BufferedRayEvaluationStrategyActor.options(
            max_concurrency=1000
        ).remote(
            self.config,
            self.genome,
            self.merge_options,
            model_storage_path=model_storage_path,
            vllm=vllm,
            num_gpus=self.num_gpus,
        )
        self.actor.process_queue.remote()

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        return ray.get([self.actor.evaluate_genotype.remote(x) for x in genotypes])

    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        return ray.get(self.actor.evaluate_genotype.remote(genotype))


@ray.remote
def evaluate_genotype_serial(
    genotype: np.ndarray,
    config: EvolMergeConfiguration,
    genome: ModelGenome,
    merge_options: MergeOptions,
    model_storage_path: Optional[str] = None,
    vllm: bool = False,
    batch_size: Optional[int] = None,
):
    pg = ray.util.placement_group([{"CPU": 1, "GPU": 1}], strategy="STRICT_PACK")
    strat = ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
        placement_group=pg
    )
    merged_path = merge_model_ray.options(scheduling_strategy=strat).remote(
        genotype, genome, model_storage_path, merge_options
    )
    res = ray.get(
        evaluate_model_ray.options(scheduling_strategy=strat).remote(
            merged_path,
            config.tasks,
            num_fewshot=config.num_fewshot,
            limit=config.limit,
            vllm=vllm,
            batch_size=batch_size,
        )
    )
    ray.util.remove_placement_group(pg)
    return res


class SerialEvaluationStrategy(EvaluationStrategyBase):
    def __init__(
        self,
        *args,
        model_storage_path: Optional[str] = None,
        vllm: bool = False,
        in_memory: bool = False,
        **kwargs,
    ):
        self.model_storage_path = model_storage_path
        self.vllm = vllm
        if in_memory:
            raise ValueError("In-memory evaluation is not supported for serial mode")
        super().__init__(*args, **kwargs)

    def evaluate_genotypes(self, genotypes: List[np.ndarray]) -> List[float]:
        return ray.get(
            [
                evaluate_genotype_serial.remote(
                    x,
                    self.config,
                    self.genome,
                    self.merge_options,
                    model_storage_path=self.model_storage_path,
                    vllm=self.vllm,
                    batch_size=self.batch_size,
                )
                for x in genotypes
            ]
        )

    def evaluate_genotype(self, genotype: np.ndarray) -> float:
        return self.evaluate_genotypes([genotype])[0]
