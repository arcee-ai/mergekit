# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
from typing import List, Optional

from pydantic import BaseModel, model_validator

from mergekit.evo.genome import ModelGenomeDefinition


class TaskConfiguration(BaseModel, frozen=True):
    name: str
    weight: float = 1.0
    metric: str = "acc,none"

    @model_validator(mode="before")
    def validate_string(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class EvolMergeConfiguration(BaseModel, frozen=True):
    genome: ModelGenomeDefinition
    tasks: List[TaskConfiguration]
    limit: Optional[int] = None
    num_fewshot: Optional[int] = None
    shuffle: bool = False
    random_init: bool = False
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool = True


NAUGHTY_PREFIXES = [
    "mmlu",
    "hendrycks",
    "agieval",
    "gsm8k",
    "hellaswag",
    "winogrande",
    "arc_",
    "ai2_arc",
    "truthfulqa",
    "bigbench",
    "piqa",
    "openbookqa",
    "leaderboard",
]


def check_for_naughty_config(config: EvolMergeConfiguration, allow: bool = False):
    """
    Check if the given configuration is naughty and should be disallowed.

    mergekit-evolve is perfectly set up to directly optimize against the test set
    of common benchmarks, which just makes the world a worse place. There are
    cases where this is useful but it deserves a giant honking warning.
    """
    suffix = ""
    if not allow:
        suffix = (
            " To proceed, set the "
            "--i-understand-the-depths-of-the-evils-i-am-unleashing flag."
        )
    for task in config.tasks:
        for prefix in NAUGHTY_PREFIXES:
            if task.name.startswith(prefix):
                if task.name.endswith("_train"):
                    # there aren't any tasks that match this pattern in base
                    # lm-eval, but it'd be a sane thing to do to add tasks for
                    # the training sets of these benchmarks. don't warn about
                    # them
                    continue

                message = (
                    f"Task {task.name} is a common benchmark task. "
                    "Optimizing against this task directly is unsporting at best "
                    "and outright malicious at worst. Using mergekit-evolve to "
                    "game benchmarks will be a black mark on your name for a "
                    f"thousand generations.{suffix}"
                )
                if not allow:
                    raise ValueError(message)
                else:
                    logging.warning(message)
