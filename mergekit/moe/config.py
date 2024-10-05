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
from typing import List, Optional

from pydantic import BaseModel

from mergekit.common import ModelReference


class Expert(BaseModel):
    """
    Defines a model to be used as a set of layerwise experts in a MoE model.
    """

    source_model: ModelReference

    positive_prompts: Optional[List[str]] = None
    negative_prompts: Optional[List[str]] = None
    noise_scale: Optional[float] = None
    residual_scale: Optional[float] = None


class MoEMergeConfig(BaseModel):
    """
    Configuration for merging a set of "expert" models into a MoE model.
    """

    base_model: ModelReference
    experts: List[Expert]
    gate_mode: str = (
        "hidden"  # possible values: "hidden", "cheap_embed", "random", "uniform_random"
    )
    # "hidden" uses hidden state vectors for the given prompts for each layer
    # "cheap_embed" uses the average of token embeddings for the prompts, same for each layer
    # "random" is random
    # "uniform_random" matches default initialization for torch.nn.Linear
    dtype: Optional[str] = None
    experts_per_token: int = 2
    shared_experts: Optional[List[Expert]] = None
    architecture: Optional[str] = None


def is_bad_config(config: MoEMergeConfig, allow_all_same: bool = False) -> bool:
    if config.experts_per_token < 1:
        logging.error("Experts per token must be >= 1")
        return True

    if len(config.experts) < config.experts_per_token:
        logging.error("Must include at least as many experts as experts_per_token.")
        return True

    if config.gate_mode == "random":
        return False  # eh we're good

    for expert_idx, expert in enumerate(config.experts):
        if not expert.positive_prompts:
            logging.error(f"Expert {expert_idx} has no positive prompts.")
            return True

    def prompt_tup(e: Expert):
        return (tuple(e.positive_prompts), tuple(e.negative_prompts or []))

    # let's just nip this trend in the bud
    p_first = prompt_tup(config.experts[0])
    if all(prompt_tup(e) == p_first for e in config.experts[1:]):
        logging.error(
            "Your positive and negative prompts are identical for all experts. This will not produce a functioning MoE."
        )
        logging.error(
            "For each expert, `positive_prompts` must contain one or more example prompt reflecting what should be routed to that expert."
        )
        return True

    if not allow_all_same:
        if all(
            e.source_model == config.experts[0].source_model for e in config.experts[1:]
        ):
            logging.error(
                "All of your expert models are the same. This will produce "
                "a model that uses more resources but gives the exact same output. "
                "If you plan to train the model after merging, proceed with the "
                "--i-understand-this-is-not-useful-without-training flag."
            )
            return True
