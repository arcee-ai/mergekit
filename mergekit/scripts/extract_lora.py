import json
import os
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import click
import torch
from peft.tuners.lora import QuantLinear
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader
from mergekit.options import add_merge_options


def _low_rank_decomposition(
    weight: torch.Tensor, reduced_rank: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a 2D matrix into low-rank matrices A and B using SVD.a

    :param weight: The matrix to decompose, of shape (H, W)
    :param reduced_rank: The final rank of the decomposition
    :return: A tuple of tensors (A, B)
    """
    if weight.dim() != 2:
        raise ValueError(
            f"Only support 2D matrix, but your input has {weight.dim()} dimensions."
        )

    # SVD Decomposition
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    # Truncated matrices
    A = Vh[:reduced_rank, :]
    B = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])

    return A, B


def decompose_delta_weight(
    new_weight: torch.Tensor,
    base_weight: torch.Tensor,
    reduced_rank: int,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    new_weight = new_weight.to(device)
    base_weight = base_weight.to(device)

    """
    Decompose the delta weight into low-rank matrices A and B.

    :param new_weight: The updated weight matrix after applying LoRA.
    :param base_weight: The original weight matrix before LoRA.
    :param reduced_rank: The rank for the low-rank decomposition.
    :param device: The device to perform computation on.
    :return: A tuple of tensors (A, B)
    """
    delta_weight = new_weight - base_weight

    max_rank = min(delta_weight.shape)
    assert (
        reduced_rank <= max_rank
    ), f"The specified rank ({reduced_rank}) must be smaller than or equal to the rank of the weight matrices ({max_rank})"

    A, B = _low_rank_decomposition(delta_weight, reduced_rank=reduced_rank)

    return A, B


def find_all_linear_names(model: PreTrainedModel) -> List[str]:
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names.append(name)

    return names


def get_linear_module_names(model_id: str) -> List[str]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, state_dict={}, device_map="meta"
    )  # avoid loading weights as we won't need them
    linear_module_names = find_all_linear_names(model)

    return linear_module_names


def create_peft_config(
    base_model_name_or_path: str, rank: int, alpha: int, target_modules: List[str]
) -> Dict[str, Any]:
    return {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_dropout": 0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }


@click.command("mergekit-extract-lora")
@click.argument("out_path", type=click.Path())
@click.option("--base-model", type=str, help="Model ID or path to use as base model")
@click.option(
    "--finetuned-model",
    type=str,
    help="Model ID or path to use as PEFT extraction target model",
)
@click.option(
    "--rank", type=int, default=32, help="Rank for the low-rank decomposition"
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="PyTorch device to perform SVD computation on",
)
def main(
    out_path: str, base_model: str, finetuned_model: str, rank: int, device: str
) -> None:
    """
    Decomposes delta weights between a base model and a finetuned model and saves a PEFT model.
    """
    os.makedirs(out_path, exist_ok=True)

    base_model_ref = ModelReference.parse(base_model)
    finetuned_model_ref = ModelReference.parse(finetuned_model)

    base_model_config = AutoConfig.from_pretrained(base_model_ref.model.path)

    linear_module_names = get_linear_module_names(base_model_ref.model.path)
    finetuned_model_linear_module_names = get_linear_module_names(
        finetuned_model_ref.model.path
    )

    assert set(linear_module_names) == set(
        finetuned_model_linear_module_names
    ), "Model architecture mismatch"

    base_loader = LazyTensorLoader(base_model_ref.tensor_index(), lazy_unpickle=True)
    finetuned_loader = LazyTensorLoader(
        finetuned_model_ref.tensor_index(), lazy_unpickle=True
    )

    lora_weights = {}
    for layer_name in tqdm(linear_module_names):
        base_weight = base_loader.get_tensor(f"{layer_name}.weight")
        finetuned_weight = finetuned_loader.get_tensor(f"{layer_name}.weight")

        lora_A, lora_B = decompose_delta_weight(
            finetuned_weight, base_weight, rank, device=device
        )

        lora_weights[f"base_model.model.{layer_name}.lora_A.weight"] = lora_A.to(
            "cpu"
        ).contiguous()
        lora_weights[f"base_model.model.{layer_name}.lora_B.weight"] = lora_B.to(
            "cpu"
        ).contiguous()

    lora_config = create_peft_config(
        base_model_name_or_path=base_model_ref.model.path,
        alpha=rank,  # Setting the alpha to the reduced rank value (instead of alpha value used) seems to give better performance. Further testing would be needed to understand what is the optimal alpha value to use
        rank=rank,
        target_modules=list(
            set([module_name.split(".")[-1] for module_name in linear_module_names])
        ),
    )

    with open(os.path.join(out_path, "adapter_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    save_file(lora_weights, os.path.join(out_path, "adapter_model.safetensors"))

    print(f"PEFT LoRA adapters saved to {out_path}")


if __name__ == "__main__":
    main()
