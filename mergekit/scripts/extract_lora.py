import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import click
import torch
from peft.tuners.lora import QuantLinear
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from mergekit.card import generate_card_lora
from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader


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

    dtype = weight.dtype

    # SVD Decomposition
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

    # Truncated matrices
    A = Vh[:reduced_rank, :]
    B = U[:, :reduced_rank] @ torch.diag(S[:reduced_rank])

    return A.to(dtype), B.to(dtype)


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


def reconstruct_invocation(args):
    """
    Reconstructs the command-line invocation string based on the given arguments stored in a dictionary.

    Parameters:
    - args: A dictionary containing the command arguments with keys matching the parameter names.
      Expected keys are 'base_model', 'finetuned_model', 'out_path', 'no_lazy_unpickle', 'desired_rank', 'model_name' and 'device'.

    Returns:
    - The reconstructed command-line invocation string.
    """
    # Provide a default value for out_path if it's not in the dictionary
    out_path = args.get("out_path", "OUTPUT_PATH")

    invocation = f"mergekit-extract-lora {args['base_model']} {args['finetuned_model']} {out_path}"
    if args.get("no_lazy_unpickle"):
        invocation += " --no-lazy-unpickle"
    invocation += f" --rank={args['desired_rank']}"
    if args.get("model_name"):
        invocation += f" --model_name={args['model_name']}"
    if args.get("device"):
        invocation += f" --device={args['device']}"

    return invocation


@click.command("mergekit-extract-lora")
@click.argument("finetuned_model", type=str)
@click.argument("base_model", type=str)
@click.argument("out_path", type=click.Path())
@click.option(
    "--no-lazy-unpickle",
    is_flag=True,
    help="Disable lazy unpickler (more stable, higher memory usage)",
)
@click.option(
    "--rank",
    "desired_rank",
    type=int,
    default=32,
    help="Rank for the low-rank decomposition",
)
@click.option(
    "--model_name",
    type=str,
    default=None,
    help="Name of the resulting model (shown in the model card)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="PyTorch device to perform SVD computation on",
)
def main(
    finetuned_model: str,
    base_model: str,
    out_path: str,
    no_lazy_unpickle: bool,
    desired_rank: int,
    model_name: str,
    device: str,
) -> None:
    """
    Decomposes delta weights between a base model and a finetuned model, saving a PEFT model to the specified output path.

    \b
    Arguments:
    FINETUNED_MODEL - the model ID or path to use as the PEFT extraction target model.
    BASE_MODEL - the model ID or path to use as the base model.
    OUT_PATH - the output path where the PEFT model will be saved.
    """

    invocation_args = {
        "base_model": base_model,
        "finetuned_model": finetuned_model,
        "desired_rank": desired_rank,
        "device": device,
        "out_path": out_path,
        "model_name": model_name,
        "no_lazy_unpickle": no_lazy_unpickle,
    }

    os.makedirs(out_path, exist_ok=True)

    base_model_ref = ModelReference.parse(base_model)
    finetuned_model_ref = ModelReference.parse(finetuned_model)

    linear_module_names = get_linear_module_names(base_model_ref.model.path)
    finetuned_model_linear_module_names = get_linear_module_names(
        finetuned_model_ref.model.path
    )

    assert set(linear_module_names) == set(
        finetuned_model_linear_module_names
    ), "Model architecture mismatch"

    base_loader = LazyTensorLoader(
        base_model_ref.tensor_index(), lazy_unpickle=(not no_lazy_unpickle)
    )
    finetuned_loader = LazyTensorLoader(
        finetuned_model_ref.tensor_index(), lazy_unpickle=(not no_lazy_unpickle)
    )

    lora_weights = {}
    for layer_name in tqdm(linear_module_names):
        base_weight = base_loader.get_tensor(f"{layer_name}.weight")
        finetuned_weight = finetuned_loader.get_tensor(f"{layer_name}.weight")

        lora_A, lora_B = decompose_delta_weight(
            finetuned_weight, base_weight, desired_rank, device=device
        )

        lora_weights[f"base_model.model.{layer_name}.lora_A.weight"] = lora_A.to(
            "cpu"
        ).contiguous()
        lora_weights[f"base_model.model.{layer_name}.lora_B.weight"] = lora_B.to(
            "cpu"
        ).contiguous()

    lora_config = create_peft_config(
        base_model_name_or_path=base_model_ref.model.path,
        alpha=desired_rank,  # Setting the alpha to the reduced rank value as `peft` will scale the LoRA weights by alpha/r when applying the adapter
        rank=desired_rank,
        target_modules=list(
            set([module_name.split(".")[-1] for module_name in linear_module_names])
        ),
    )

    with open(os.path.join(out_path, "adapter_config.json"), "w") as f:
        json.dump(lora_config, f, indent=2)

    save_file(lora_weights, os.path.join(out_path, "adapter_model.safetensors"))

    invocation_args.pop("out_path")  # don't include out_path for privacy
    invocation = reconstruct_invocation(invocation_args)

    card_md = generate_card_lora(
        base_model_ref=base_model_ref,
        finetuned_model_ref=finetuned_model_ref,
        invocation=invocation,
        name=model_name,
    )

    with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as fp:
        fp.write(card_md)

    logging.info(f"PEFT LoRA adapters saved to {out_path}")


if __name__ == "__main__":
    main()
