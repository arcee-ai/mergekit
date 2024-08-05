import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import click
import torch
from peft.tuners.lora import QuantLinear
from safetensors.torch import save_file
from torch.nn.functional import pad
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D

from mergekit.card import generate_card_lora
from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader


def low_rank_decomposition(
    weight: torch.Tensor, max_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a 2D matrix into low-rank matrices L and R using SVD.

    :param weight: The matrix to decompose, of shape (H, W)
    :param max_rank: The maximum rank of the decomposition
    :return: A tuple of tensors (L, R)
    """
    assert (
        weight.dim() == 2
    ), f"Only support 2D matrix, but input has {weight.dim()} dimensions."
    assert (
        max_rank >= 1
    ), f"Maximum rank must be a positive integer, but input max_rank={max_rank}."

    dtype = weight.dtype

    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

    final_rank = min(min(weight.shape), max_rank)

    # Distribute S to both to improve numerical precision.
    sqrt_S = torch.sqrt(torch.diag(S[:final_rank]))
    L = sqrt_S @ Vh[:final_rank, :]
    R = U[:, :final_rank] @ sqrt_S

    return L.to(dtype), R.to(dtype)


def decompose_delta_weight(
    base_weight: torch.Tensor,
    finetuned_weight: torch.Tensor,
    max_rank: int,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose the delta weight into low-rank matrices L and R.

    :param new_weight: The updated weight matrix after applying LoRA
    :param base_weight: The original weight matrix before LoRA
    :param max_rank: The maximum rank for the low-rank decomposition
    :param device: The device to perform computation on
    :return: A tuple of tensors (L, R)
    """
    assert (
        base_weight.size() == finetuned_weight.size()
    ), f"Mismatched dimensions: {base_weight.size()} != {finetuned_weight.size()}"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_weight = base_weight.to(device)
    finetuned_weight = finetuned_weight.to(device)

    delta_weight = finetuned_weight - base_weight

    L, R = low_rank_decomposition(delta_weight, max_rank)

    return L, R


def get_model_details(
    model_id: str, skip_undecomposable: bool
) -> List[Tuple[str, str, torch.Size]]:
    """
    Retrieve architectural details of a given pre-trained model.

    :param model_id: The identifier of the pre-trained model to load
    :param skip_undecomposable: Skip saving undecomposable modules
    :return: A list of tuples where each tuple contains:
             - type: The type of the module ('embedding', 'linear', or 'to_save')
             - name: The full name of the module
             - size: The dimensions of the module's weight tensor
    """

    # Avoid loading weights as we won't need them
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_id, state_dict={}, device_map="meta"
    )

    module_details = []

    for name, module in pretrained_model.named_modules():
        if module == pretrained_model.get_input_embeddings():
            # if isinstance(module, torch.nn.Embedding):
            module_details.append(("embedding", name, module.weight.size()))
        elif module == pretrained_model.get_output_embeddings():
            # if isinstance(module, torch.nn.Embedding):
            module_details.append(("output", name, module.weight.size()))
        elif hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            if (
                # SEE: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/model.py
                isinstance(
                    module,
                    (
                        torch.nn.Linear,
                        torch.nn.Conv2d,
                        bnb.nn.Linear4bit,
                        bnb.nn.Linear8bitLt,
                        QuantLinear,
                        Conv1D,
                    ),
                )
                or (
                    "Linear" in module.__class__.__name__
                    and module.__class__.__name__
                    not in ("LlamaLinearScalingRotaryEmbedding",)
                )
            ):
                module_details.append(("linear", name, module.weight.size()))
            elif not skip_undecomposable:
                module_details.append(("to_save", name, module.weight.size()))
            else:
                logging.info(f"Skipping undecomposable module '{name}'.")

    return module_details


def validate_and_combine_details(
    base_model_id: str,
    finetuned_model_id: str,
    skip_undecomposable: bool,
    extend_vocab: bool,
) -> List[Tuple[str, str]]:
    """
    Validate and combine details from a base model and a fine-tuned model.

    :param base_model_id: The identifier for the base model
    :param finetuned_model_id: The identifier for the fine-tuned model
    :param skip_undecomposable: Skip saving undecomposable modules
    :return: A list of tuples with the type and name of the validated/combined model layers
    """

    base_model_details = get_model_details(base_model_id, skip_undecomposable)
    finetuned_model_details = get_model_details(finetuned_model_id, skip_undecomposable)

    module_details = []

    base_model_embedding_size = None
    finetuned_model_embedding_size = None

    for i, (base_layer, finetuned_layer) in enumerate(
        zip(base_model_details, finetuned_model_details)
    ):
        base_type, base_name, base_size = base_layer
        finetuned_type, finetuned_name, finetuned_size = finetuned_layer

        assert (
            base_type == finetuned_type
        ), f"Layer type mismatch: {base_type} != {finetuned_type}"
        assert (
            base_name == finetuned_name
        ), f"Layer name mismatch: {base_name} != {finetuned_name}"

        if base_type == "embedding":
            base_model_embedding_size = base_size[0]

        if finetuned_type == "embedding":
            finetuned_model_embedding_size = finetuned_size[0]

        # Fine-tuned models with added vocab will have have their extra rows truncated unless `extend_vocab` is specified
        if base_type != "to_save" and finetuned_size[0] > base_size[0]:
            assert (
                base_size[1] == finetuned_size[1]
            ), f"Column dimension mismatch in layer '{base_name}': {base_size} != {finetuned_size}"

            if base_type == "embedding" or base_type == "output":
                if not extend_vocab:
                    logging.warning(
                        f"Finetuned module '{base_name}' will have {finetuned_size[0] - base_size[0]} rows truncated for weight decomposition! To preserve all embeddings, invoke script with --extend-vocab"
                    )
                else:
                    logging.warning(
                        f"Base module '{base_name}' will have {finetuned_size[0] - base_size[0]} rows added for weight decomposition. Make sure to call `model.resize_token_embeddings({finetuned_size[0]})` before applying LoRA for inference!"
                    )
            else:
                logging.warning(
                    f"Finetuned module '{base_name}' will have {finetuned_size[0] - base_size[0]} rows truncated for weight decomposition!"
                )

        else:
            assert (
                base_size == finetuned_size
            ), f"Dimension mismatch in layer '{base_name}': {base_size} != {finetuned_size}"

        module_details.append((base_type, base_name))

    return module_details, base_model_embedding_size, finetuned_model_embedding_size


def extract_lora(
    module_details: List[Tuple[str, str]],
    base_model_ref: ModelReference,
    finetuned_model_ref: ModelReference,
    max_rank: int,
    extend_vocab: bool,
    no_lazy_unpickle: bool,
    device: Optional[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Process module details to decompose weights and generate LoRA weights and ranks.

    :param module_details: List of module details.
    :param base_model_ref: Reference to the base model.
    :param finetuned_model_ref: Reference to the fine-tuned model.
    :param max_rank: The maximum rank for the low-rank decomposition.
    :param no_lazy_unpickle: Flag to disable lazy unpickle.
    :param device: The device to perform computation on.
    :return: A tuple containing LoRA weights dictionary and ranks dictionary.
    """

    base_loader = LazyTensorLoader(
        base_model_ref.tensor_index(), lazy_unpickle=(not no_lazy_unpickle)
    )
    finetuned_loader = LazyTensorLoader(
        finetuned_model_ref.tensor_index(), lazy_unpickle=(not no_lazy_unpickle)
    )

    lora_weights = {}
    ranks = {}

    for module_type, module_name in tqdm(module_details):
        base_weight = base_loader.get_tensor(f"{module_name}.weight")
        finetuned_weight = finetuned_loader.get_tensor(f"{module_name}.weight")

        if module_type == "to_save":
            lora_weights[
                f"base_model.model.{module_name}.weight"
            ] = finetuned_weight.to("cpu").contiguous()

            logging.info(
                f"[{module_type}] {module_name}: output_dims=({finetuned_weight.shape})"
            )

        else:
            if finetuned_weight.shape[0] > base_weight.shape[0]:
                if extend_vocab:
                    print(f"Extra tokens found!, module name : {module_name}")

                    new_base_weight = torch.empty(
                        finetuned_weight.shape, device=base_weight.device
                    )
                    new_base_weight.normal_(mean=0.0, std=0.02)

                    # Copy original base_weight values into the new tensor
                    new_base_weight[: base_weight.shape[0]] = base_weight

                    if module_type == "embedding" or module_type == "output":
                        lora_weights[
                            f"base_model.model.{module_name}.base_layer.weight"
                        ] = new_base_weight.to("cpu").contiguous()

                    base_weight = new_base_weight
                else:
                    logging.warning(
                        f"Finetuned module '{module_name}' will have {finetuned_weight.shape[0] - base_weight.shape[0]} rows truncated for weight decomposition!"
                    )
                    finetuned_weight = finetuned_weight[: base_weight.shape[0]]

            if module_type == "embedding":
                # These need to be transposed for some reason...
                lora_embedding_A, lora_embedding_B = decompose_delta_weight(
                    base_weight.T, finetuned_weight.T, max_rank, device=device
                )

                lora_weights[
                    f"base_model.model.{module_name}.lora_embedding_A"
                ] = lora_embedding_A.to("cpu").contiguous()
                lora_weights[
                    f"base_model.model.{module_name}.lora_embedding_B"
                ] = lora_embedding_B.to("cpu").contiguous()

                ranks[module_name] = lora_embedding_A.shape[0]

                logging.info(
                    f"[{module_type}] {module_name}: final_rank={ranks[module_name]}, "
                    f"input_dims=({base_weight.shape}), "
                    f"output_dims=({lora_embedding_A.shape}, {lora_embedding_B.shape})"
                )

            else:
                lora_A, lora_B = decompose_delta_weight(
                    base_weight, finetuned_weight, max_rank, device=device
                )

                lora_weights[
                    f"base_model.model.{module_name}.lora_A.weight"
                ] = lora_A.to("cpu").contiguous()
                lora_weights[
                    f"base_model.model.{module_name}.lora_B.weight"
                ] = lora_B.to("cpu").contiguous()

                ranks[module_name] = lora_A.shape[0]

                logging.info(
                    f"[{module_type}] {module_name}: final_rank={ranks[module_name]}, "
                    f"input_dims=({base_weight.shape}), "
                    f"output_dims=({lora_A.shape}, {lora_B.shape})"
                )

    return lora_weights, ranks


def reconstruct_invocation(args: Dict[str, Any]) -> str:
    """
    Reconstruct the command-line invocation string based on the given arguments.

    :param args: A dictionary containing the command arguments with keys matching the parameter names.
                 Expected keys are 'base_model', 'finetuned_model', 'out_path', 'no_lazy_unpickle',
                 'skip_undecomposable, 'max_rank', 'model_name', 'device' and 'verbose'.
    :return: The reconstructed command-line invocation string.
    """

    # Provide a default value for out_path if it's not in the dictionary
    out_path = args.get("out_path", "OUTPUT_PATH")

    invocation = f"mergekit-extract-lora {args['finetuned_model']} {args['base_model']} {out_path}"
    if args.get("no_lazy_unpickle"):
        invocation += " --no-lazy-unpickle"
    if args.get("skip_undecomposable"):
        invocation += " --skip-undecomposable"
    if args.get("max_rank"):
        invocation += f" --rank={args['max_rank']}"
    if args.get("extend_vocab"):
        invocation += " --extend-vocab"
    if args.get("model_name"):
        invocation += f" --model_name={args['model_name']}"
    if args.get("device"):
        invocation += f" --device={args['device']}"
    if args.get("verbose"):
        invocation += " --verbose"

    return invocation


def create_peft_config(
    base_model_name_or_path: str,
    rank: int,
    alpha: int,
    rank_pattern: Dict[str, int],
    alpha_pattern: Dict[str, int],
    target_modules: List[str],
    modules_to_save: List[str],
) -> Dict[str, Any]:
    """
    Create a PEFT (Parameter-Efficient Fine-Tuning) configuration dictionary.

    :param base_model_name_or_path: The path or name of the base model.
    :param rank: The rank for the low-rank adaptation.
    :param alpha: The scaling factor for low-rank adaptation.
    :param rank_pattern: A dictionary specifying rank patterns for different modules.
    :param alpha_pattern: A dictionary specifying alpha patterns for different modules.
    :param target_modules: A list of module names to apply the adaptation to.
    :param modules_to_save: A list of module names to save during the adaptation.
    :return: A dictionary containing the PEFT configuration.
    """
    return {
        "alpha_pattern": alpha_pattern,
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
        "modules_to_save": modules_to_save,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": rank_pattern,
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }


def save_model_and_config(
    lora_weights: Dict[str, torch.Tensor],
    ranks: Dict[str, int],
    extended: bool,
    embedding_size: int,
    module_details: List[Tuple[str, str]],
    invocation_args: Dict[str, Any],
) -> None:
    """
    Save the PEFT model and configuration to the specified output path.

    :param lora_weights: The LoRA weights.
    :param ranks: The ranks of the LoRA weights.
    :param module_details: Details of the model modules.
    :param invocation_args: The command-line invocation arguments.
    """

    base_model_ref = ModelReference.parse(invocation_args["base_model"])
    finetuned_model_ref = ModelReference.parse(invocation_args["finetuned_model"])
    out_path = invocation_args["out_path"]
    model_name = invocation_args["model_name"]

    # Work out the actual final rank and only retain those that were lower.
    final_max_rank = max(ranks.values())
    ranks = {k: v for k, v in ranks.items() if v != final_max_rank}

    lora_config = create_peft_config(
        base_model_name_or_path=base_model_ref.model.path,
        rank=final_max_rank,
        alpha=final_max_rank,  # Setting the alpha to the rank value as `peft` will scale the LoRA weights by alpha/r when applying the adapter
        rank_pattern=ranks,
        alpha_pattern=ranks,
        target_modules=list(
            set(
                module_name.split(".")[-1]
                for module_type, module_name in module_details
                if module_type != "to_save"
            )
        ),
        modules_to_save=list(
            set(
                module_name.split(".")[-1]
                for module_type, module_name in module_details
                if module_type == "to_save"
            )
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
        extended=extended,
        vocab_size=embedding_size,
        name=model_name,
    )

    with open(os.path.join(out_path, "README.md"), "w", encoding="utf-8") as fp:
        fp.write(card_md)

    logging.info(f"PEFT LoRA adapters saved to {out_path}")


@click.command("mergekit-extract-lora")
@click.argument("finetuned_model", type=str)
@click.argument("base_model", type=str)
@click.argument("out_path", type=click.Path())
@click.option(
    "--no-lazy-unpickle",
    type=bool,
    is_flag=True,
    default=False,
    help="Disable lazy unpickler (more stable, higher memory usage)",
)
@click.option(
    "--skip-undecomposable",
    type=bool,
    is_flag=True,
    default=False,
    help="Skip saving undecomposable modules in the LoRA",
)
@click.option(
    "--rank",
    "max_rank",
    type=int,
    default=32,
    help="The maximum rank for the low-rank decomposition",
)
@click.option(
    "--extend-vocab",
    is_flag=True,
    default=False,
    help="Extend vocabulary for models with additional tokens instead of truncating",
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
@click.option(
    "--verbose", "-v", type=bool, is_flag=True, default=False, help="Verbose logging"
)
def main(
    finetuned_model: str,
    base_model: str,
    out_path: str,
    no_lazy_unpickle: bool,
    skip_undecomposable: bool,
    max_rank: int,
    extend_vocab: bool,
    model_name: str,
    device: str,
    verbose: bool,
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
        "max_rank": max_rank,
        "extend_vocab": extend_vocab,
        "device": device,
        "out_path": out_path,
        "model_name": model_name,
        "no_lazy_unpickle": no_lazy_unpickle,
        "skip_undecomposable": skip_undecomposable,
        "verbose": verbose,
    }

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    os.makedirs(out_path, exist_ok=True)

    base_model_ref = ModelReference.parse(base_model)
    finetuned_model_ref = ModelReference.parse(finetuned_model)

    (
        module_details,
        base_model_embedding_size,
        finetuned_model_embedding_size,
    ) = validate_and_combine_details(
        ModelReference.parse(base_model).model.path,
        ModelReference.parse(finetuned_model).model.path,
        skip_undecomposable,
        extend_vocab,
    )

    lora_weights, ranks = extract_lora(
        module_details,
        base_model_ref,
        finetuned_model_ref,
        max_rank,
        extend_vocab,
        no_lazy_unpickle,
        device,
    )

    save_model_and_config(
        lora_weights,
        ranks,
        finetuned_model_embedding_size > base_model_embedding_size and extend_vocab,
        finetuned_model_embedding_size if extend_vocab else base_model_embedding_size,
        module_details,
        invocation_args,
    )


if __name__ == "__main__":
    main()
