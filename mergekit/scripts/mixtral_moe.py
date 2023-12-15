from typing import Dict, List, Optional, Union

import torch
import tqdm
import transformers
import typer
import yaml
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing_extensions import Annotated

import mergekit.architecture
from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader, TensorWriter

# Create a Mixtral MoE from a set of equally-sized Mistral (or Llama) models.
# Takes the path to a yml config and an output path.
# Config schema is the two classes below.


class Expert(BaseModel):
    source_model: str

    positive_prompts: List[str]
    negative_prompts: Optional[List[str]] = None
    noise_scale: Optional[float] = None

    @property
    def model_ref(self):
        return ModelReference.parse(self.source_model)


class MistralMOEConfig(BaseModel):
    base_model: str
    experts: List[Expert]
    gate_mode: str = "hidden"  # possible values: "hidden", "cheap_embed", "random"
    # "hidden" uses hidden state vectors for the given prompts for each layer
    # "cheap_embed" uses the average of token embeddings for the prompts, same for each layer
    # "random" is random


def get_hidden_states(
    model: Union[MistralForCausalLM, LlamaForCausalLM],
    tokenized: transformers.BatchEncoding,
    average: bool = True,
) -> List[torch.Tensor]:
    with torch.no_grad():
        output: CausalLMOutputWithPast = model(
            **tokenized.to(model.device), output_hidden_states=True, return_dict=True
        )
    hidden_states = torch.stack(
        output.hidden_states[:-1]
    )  # (num_layers, batch_size, seq_len, hidden_size)
    if average:
        # use average over sequence
        hidden_states = hidden_states.sum(dim=2) / hidden_states.shape[2]
    else:
        # take last value
        hidden_states = hidden_states[:, :, -1, :]
    return hidden_states.sum(dim=1) / hidden_states.shape[1]


def get_cheap_embedding(
    embed: torch.Tensor,
    tokenized: Dict[str, torch.Tensor],
    num_layers: int,
) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(
        tokenized["input_ids"], num_classes=32000
    )  # (batch_size, seq_len, 32000)
    h = onehot.float() @ embed.float()  # (batch_size, seq_len, hidden_size)
    embedded = (h * tokenized["attention_mask"].unsqueeze(-1)).sum(dim=1)
    res = embedded / embedded.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return res.unsqueeze(0).repeat(num_layers, 1, 1, 1)


def tokenize_prompts(
    prompts: List[str], tokenizer: transformers.PreTrainedTokenizerBase
):
    return tokenizer(
        [tokenizer.bos_token + p for p in prompts],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )


def get_gate_params(
    model_ref: ModelReference,
    tokenizer: transformers.PreTrainedTokenizerBase,
    experts: List[Expert],
    mode: str = "hidden",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    gate_vecs = []
    _do_it = None

    model_cfg = model_ref.config()

    if mode == "random":
        return torch.randn(
            (model_cfg.num_hidden_layers, len(experts), model_cfg.hidden_size)
        )
    elif mode == "cheap_embed":
        embed = LazyTensorLoader(model_ref.tensor_index()).get_tensor(
            "model.embed_tokens.weight"
        )
        _do_it = lambda tokenized: get_cheap_embedding(
            embed, tokenized, num_layers=model_cfg.num_hidden_layers
        )
    elif mode == "hidden":
        model = AutoModelForCausalLM.from_pretrained(
            model_ref.path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        _do_it = lambda tokenized: get_hidden_states(
            model, tokenized=tokenized, average=True
        )

    gate_vecs = []
    for expert in tqdm.tqdm(experts, desc="expert prompts"):
        hidden_states = _do_it(tokenize_prompts(expert.positive_prompts, tokenizer))
        if expert.negative_prompts:
            hidden_states -= _do_it(
                tokenize_prompts(expert.negative_prompts, tokenizer)
            )

        hidden_states /= hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gate_vecs.append(hidden_states)
    gate_vecs = torch.stack(gate_vecs, dim=0)  # (num_expert, num_layer, hidden_size)
    return gate_vecs.permute(1, 0, 2)


def build(
    config: MistralMOEConfig,
    out_path: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    base_model = ModelReference.parse(config.base_model)
    base_cfg = base_model.config()
    base_cfg_mistral = MistralConfig(**base_cfg.to_dict())
    base_cfg_mistral.sliding_window = base_cfg.max_position_embeddings
    base_cfg_mistral.max_position_embeddings = 32768

    out_cfg = MixtralConfig(**base_cfg_mistral.to_dict())
    out_cfg.architectures = ["MixtralForCausalLM"]
    out_cfg.num_local_experts = len(config.experts)
    out_cfg.save_pretrained(out_path)

    loaders: Dict[ModelReference, LazyTensorLoader] = {}
    for model in tqdm.tqdm(
        [base_model] + [e.model_ref for e in config.experts], desc="Warm up loaders"
    ):
        loaders[model] = LazyTensorLoader(model.tensor_index())

    base_loader = loaders.get(base_model)
    writer = TensorWriter(out_path=out_path)

    print("Copying parameters...")
    MISTRAL_INFO = mergekit.architecture.MISTRAL_INFO
    for tensor_name in MISTRAL_INFO.pre_weight_names + MISTRAL_INFO.post_weight_names:
        writer.save_tensor(tensor_name, base_loader.get_tensor(tensor_name))

    for name_format in tqdm.tqdm(MISTRAL_INFO.layer_weight_formats()):
        for layer_idx in range(base_cfg.num_hidden_layers):
            tensor_name = name_format.format(idx=layer_idx)

            if ".mlp." in name_format:
                for moe_index, expert in enumerate(config.experts):
                    expert_name = tensor_name.replace(
                        ".mlp.gate_proj", f".block_sparse_moe.experts.{moe_index}.w1"
                    )
                    expert_name = expert_name.replace(
                        ".mlp.down_proj", f".block_sparse_moe.experts.{moe_index}.w2"
                    )
                    expert_name = expert_name.replace(
                        ".mlp.up_proj", f".block_sparse_moe.experts.{moe_index}.w3"
                    )
                    expert_loader = loaders.get(expert.model_ref)
                    tensor = expert_loader.get_tensor(tensor_name)
                    if expert.noise_scale:
                        tensor += torch.randn_like(tensor) * expert.noise_scale
                    writer.save_tensor(expert_name, tensor, clone=True)
                continue
            writer.save_tensor(tensor_name, base_loader.get_tensor(tensor_name))

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model.path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.bos_token_id

    print("Getting gate parameters...")
    gate_vecs = get_gate_params(
        base_model,
        tokenizer,
        config.experts,
        mode=config.gate_mode,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    # gate_vecs: (num_layers, num_experts, hidden_size)

    for layer_idx in range(base_cfg.num_hidden_layers):
        writer.save_tensor(
            f"model.layers.{layer_idx}.block_sparse_moe.gate.weight",
            gate_vecs[layer_idx, :, :].contiguous(),
        )
    writer.finalize()
    print("Saving tokenizer...")
    tokenizer.save_pretrained(out_path, safe_serialization=True)
    print("Done.")


def main(
    config_path: Annotated[
        str, typer.Argument(help="Path to a YML config file", metavar="PATH")
    ],
    out_path: Annotated[str, typer.Argument(help="Output model path", metavar="PATH")],
    load_in_4bit: Annotated[
        bool, typer.Option(help="Load model in 4bit for computing hidden states")
    ] = False,
    load_in_8bit: Annotated[
        bool, typer.Option(help="Load model in 8bit for computing hidden states")
    ] = False,
):
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.load(file, yaml.SafeLoader)

    config = MistralMOEConfig.model_validate(data)
    build(
        config, out_path=out_path, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
    )


def _main():
    typer.run(main)


if __name__ == "__main__":
    _main()
