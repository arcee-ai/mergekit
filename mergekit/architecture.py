# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import importlib.resources
import logging
import re
import string
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from transformers import PretrainedConfig
from typing_extensions import Literal

import mergekit._data.architectures
from mergekit.io.lazy_tensor_loader import ShardedTensorIndex


class WeightInfo(BaseModel, frozen=True):
    """Information about an individual weight tensor in a model.

    Attributes:
        name (str):
            The name of the tensor representing the weight.
        is_embed (bool):
            Indicates whether the weight is for an embedding or language model head.
        input_space (Optional[str]):
            The name of the input space associated with the weight, if applicable.
        output_space (Optional[str]):
            The name of the output space associated with the weight, if applicable.
        optional (bool):
            Indicates whether the weight can be omitted from a model.
        aliases (Optional[List[str]]):
            List of alternative names for the weight, if applicable.
        tied_names (Optional[List[str]]):
            List of names for weights that are tied to this weight, if applicable.
        force_dtype (Optional[str]):
            Mandatory dtype for the weight, if applicable.
    """

    name: str
    is_embed: bool = False
    input_space: Optional[str] = None
    output_space: Optional[str] = None
    optional: bool = False
    tied: bool = False
    aliases: Optional[Tuple[str, ...]] = None
    tied_names: Optional[Tuple[str, ...]] = None
    force_dtype: Optional[str] = None
    head_split: Literal[None, "input", "output"] = None
    is_kq: Optional[bool] = False


class ProceduralSpaceInfo(BaseModel, frozen=True):
    """Defines a procedural space computed from one or more other spaces.

    Currently only supports residual connections.

    Attributes:
        name (str): The name of the space defined.
        type (str): The type of procedural space.
        inputs (List[str]): List of names of spaces used to define this space."""

    name: str
    type: Literal["residual"]
    inputs: List[str]


class ArchitectureInfo(ABC):
    @abstractmethod
    def name(self) -> str:
        """Return the name of the architecture."""
        ...

    @abstractmethod
    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return a list of all weights preceding the first layer."""
        ...

    @abstractmethod
    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return a list of all weights following the final layer."""
        ...

    @abstractmethod
    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        """Return a list of all weights associated with a given layer."""
        ...

    @abstractmethod
    def sliceable(self) -> bool:
        """
        Return True if the layers of this architecture can be meaningfully sliced.
        """
        ...

    def num_layers_config_key(self) -> str:
        """Key in config that represents number of layers"""
        return "num_hidden_layers"

    def num_layers(self, config: PretrainedConfig) -> int:
        """Return the number of layers in a model."""
        return getattr(config, self.num_layers_config_key())

    def all_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """Return all weights associated with a model."""
        num_layers = self.num_layers(config)
        res = list(self.pre_weights(config))
        for layer_idx in range(num_layers):
            res.extend(self.layer_weights(layer_idx, config))
        res.extend(self.post_weights(config))
        return res

    def procedural_spaces(self, config: PretrainedConfig) -> List[ProceduralSpaceInfo]:
        """Return a list of all procedurally defined spaces in a model."""
        return []

    def has_defined_spaces(self) -> bool:
        """
        Return True if this architecture defines space information needed for
        matching-based merge methods.
        """
        return False


class ConfiguredArchitectureInfo(BaseModel, frozen=True, arbitrary_types_allowed=True):
    info: ArchitectureInfo
    config: PretrainedConfig

    def name(self) -> str:
        return self.info.name()

    def num_layers(self) -> int:
        return self.info.num_layers(self.config)

    def pre_weights(self) -> List[WeightInfo]:
        return self.info.pre_weights(self.config)

    def post_weights(self) -> List[WeightInfo]:
        return self.info.post_weights(self.config)

    def layer_weights(self, index: int) -> List[WeightInfo]:
        return self.info.layer_weights(index, self.config)

    def procedural_spaces(self) -> List[ProceduralSpaceInfo]:
        return self.info.procedural_spaces(self.config)

    def all_weights(self) -> List[WeightInfo]:
        return self.info.all_weights(self.config)


class JSONLayerTemplates(BaseModel, frozen=True):
    weights: List[WeightInfo]
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None


class JSONArchitectureDefinition(BaseModel, frozen=True):
    expected_model_type: str = Field(alias="model_type")
    architectures: List[str]
    pre_weights: List[WeightInfo]
    layer_templates: JSONLayerTemplates
    post_weights: List[WeightInfo]
    procedural_spaces: Optional[List[ProceduralSpaceInfo]] = None
    num_layers_config_key: Optional[str] = None


class TemplateWithArithmetic(string.Template):
    idpattern = r"(?a:[_a-z][_a-z0-9]*([+-]1)?)"


def _template_substitution(
    template: str, num_layers: int, layer_idx: Optional[int] = None
) -> str:
    if "{" not in template:
        return template

    substitutions = {
        "num_layers": num_layers,
        "num_layers+1": num_layers + 1,
        "num_layers-1": num_layers - 1,
    }

    if layer_idx is not None:
        substitutions.update(
            {
                "layer_index": layer_idx,
                "layer_index+1": layer_idx + 1,
                "layer_index-1": layer_idx - 1,
            }
        )

    return TemplateWithArithmetic(template).substitute(substitutions)


def _hierarchy(names, layer_prefix=r"\.\d+\.") -> Dict[str, List[str]]:
    hierarchy = defaultdict(list)

    # Regular expression to match layers (denoted by .{integer}. by default)
    layer_pattern = re.compile(layer_prefix)

    if names:
        for name in names:
            # Find the layer part of the string (e.g., 'model.layers.0.')
            match = layer_pattern.search(name)
            if match:
                # Extract everything up to the layer identifier
                layer_prefix = name[: match.end() - 1]  # e.g., 'model.layers.0'
                # Extract the parameter name after the layer identifier
                param_name = name[match.end() :]  # e.g., 'input_layernorm.weight'
                # Add the parameter name to the corresponding layer in the hierarchy
                hierarchy[layer_prefix].append(param_name)
            else:
                hierarchy[name].append("")

    return hierarchy


class AutomaticArchitectureInfo(ArchitectureInfo, BaseModel):
    arch_name: str = Field(default="")
    parameter_names: List[str] = Field(default_factory=list)
    embed: List[str] = Field(default_factory=list)
    layered_parameter_names: Dict[str, List[str]] = Field(default_factory=dict)
    prefix_tracker: Dict[str, str] = Field(default_factory=dict)
    post_fill_parameters: bool = False

    def __init__(
        self,
        arch_name: str,
        parameter_names: List[str],
        prefix_tracker: Optional[Dict[str, str]] = None,
        post_fill_parameters: bool = False,
    ):
        super().__init__()
        self.arch_name = arch_name
        self.parameter_names = parameter_names
        self.layered_parameter_names = _hierarchy(self.parameter_names)
        self.prefix_tracker = prefix_tracker or {}
        self.embed = self._find_embed_params()
        self.post_fill_parameters = post_fill_parameters

    def _find_embed_params(self) -> List[str]:
        """Identify embedding parameters (e.g., 'lm_head', 'embed') that may require special handling."""
        embed_params = []
        for name in self.parameter_names:
            if any(embedding_name in name for embedding_name in ["lm_head", "embed"]):
                embed_params.append(name)
        return embed_params

    def name(self) -> str:
        """Returns the architecture name."""
        return self.arch_name

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """This architecture does not distinguish pre-weights."""
        return []

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        """This architecture does not distinguish post-weights."""
        return []

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        """
        Retrieves the weights for a specified layer, adjusting names for prefixes if applicable.
        """
        layer_name = list(self.layered_parameter_names.keys())[index]
        adjusted_layer_name = self._adjust_layer_name(layer_name, config)

        weights = [
            WeightInfo(
                name=f"{adjusted_layer_name}.{param}" if param else adjusted_layer_name,
                is_embed=(layer_name in self.embed),
            )
            for param in self.layered_parameter_names[layer_name]
        ]
        return (
            weights
            if weights
            else [
                WeightInfo(
                    name=adjusted_layer_name, is_embed=(layer_name in self.embed)
                )
            ]
        )

    def _adjust_layer_name(self, layer_name: str, config: PretrainedConfig) -> str:
        """Adjust layer names by removing any prefix as indicated in the prefix tracker."""
        if config and config.name_or_path in self.prefix_tracker:
            prefix = self.prefix_tracker.get(config.name_or_path, "")
            if layer_name.startswith(prefix):
                return layer_name[len(prefix) :]
        return layer_name

    def sliceable(self) -> bool:
        """Indicates if the architecture supports slicing."""
        return True

    def num_layers(self, config: PretrainedConfig) -> int:
        """Returns the number of layers based on layered parameter names."""
        return len(self.layered_parameter_names)


class JsonArchitectureInfo(ArchitectureInfo, BaseModel, frozen=True):
    definition: JSONArchitectureDefinition

    def _substitute(
        self,
        item: Union[WeightInfo, ProceduralSpaceInfo],
        config: PretrainedConfig,
        layer_idx: Optional[int] = None,
    ) -> Union[WeightInfo, ProceduralSpaceInfo]:
        num_layers = self.num_layers(config)

        obj_dict = item.model_dump(mode="json", exclude_unset=True)
        for key in obj_dict:
            if isinstance(obj_dict[key], str):
                obj_dict[key] = _template_substitution(
                    obj_dict[key], num_layers, layer_idx
                )
            elif isinstance(obj_dict[key], list):
                obj_dict[key] = [
                    (
                        _template_substitution(s, num_layers, layer_idx)
                        if isinstance(s, str)
                        else s
                    )
                    for s in obj_dict[key]
                ]
        return type(item).model_validate(obj_dict)

    def name(self) -> str:
        return self.definition.expected_model_type

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            self._substitute(wi, config=config) for wi in self.definition.pre_weights
        ]

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        return [
            self._substitute(wi, config=config, layer_idx=index)
            for wi in self.definition.layer_templates.weights
        ]

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return [
            self._substitute(wi, config=config) for wi in self.definition.post_weights
        ]

    def sliceable(self) -> bool:
        return True

    def procedural_spaces(self, config: PretrainedConfig) -> List[ProceduralSpaceInfo]:
        res = []
        for s in self.definition.procedural_spaces or []:
            res.append(self._substitute(s, config=config))
        for idx in range(self.num_layers(config)):
            for s in self.definition.layer_templates.procedural_spaces or []:
                res.append(self._substitute(s, config=config, layer_idx=idx))
        return res

    def has_defined_spaces(self) -> bool:
        if (
            self.definition.procedural_spaces
            or self.definition.layer_templates.procedural_spaces
        ):
            return True
        for wi in (
            self.definition.layer_templates.weights
            + self.definition.pre_weights
            + self.definition.post_weights
        ):
            if wi.input_space or wi.output_space:
                return True
        return False

    def num_layers_config_key(self) -> str:
        return self.definition.num_layers_config_key


class MixtralTensorNames(ArchitectureInfo, BaseModel):
    ARCHITECTURE_NAME: ClassVar[str] = "MixtralForCausalLM"
    num_local_experts: int

    def name(self) -> str:
        return "mixtral"

    @classmethod
    def from_config(cls, config: PretrainedConfig):
        return MixtralTensorNames(num_local_experts=config.num_local_experts)

    def pre_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_INFO.pre_weights(config)

    def post_weights(self, config: PretrainedConfig) -> List[WeightInfo]:
        return MISTRAL_INFO.post_weights(config)

    def num_layers_config_key(self) -> str:
        return MISTRAL_INFO.num_layers_config_key()

    def layer_weights(
        self, index: int, config: PretrainedConfig
    ) -> Optional[List[WeightInfo]]:
        num_experts = self.num_local_experts
        prefix = f"model.layers.{index}"
        tensor_names = []
        for expert_idx in range(num_experts):
            for param in ("w1", "w2", "w3"):
                tensor_names.append(
                    prefix + f".block_sparse_moe.experts.{expert_idx}.{param}.weight"
                )
        tensor_names.append(prefix + ".block_sparse_moe.gate.weight")
        res = []
        for name in tensor_names:
            res.append(WeightInfo(name=name))
        for weight_info in MISTRAL_INFO.layer_weights(index, config):
            if ".mlp." in weight_info.name:
                continue
            res.append(weight_info)
        return res

    def sliceable(self) -> bool:
        return True

    def has_defined_spaces(self) -> bool:
        return False


def _load_json_arch(name: str) -> JsonArchitectureInfo:
    text = importlib.resources.read_text(mergekit._data.architectures, name)
    return JsonArchitectureInfo(
        definition=JSONArchitectureDefinition.model_validate_json(text)
    )


def _load_all_architectures() -> (
    Tuple[List[JsonArchitectureInfo], Dict[str, List[JsonArchitectureInfo]]]
):
    architectures: List[JsonArchitectureInfo] = []
    for f in importlib.resources.contents(mergekit._data.architectures):
        if f.lower().endswith(".json"):
            architectures.append(_load_json_arch(f))

    name_to_arch: Dict[str, List[JsonArchitectureInfo]] = {}
    for arch_info in architectures:
        for name in arch_info.definition.architectures:
            name_to_arch[name] = name_to_arch.get(name, [])
            name_to_arch[name].append(arch_info)
    return architectures, name_to_arch


JSON_ARCHITECTURES, NAME_TO_ARCH = _load_all_architectures()
MISTRAL_INFO = _load_json_arch("mistral.json")
QWEN2_INFO = _load_json_arch("qwen2.json")


class ArchitectureInfoUtils:
    """Functions for inferring architecture information from a merge configuration."""

    @staticmethod
    def get_architecture_info(config: PretrainedConfig) -> Optional[ArchitectureInfo]:
        """Get architecture info from an existing model config."""
        if len(config.architectures) != 1:
            raise RuntimeError("More than one architecture in config?")

        arch_name = config.architectures[0]

        if arch_name == MixtralTensorNames.ARCHITECTURE_NAME:
            return MixtralTensorNames.from_config(config)

        if arch_name in NAME_TO_ARCH:
            candidates = list(NAME_TO_ARCH[arch_name])
            if len(candidates) == 1:
                return candidates[0]

            for c in candidates:
                if c.definition.expected_model_type == config.model_type:
                    return c

        warnings.warn(f"No architecture config available for: {arch_name}.")
        return None

    @staticmethod
    def infer_architecture_info(merge_config) -> AutomaticArchitectureInfo:
        """
        Infer architecture info and prefixes for alignment.
        Prefixes typically denote where a model is used as a subcomponent of another model.
        e.g., [layer.0, layer.1, ...] and []'vision_tower.layer.0', vision_tower.layer.1', ...]
            inferring ßprefix = 'vision_tower' is required to align the two models.

        Usage:
            Similar to `get_architecture_info`, but requires a merge configuration object rather than a model config.
            This is so the common parameter names between all models can be inferred.
        """
        param_names = [
            ParameterNamesUtils.get_model_parameter_names(source_model.model.path)
            for source_model in merge_config.referenced_models()
        ]
        base_model = merge_config.base_model

        paired_list = list(zip(param_names, merge_config.referenced_models()))
        paired_list.sort(key=lambda x: len(x[0]), reverse=True)
        for i, (_, model_name) in enumerate(paired_list):
            if model_name == base_model:
                paired_list.insert(0, paired_list.pop(i))
                break
        param_names, referenced_models = zip(*paired_list)
        logging.info(f"Base model selected: {referenced_models[0].model.path}")

        prefixes = [""]
        for i in range(1, len(param_names)):
            assert len(param_names[0]) >= len(
                param_names[i]
            ), f"base model names list can't be shorter than model {i} names list"
            prefixes.append(
                ParameterNamesUtils.find_prefix(param_names[0], param_names[i])
            )

        common_names = ParameterNamesUtils.find_common_ordered_names(
            param_names, prefixes
        )

        common_names = ParameterNamesUtils.remove_size_conflicts(
            common_names, referenced_models, prefixes
        )

        ArchitectureInfoUtils.log_info(common_names, param_names, referenced_models)

        if not common_names or any([p is None for p in prefixes]):
            raise ValueError("Could not resolve model architecture automatically.")

        prefix_tracker = {
            model.model.path: f"{prefix}." if prefix else ""
            for model, prefix in zip(referenced_models, prefixes)
        }

        arch_name = referenced_models[0].model.path
        parameter_names = common_names

        return AutomaticArchitectureInfo(
            arch_name=arch_name,
            parameter_names=parameter_names,
            prefix_tracker=prefix_tracker,
            post_fill_parameters=(
                referenced_models[0].model.path  # base model name
                if len(common_names) != len(param_names[0])
                else None  # no post-fill needed
            ),
        )

    @staticmethod
    def log_info(common_names, param_names, referenced_models):
        for i in range(1, len(param_names)):
            prefix, case_message = ParameterNamesUtils.report_names_similarity(
                param_names[0], param_names[i]
            )
            logging.info(
                f"Model {referenced_models[i].model.path}: \
                    \n  {f'Best prefix found: {prefix}' if prefix else 'No prefix found'}\
                    \n  {case_message.replace('MODEL_ID', referenced_models[i].model.path)}"
            )

        if len(common_names) != len(param_names[0]):
            warnings.warn(
                f"Merging {len(common_names)}/{len(param_names[0])} base model parameters. \
                \n Base model selected: {referenced_models[0].model.path} \
                \n copy_and_fill_missing_params will run when merge is complete, to fill in missing params from base model."
            )

        if len(common_names) < 0.3 * len(param_names[0]):
            warnings.warn(
                "Not many common parameters found. Are you sure you are merging the correct models?"
            )


class ParameterNamesUtils:
    """Utility functions for handling parameter names."""

    @staticmethod
    def resolve_model_directory(repo_id: str) -> Path:
        """Resolve the model directory (local or Hugging Face Hub)."""
        if Path(repo_id).is_dir():
            return Path(repo_id)

        return Path(snapshot_download(repo_id))

    @staticmethod
    def get_model_parameter_names(repo_id: str) -> List[str]:
        """Get parameter names of a model from a Hugging Face repo or local directory."""
        model_dir = ParameterNamesUtils.resolve_model_directory(repo_id)
        return list(ShardedTensorIndex.from_disk(str(model_dir)).tensor_paths.keys())

    @staticmethod
    def strip_prefix(name: str, prefix: str) -> str:
        """Remove a single prefix from the start of a name."""
        if prefix != "" and name.startswith(prefix + "."):
            return name[len(prefix) + 1 :]
        return name

    @staticmethod
    def find_prefix(list1: List[str], list2: List[str]) -> Optional[str]:
        """
        Find a prefix in list1 that, after removal, makes list2 an ordered sublist.
        """
        assert len(list1) >= len(list2), "params name list1 can't be shorter than list2"

        possible_prefixes = {item.split(".")[0] for item in list1 if "." in item}
        possible_prefixes = [""] + list(possible_prefixes)

        prefix_matches = {}
        best_prefix = ""  # Default to no prefix
        for prefix in possible_prefixes:
            stripped_list1 = [
                ParameterNamesUtils.strip_prefix(item, prefix) for item in list1
            ]
            prefix_matches[prefix] = len(
                [item for item in list2 if item in stripped_list1]
            )

        if max(prefix_matches.values()) > prefix_matches[""]:
            best_prefix = max(prefix_matches, key=prefix_matches.get)

        return best_prefix

    @staticmethod
    def find_common_ordered_names(
        param_names: List[List[str]], prefixes: List[str]
    ) -> List[str]:
        """Identify and return common parameter names across all models, ensuring correct order. Also account for prefix."""
        common_names = set(param_names[0])
        for i in range(1, len(param_names)):
            prefix = f"{prefixes[i]}." if prefixes[i] else ""
            common_names.intersection_update({prefix + name for name in param_names[i]})
        return [name for name in param_names[0] if name in common_names]

    @staticmethod
    def remove_size_conflicts(common_names, referenced_models, prefixes):
        model_dirs = [
            ParameterNamesUtils.resolve_model_directory(m.model.path)
            for m in referenced_models
        ]
        model_indices = [ShardedTensorIndex.from_disk(str(dir)) for dir in model_dirs]

        common_name_and_shape = common_names.copy()
        removed_names = []

        for name in common_names:
            base_shape = ParameterNamesUtils.tensor_shape(name, model_indices[0])

            for i in range(1, len(referenced_models)):
                other_name = name
                prefix = f"{prefixes[i]}." if prefixes[i] else ""
                if name.startswith(prefix) and prefix != "":
                    other_name = name[len(prefix) :]
                shape = ParameterNamesUtils.tensor_shape(other_name, model_indices[i])

                if base_shape != shape:
                    common_name_and_shape.remove(name)
                    removed_names.append((name, base_shape, shape, i))
                    break

        size_mismatch_count = len(removed_names)
        if size_mismatch_count > 0:
            logging.warning(
                f"Size mismatch detected for {size_mismatch_count}/{size_mismatch_count + len(common_names)} tensors. "
                "These names were removed from the merge list."
            )
            logging.info(
                "The following tensors have different shapes across models and were removed from the merge list:"
            )
            for name, base_shape, shape, i in removed_names:
                logging.info(
                    f"Tensor name: {name}, Base model shape: {base_shape}, Mismatched shape: {shape} in model {referenced_models[i].model.path}"
                )

        return common_name_and_shape

    @staticmethod
    def are_common_params_ordered(list1: List[str], list2: List[str]) -> bool:
        """
        Check if common elements of list2 maintain their relative order in list1.
        """
        common_params = set(list1).intersection(set(list2))
        last_index = -1

        for param in list2:
            if param in common_params:
                current_index = list1.index(param)
                if current_index < last_index:
                    return False
                last_index = current_index
        return True

    @staticmethod
    def ordered_sublist(list1: List[str], list2: List[str]) -> bool:
        """
        Check if list2 is a contiguous ordered sublist of list1.
        """
        n, m = len(list1), len(list2)

        for i in range(n - m + 1):
            if list1[i : i + m] == list2:
                return True
        return False

    @staticmethod
    def report_names_similarity(
        base_names: List[str], other_names: List[str]
    ) -> Tuple[Optional[str], str]:
        """
        Analyze similarity between parameter names of two models and identify shared prefixes.

        Returns:
            best_prefix (str): Best matching prefix for parameter names.
            case_message (str): Explanation of the structural relationship.
        """
        possible_prefixes = {""}
        possible_prefixes.update(
            {item.split(".")[0] for item in base_names if "." in item}
        )

        prefixes_subset_overlap = {}
        best_prefix = None
        case_message = "No common parameter names found for any prefix"

        for prefix in possible_prefixes:
            base_names_stripped = [
                ParameterNamesUtils.strip_prefix(name, prefix) for name in base_names
            ]

            if ParameterNamesUtils.ordered_sublist(base_names_stripped, other_names):
                return prefix, "All params in model have exact match in base model."

            intersection = set(base_names_stripped).intersection(set(other_names))
            prefixes_subset_overlap[prefix] = intersection

        if prefixes_subset_overlap:
            best_prefix = max(
                prefixes_subset_overlap, key=lambda x: len(prefixes_subset_overlap[x])
            )
            base_names_stripped = [
                ParameterNamesUtils.strip_prefix(name, best_prefix)
                for name in base_names
            ]

            overlap = len(prefixes_subset_overlap[best_prefix])
            ordered = ParameterNamesUtils.are_common_params_ordered(
                base_names_stripped, other_names
            )
            mismatched = [
                item for item in other_names if item not in base_names_stripped
            ]
            mismatched = "\n    ".join(mismatched)
            case_message = (
                f"{overlap}/{len(other_names)} ({100 * overlap / len(other_names):.2f}%) "
                f"of model parameters are in the base model. \n"
                f"  Name ordering is {'preserved' if ordered else 'not preserved'}.\n"
                f"  Missing parameters:\n    {mismatched}"
            )

        return best_prefix, case_message

    @staticmethod
    def tensor_shape(name, index) -> Tuple[int]:
        from safetensors import safe_open

        with safe_open(
            Path(index.base_path) / index.tensor_paths[name], framework="pt"
        ) as f:
            return f.get_slice(name).get_shape()
