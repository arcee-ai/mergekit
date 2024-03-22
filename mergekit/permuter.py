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

import functools
import logging
from typing import DefaultDict, Dict, List, Optional, Set

import torch
from transformers import PretrainedConfig

from mergekit.architecture import (
    ArchitectureInfo,
    ProceduralSpaceInfo,
    WeightInfo,
    get_architecture_info,
)
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io import TensorWriter
from mergekit.io.tasks import LoaderCache
from mergekit.options import MergeOptions


class ModelPermuter:
    model: ModelReference
    transforms: Dict[str, torch.Tensor]
    transforms_inv: Dict[str, torch.Tensor]
    space_in_tensors: DefaultDict[str, List[WeightInfo]]
    space_out_tensors: DefaultDict[str, List[WeightInfo]]
    space_proc_refs: DefaultDict[str, List[ProceduralSpaceInfo]]
    head_group_weights: DefaultDict[str, List[WeightInfo]]
    all_spaces: List[str]
    dtype: Optional[torch.dtype]
    loader_cache: LoaderCache
    model_config: PretrainedConfig
    model_arch_info: ArchitectureInfo

    @functools.lru_cache()
    def space_dimension(self, space: str, transform: bool = False) -> int:
        res = None
        for tensor_info, x in self.space_tensors(
            self.model, space=space, transform_in=transform, transform_out=transform
        ).items():
            if res is not None and x.shape[0] != res:
                raise RuntimeError(
                    f"Space {space} has tensors of different sizes: {tensor_info.name} has {x.shape[0]} but expected {res}"
                )
            res = x.shape[0]
        return res

    def __init__(
        self,
        model: ModelReference,
        options: MergeOptions,
        dtype: Optional[str] = None,
    ):
        self.model = model
        self.transforms = {}
        self.transforms_inv = {}
        self.space_in_tensors = DefaultDict(list)
        self.space_out_tensors = DefaultDict(list)
        self.space_proc_refs = DefaultDict(list)
        self.head_group_weights = DefaultDict(list)
        self.all_spaces = []
        self.dtype = dtype_from_name(dtype) if dtype else None

        self.merge_options = options

        self.loader_cache = LoaderCache()
        self.loader_cache.lazy_unpickle = options.lazy_unpickle
        self.loader_cache.lora_cache_dir = options.lora_merge_cache
        self.loader_cache.hf_cache_dir = options.transformers_cache

        self.loader_cache.get(model)

        self.model_config = model.config(trust_remote_code=options.trust_remote_code)
        self.model_arch_info = get_architecture_info(self.model_config)
        if not self.model_arch_info.has_defined_spaces():
            raise RuntimeError(
                f"Model {model} does not have defined spaces - cannot align"
            )

        self.head_dim = (
            self.model_config.hidden_size // self.model_config.num_attention_heads
        )

        for weight_info in self.model_arch_info.all_weights(config=self.model_config):
            if weight_info.input_space:
                self.space_in_tensors[weight_info.input_space].append(weight_info)
                if weight_info.input_space not in self.all_spaces:
                    self.all_spaces.append(weight_info.input_space)
            else:
                logging.warning(f"Weight {weight_info.name} has no input space")

            if weight_info.output_space:
                self.space_out_tensors[weight_info.output_space].append(weight_info)
                if weight_info.output_space not in self.all_spaces:
                    self.all_spaces.append(weight_info.output_space)
            elif not weight_info.is_vector:
                logging.warning(f"Weight {weight_info.name} has no output space")

            if weight_info.head_group is not None:
                self.head_group_weights[weight_info.head_group].append(weight_info)

        for ps in self.model_arch_info.procedural_spaces(config=self.model_config):
            self.all_spaces.append(ps.name)
            for s in ps.inputs:
                self.space_proc_refs[s].append(ps)

        self.all_spaces = list(set(self.all_spaces))

    def get_transform(
        self, space: Optional[str], inverse: bool = False, hard: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Retrieves the transformation matrix for the specified space.

        Args:
            space (str): The name of the space.
            inverse (bool, optional): If True, retrieves the inverse transformation matrix.
            hard (bool, optional): If True, converts transport maps to permutation matrices.
        """
        if not space:
            return None

        tfs = self.transforms_inv if inverse else self.transforms
        if space not in tfs:
            return None

        res = tfs[space]
        if hard:
            # convert soft transport map to permutation matrix
            max_indices = res.argmax(dim=1)
            res = torch.zeros_like(res)
            res[torch.arange(res.shape[0]), max_indices] = 1

        return res

    def get_weight(
        self,
        wi: WeightInfo,
        model: ModelReference,
        transform_in: bool = True,
        transform_out: bool = True,
        verbose: bool = False,
        hard_perm: bool = False,
    ):
        loader = self.loader_cache.get(model)

        # HACK: should use WeightInfo.aliases instead of hard coding this
        if wi.name == "lm_head.weight" and wi.name not in loader.index.tensor_paths:
            wi = WeightInfo(
                name="model.embed_tokens.weight", **wi.model_dump(exclude=["name"])
            )

        tensor = loader.get_tensor(wi.name, device="cuda")
        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)

        # HACK: cut vocab down to 32000 assuming llama or mistral
        if wi.name in ("lm_head.weight", "model.embed_tokens.weight"):
            tensor = tensor[:32000, :]

        if verbose:
            logging.warning(f"{wi.name}: {wi.input_space} -> {wi.output_space}")

        if wi.is_vector:
            if (
                transform_in
                and (
                    M := self.get_transform(
                        wi.input_space, inverse=False, hard=hard_perm
                    )
                )
                is not None
            ):
                tensor = M.to(device=tensor.device) @ tensor
        else:
            M_in = self.get_transform(wi.input_space, inverse=True, hard=hard_perm)
            M_out = self.get_transform(wi.output_space, inverse=False, hard=hard_perm)

            # print(f"{wi.name}: {wi.input_space} -> {wi.output_space}")
            # print(f"M_in: {M_in.shape if M_in is not None else None}")
            # print(f"tensor: {tensor.shape}")
            # print(f"M_out: {M_out.shape if M_out is not None else None}")
            if wi.is_embed:
                # nn.Embedding stores the embedding matrix as (vocab_size, embed_dim)
                # but we want to treat it as (embed_dim, vocab_size) for the purposes of
                # the permutations
                tensor = tensor.T

            if transform_in and M_in is not None:
                tensor = tensor @ M_in.to(device=tensor.device)

            if transform_out and M_out is not None:
                tensor = M_out.to(device=tensor.device) @ tensor

            if transform_out and wi.is_embed:
                # undo the transpose we did earlier
                tensor = tensor.T
        return tensor

    def space_tensors(
        self,
        model: ModelReference,
        space: str,
        transform_in: bool = True,
        transform_out: bool = True,
        hard_perm: bool = False,
    ) -> Dict[WeightInfo, torch.Tensor]:
        res: Dict[WeightInfo, torch.Tensor] = {}
        tensors = self.space_out_tensors.get(space, [])
        if space.startswith("head:"):
            tensors = self.head_group_weights.get(space[len("head:") :], [])

        for weight_info in tensors:
            res[weight_info] = self.get_weight(
                weight_info,
                model,
                transform_in=transform_in,
                transform_out=transform_out,
                hard_perm=hard_perm,
            )

        return res

    def update_proc_space(
        self,
        space: ProceduralSpaceInfo,
        touched: Optional[Set[ProceduralSpaceInfo]] = None,
    ):
        if touched is None:
            touched = set()

        if space.type == "residual":
            # as in Transformer Fusion with Optimal Transport, average the input transforms
            tfs = [self.transforms.get(s, None) for s in space.inputs]
            if not all(t is not None for t in tfs):
                return
            new_transform = sum(tfs) / max(len(tfs), 1)

        elif space.type == "kronecker":
            # kronecker product of input transforms
            tfs = [self.transforms.get(s, None) for s in space.inputs]
            if not all(t is not None for t in tfs):
                return
            new_transform = torch.kron(*tfs)

        else:
            raise RuntimeError(f"Unsupported procedural space type {space.type}")

        if (space.name not in self.transforms) or not torch.allclose(
            new_transform, self.transforms[space.name]
        ):
            logging.warning(
                f"Updating procedural space {space.name} of type {space.type}"
            )
            self.transforms[space.name] = new_transform
            self.transforms_inv[space.name] = torch.linalg.pinv(
                new_transform.float(), rtol=1e-5
            ).to(new_transform.dtype)
            if self.transforms_inv[space.name].isinf().any():
                logging.warning(f"Space {space.name} has inf in pinverse")
            if self.transforms_inv[space.name].isnan().any():
                logging.warning(f"Space {space.name} has nan in pinverse")
            for other in self.space_proc_refs[space.name]:
                if other not in touched:
                    touched.add(other)
                    self.update_proc_space(other, touched=touched)

    def update_all_proc_spaces(self):
        for ps in self.model_arch_info.procedural_spaces(config=self.model_config):
            self.update_proc_space(ps)

    def set_transform(
        self,
        space: str,
        transform: torch.Tensor,
        inverse: Optional[torch.Tensor] = None,
        compute_inverse: bool = True,
    ):
        self.transforms[space] = transform
        if compute_inverse and inverse is None:
            inverse = torch.linalg.pinv(transform.float(), rtol=1e-5).to(
                transform.dtype
            )

        if inverse is not None:
            self.transforms_inv[space] = inverse

    def write_permuted_model(self, out_path: str, hard_perm: bool = False):
        writer = TensorWriter(
            out_path=out_path,
            max_shard_size=self.merge_options.out_shard_size,
            safe_serialization=self.merge_options.safe_serialization,
        )
        for weight_info in self.model_arch_info.all_weights(config=self.model_config):
            tensor = self.get_weight(
                weight_info,
                self.model,
                transform_in=True,
                transform_out=True,
                verbose=False,
                hard_perm=hard_perm,
            ).cpu()
            writer.save_tensor(
                weight_info.name, tensor, clone=self.merge_options.clone_tensors
            )
        writer.finalize()

        # HACK: force vocab size to 32000
        self.model_config.vocab_size = 32000

        self.model_config.save_pretrained(out_path)
