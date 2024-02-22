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
from typing import DefaultDict, Dict, List, Optional, Set

import click
import numpy as np
import ot
import torch
import tqdm
from scipy.optimize import linear_sum_assignment

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo, get_architecture_info
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options


@click.command("mergekit-align-model")
@click.argument("model_path", type=str)
@click.option(
    "--target", "-t", required=True, type=str, help="Target model to align weights to"
)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option("--iters", "-i", type=int, default=10, help="Number of iterations")
@click.option(
    "--sinkhorn/--no-sinkhorn",
    "-s",
    type=bool,
    default=False,
    help="Use Sinkhorn algorithm",
)
@click.option(
    "--sinkhorn-reg",
    type=float,
    default=0.05,
    help="Regularization for Sinkhorn algorithm",
)
@click.option(
    "--dtype",
    type=str,
    default=None,
    help="Data type to convert weights to",
)
@add_merge_options
def main(
    model_path: str,
    out_path: str,
    target: str,
    iters: int,
    sinkhorn: bool,
    sinkhorn_reg: float,
    dtype: Optional[str],
    merge_options: MergeOptions,
):
    model = ModelReference.model_validate(model_path)
    target_model = ModelReference.model_validate(target)

    cache = LoaderCache()
    cache.lazy_unpickle = merge_options.lazy_unpickle
    cache.lora_cache_dir = merge_options.lora_merge_cache
    cache.hf_cache_dir = merge_options.transformers_cache

    for m in tqdm.tqdm([model, target_model], desc="Preparing models"):
        cache.get(m)

    model_config = model.config(trust_remote_code=merge_options.trust_remote_code)
    model_arch_info = get_architecture_info(
        model.config(trust_remote_code=merge_options.trust_remote_code)
    )
    if not model_arch_info.has_defined_spaces():
        raise RuntimeError(f"Model {model} does not have defined spaces - cannot align")

    space_in_tensors = DefaultDict[str, List[WeightInfo]](list)
    space_out_tensors = DefaultDict[str, List[WeightInfo]](list)
    all_spaces = []
    for weight_info in model_arch_info.all_weights(config=model_config):
        if weight_info.input_space:
            space_in_tensors[weight_info.input_space].append(weight_info)
            if weight_info.input_space not in all_spaces:
                all_spaces.append(weight_info.input_space)
        else:
            logging.warning(f"Weight {weight_info.name} has no input space")

        if weight_info.output_space:
            space_out_tensors[weight_info.output_space].append(weight_info)
            if weight_info.output_space not in all_spaces:
                all_spaces.append(weight_info.output_space)
        elif not weight_info.is_vector:
            logging.warning(f"Weight {weight_info.name} has no output space")

    space_proc_refs: DefaultDict[str, List[ProceduralSpaceInfo]] = DefaultDict(list)
    for ps in model_arch_info.procedural_spaces(config=model_config):
        for s in ps.inputs:
            space_proc_refs[s].append(ps)

    transforms: Dict[str, torch.Tensor] = {}
    transforms_inv: Dict[str, torch.Tensor] = {}

    dtype = dtype_from_name(dtype) if dtype else None

    for space in all_spaces:
        if space not in space_out_tensors:
            logging.warning(f"Space {space} has no output tensors")
        if space not in space_in_tensors:
            logging.warning(f"Space {space} has no input tensors")

    hidden_size = model_config.hidden_size
    num_attention_heads = model_config.num_attention_heads
    head_dim = hidden_size // num_attention_heads

    def _get_weight(
        wi: WeightInfo,
        model: ModelReference,
        transform_in: bool = True,
        transform_out: bool = True,
        verbose: bool = False,
        transpose_embed: bool = True,
    ):
        loader = cache.get(model)
        if wi.name == "lm_head.weight" and wi.name not in loader.index.tensor_paths:
            wi = WeightInfo(
                name="model.embed_tokens.weight", **wi.model_dump(exclude=["name"])
            )
        tensor = loader.get_tensor(wi.name, device="cuda")
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)

        if wi.name in ("lm_head.weight", "model.embed_tokens.weight"):
            tensor = tensor[:32000, :]

        if verbose:
            logging.warning(f"{wi.name}: {wi.input_space} -> {wi.output_space}")

        if wi.is_vector:
            if transform_in and wi.input_space:
                tensor = transforms[wi.input_space] @ tensor
        else:
            if wi.is_embed:
                # nn.Embedding stores the embedding matrix as (vocab_size, embed_dim)
                # but we want to treat it as (embed_dim, vocab_size) for the purposes of
                # the permutations
                tensor = tensor.T

            if wi.head_split:
                # split input or output into attention heads and then permute to (head_dim, output_dim, input_dim)
                if wi.head_split == "input":
                    tensor = tensor.view(tensor.shape[0], -1, head_dim).permute(2, 0, 1)
                    # (head_dim, output_dim, num_heads)
                elif wi.head_split == "output":
                    tensor = tensor.view(-1, head_dim, tensor.shape[-1]).permute(
                        1, 0, 2
                    )
                    # (head_dim, num_heads, input_dim)

            if transform_in and wi.input_space and wi.input_space in transforms_inv:
                tensor = tensor @ transforms_inv[wi.input_space]

            if transform_out and wi.output_space and wi.output_space in transforms:
                tensor = transforms[wi.output_space] @ tensor

            if transform_out and wi.head_split:
                if wi.head_split == "input":
                    tensor = tensor.permute(1, 2, 0).reshape(tensor.shape[1], -1)
                elif wi.head_split == "output":
                    tensor = tensor.permute(1, 0, 2).reshape(-1, tensor.shape[-1])

            if wi.is_embed and not transpose_embed:
                # undo the transpose we did earlier
                tensor = tensor.T
        return tensor

    def _space_tensors(
        model: ModelReference,
        space: str,
        transform_in: bool = True,
        transform_out: bool = True,
        transpose_embed: bool = True,
    ):
        res: Dict[WeightInfo, torch.Tensor] = {}
        for weight_info in space_out_tensors.get(space, []):
            res[weight_info] = _get_weight(
                weight_info,
                model,
                transform_in=transform_in,
                transform_out=transform_out,
                transpose_embed=transpose_embed,
            )

        tensors = []
        for weight_info in sorted(res.keys(), key=lambda x: x.name):
            tensor = res[weight_info]
            if len(tensor.shape) > 2:
                tensors.extend([t.squeeze(0) for t in tensor.split(1, dim=0)])
            else:
                tensors.append(tensor)
        return tensors

    def _update_proc(
        space: ProceduralSpaceInfo, touched: Optional[Set[ProceduralSpaceInfo]] = None
    ):
        if touched is None:
            touched = set()

        if space.type == "residual":
            # as in Transformer Fusion with Optimal Transport, average the input transforms
            tfs = [transforms.get(s, None) for s in space.inputs]
            if not all(t is not None for t in tfs):
                return
            new_transform = sum(tfs) / max(len(tfs), 1)

        elif space.type == "kronecker":
            # kronecker product of input transforms
            tfs = [transforms.get(s, None) for s in space.inputs]
            if not all(t is not None for t in tfs):
                return
            new_transform = torch.kron(*tfs)

        else:
            raise RuntimeError(f"Unsupported procedural space type {space.type}")

        if (space.name not in transforms) or not torch.allclose(
            new_transform, transforms[space.name]
        ):
            logging.warning(
                f"Updating procedural space {space.name} of type {space.type}"
            )
            transforms[space.name] = new_transform
            transforms_inv[space.name] = torch.linalg.pinv(
                new_transform.float(), rtol=1e-5
            ).to(new_transform.dtype)
            if transforms_inv[space.name].isinf().any():
                logging.warning(f"Space {space.name} has inf in pinverse")
            if transforms_inv[space.name].isnan().any():
                logging.warning(f"Space {space.name} has nan in pinverse")
            for other in space_proc_refs[space.name]:
                if other not in touched:
                    touched.add(other)
                    _update_proc(other, touched=touched)

    for iter_idx in tqdm.tqdm(range(iters), desc="Iterating"):
        change_count = 0
        perm = np.random.permutation(all_spaces) if iter_idx > 0 else list(all_spaces)
        for space in tqdm.tqdm(
            perm,
            leave=False,
            total=len(all_spaces),
            desc="Aligning spaces",
        ):
            if space not in space_in_tensors and space not in space_out_tensors:
                continue

            in_tensors = _space_tensors(
                model, space, transform_in=True, transform_out=False
            )
            if not in_tensors:
                continue
            target_tensors = _space_tensors(
                target_model, space, transform_in=False, transform_out=False
            )

            assert len(in_tensors) == len(target_tensors)

            out_dim = target_tensors[0].shape[0]
            if not all(x.shape[0] == out_dim for x in target_tensors):
                logging.error(f"Output dimension mismatch for space {space}")
                logging.error(f"Target: {target_tensors[0].shape[0]}")
                logging.error(f"Found shapes: {[x.shape for x in target_tensors]}")
                logging.error(
                    f"Weight names: {[x.name for x in space_out_tensors[space]]}"
                )
                raise RuntimeError(f"Output dimension mismatch for space {space}")

            work_dtype = (
                torch.float32
                if target_tensors[0].device.type == "cpu"
                else target_tensors[0].dtype
            )

            if sinkhorn:
                honk_in = torch.cat(in_tensors, dim=1)
                honk_target = torch.cat(target_tensors, dim=1)
                cost_mat = ot.dist(honk_in, honk_target, metric="sqeuclidean")

                mass = (
                    torch.ones(out_dim, device=cost_mat.device, dtype=cost_mat.dtype)
                    / out_dim
                )
                model_to_base, log = ot.sinkhorn(
                    a=mass,
                    b=mass,
                    M=cost_mat,
                    reg=sinkhorn_reg,
                    stopThr=1e-6,
                    verbose=False,
                    log=True,
                    method="sinkhorn_log",
                )
                model_to_base *= out_dim
                if log["err"] and log["err"][-1] > 1e-4:
                    logging.warning(f"Space {space}:")
                    logging.warning(f'niter: {log["niter"]}')
                    logging.warning("err: " + str(log["err"][-1]))
            else:
                cost_mat = torch.zeros(
                    out_dim, out_dim, device=target_tensors[0].device, dtype=work_dtype
                )
                for x_target, x_model in zip(target_tensors, in_tensors):
                    cost_mat += x_target.to(work_dtype) @ x_model.T.to(work_dtype)

                ri, ci = linear_sum_assignment(cost_mat.cpu().numpy(), maximize=True)
                model_to_base = torch.zeros_like(
                    cost_mat, dtype=target_tensors[0].dtype
                )
                model_to_base[(ri, ci)] = 1

            old_transform = transforms.get(space, None)
            transforms[space] = model_to_base
            transforms_inv[space] = model_to_base.T

            if old_transform is None or not torch.allclose(
                old_transform, model_to_base
            ):
                if not torch.allclose(
                    model_to_base,
                    torch.eye(
                        out_dim, device=model_to_base.device, dtype=model_to_base.dtype
                    ),
                ):
                    change_count += 1
                for proc in space_proc_refs[space]:
                    _update_proc(proc)
        logging.warning(f"Iteration {iter_idx}: {change_count} changes")

    for space_name in all_spaces:
        if space_name not in transforms:
            logging.warning(f"Space {space_name} not transformed")

    # write aligned model
    writer = TensorWriter(
        out_path=out_path,
        max_shard_size=merge_options.out_shard_size,
        safe_serialization=merge_options.safe_serialization,
    )
    for weight_info in model_arch_info.all_weights(config=model_config):
        tensor = _get_weight(
            weight_info,
            model,
            transform_in=True,
            transform_out=True,
            verbose=False,
            transpose_embed=False,
        ).cpu()
        writer.save_tensor(weight_info.name, tensor, clone=merge_options.clone_tensors)
    writer.finalize()
    model_config.save_pretrained(out_path)


if __name__ == "__main__":
    with torch.no_grad():
        main()
