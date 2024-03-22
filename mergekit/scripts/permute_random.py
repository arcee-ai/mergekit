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

from typing import Optional

import click
import torch

from mergekit.common import ModelReference
from mergekit.options import MergeOptions, add_merge_options
from mergekit.permuter import ModelPermuter


def rand_perm_mat(n: int) -> torch.Tensor:
    P = torch.zeros(n, n)
    for i, j in enumerate(torch.randperm(n)):
        P[i, j] = 1
    return P


def rand_rope_rotations(n_heads: int, head_dim: int) -> torch.Tensor:
    theta = torch.randn(n_heads, head_dim // 2)
    theta_p = torch.cat([theta, theta], dim=-1)
    cos_theta = torch.cos(theta_p)
    sin_theta = torch.sin(theta_p)

    P = torch.zeros(n_heads, head_dim, head_dim)
    idx = torch.arange(head_dim // 2)
    P[:, idx, idx] = cos_theta[:, idx]
    P[:, idx, head_dim // 2 + idx] = sin_theta[:, idx]
    P[:, head_dim // 2 + idx, idx] = -sin_theta[:, idx]
    P[:, head_dim // 2 + idx, head_dim // 2 + idx] = cos_theta[:, idx]
    return P


@click.command("mergekit-permute-random")
@click.argument("model_path", type=str)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option(
    "--dtype",
    type=str,
    default=None,
    help="Data type to convert weights to",
)
@click.option(
    "--space",
    type=str,
    default=None,
    help="Space to permute",
)
@click.option(
    "--permute-head-dims/--no-permute-head-dims",
    default=True,
    help="Permute head dimensions",
)
@add_merge_options
def main(
    model_path: str,
    out_path: str,
    dtype: Optional[str],
    space: Optional[str],
    permute_head_dims: bool,
    merge_options: MergeOptions,
):
    chosen_space = space
    model = ModelReference.model_validate(model_path)
    permuter = ModelPermuter(model=model, options=merge_options, dtype=dtype)

    num_heads = permuter.model_config.num_attention_heads

    head_permutations = {}
    for head_group in permuter.head_group_weights.keys():
        P = torch.eye(num_heads, num_heads)
        if chosen_space == f"head:{head_group}" or not chosen_space:
            P = rand_perm_mat(num_heads)
        head_permutations[head_group] = P

    for space in permuter.all_spaces:
        tensor_info = permuter.space_tensors(
            model, space, transform_in=False, transform_out=False
        )
        if not tensor_info:
            # must be a residual space
            continue

        tensors = []
        head_group = None
        rope = None
        for weight_info in sorted(tensor_info.keys(), key=lambda x: x.name):
            tensor = tensor_info[weight_info]
            tensors.append(tensor)

            if rope is None:
                rope = weight_info.rope
            if rope != weight_info.rope:
                raise RuntimeError(
                    f"Space {space} has tensors with different RoPE status"
                )
            if weight_info.head_group:
                if head_group is not None and head_group != weight_info.head_group:
                    raise RuntimeError(
                        f"Space {space} has tensors from different head groups"
                    )
                head_group = weight_info.head_group

        out_dim = tensors[0].shape[0]
        if not all(t.shape[0] == out_dim for t in tensors):
            print([t.shape for t in tensors])
            raise RuntimeError(f"Space {space} has tensors of different sizes")

        if head_group:
            P_head_dim = (
                torch.eye(permuter.head_dim, permuter.head_dim)
                .unsqueeze(0)
                .repeat(num_heads, 1, 1)
            )
            if permute_head_dims:
                if rope:
                    P_head_dim = rand_rope_rotations(num_heads, permuter.head_dim)
                else:
                    for i in range(num_heads):
                        P_head_dim[i] = rand_perm_mat(permuter.head_dim)

            print(f"{space}: RoPE = {rope}, head_group = {head_group}")
            P_heads = head_permutations[head_group]
            P = torch.zeros(
                out_dim, out_dim, device=tensors[0].device, dtype=tensors[0].dtype
            )
            for head_idx in range(num_heads):
                new_head_idx = torch.argmax(P_heads[head_idx])
                P[
                    head_idx * permuter.head_dim : (head_idx + 1) * permuter.head_dim,
                    new_head_idx
                    * permuter.head_dim : (new_head_idx + 1)
                    * permuter.head_dim,
                ] = P_head_dim[head_idx]
            print(f"{space}: {P.max(dim=-1).values}")
        else:
            P = torch.eye(
                out_dim, out_dim, device=tensors[0].device, dtype=tensors[0].dtype
            )
            if space == chosen_space or not chosen_space:
                P = rand_perm_mat(out_dim).to(tensors[0].device, tensors[0].dtype)
        permuter.set_transform(space, P, P.t())

    permuter.update_all_proc_spaces()
    permuter.write_permuted_model(out_path)


if __name__ == "__main__":
    with torch.no_grad():
        main()
