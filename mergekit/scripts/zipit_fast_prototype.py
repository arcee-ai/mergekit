import logging
from typing import DefaultDict, Dict, Enum, List, Optional, Set, Tuple

import click
import numpy as np
import torch
import tqdm

from mergekit.architecture import ProceduralSpaceInfo, WeightInfo, get_architecture_info
from mergekit.common import ModelReference, dtype_from_name
from mergekit.io.tasks import LoaderCache
from mergekit.io.tensor_writer import TensorWriter
from mergekit.options import MergeOptions, add_merge_options


# unmerge -> direction difference
# merge -> same direction terminal end
class ZipType(Enum):
    MERGE = 1
    UNMERGE = 2


class Node:
    pseudo_node: bool
    name: str
    color: Optional[str]
    zip_type: ZipType = ZipType.UNMERGE

    def __init__(self, name: str, pseudo_node: bool = False):
        self.name = name
        self.pseudo_node = pseudo_node

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


# TODO: revisit Node definition after subgraphs construction part
class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = {}

    def add_node(self, node: str):
        self.nodes.add(node)
        self.edges[node] = []

    def add_edge(self, node1, node2):
        self.edges[node1].append(node2)

    def successors(self, node):
        return self.edges[node]

    def predecessors(self, node):
        return [n for n in self.nodes if node in self.edges[n]]

    def add_color(self, color):
        for node in self.nodes:
            node.color = color

    # overload addition operator to merge two graphs
    # TODO check for correctness
    def __add__(self, other):
        new_graph = Graph()
        new_graph.nodes = self.nodes.union(other.nodes)
        new_graph.edges = {**self.edges, **other.edges}
        return new_graph


# NOTE: why bother with a subgraph?
# I believe the original code  is constructing a subgraph
# to carve out a subgraph is a step towards the code becoming declarative rather than purely procedural
# also I imagine this would be a nice launchpad for constructing tasks
def SubGraphConstructor():
    def __init__(self, all_weights_info, procs_spaces):
        self.space_in_tensors = DefaultDict[str, List[WeightInfo]](list)
        self.space_out_tensors = DefaultDict[str, List[WeightInfo]](list)

        for weight_info in all_weights_info:
            if weight_info.input_space:
                self.space_in_tensors[weight_info.input_space].append(weight_info)
            else:
                logging.warning(f"Weight {weight_info.name} has no input space")

            if weight_info.output_space:
                self.space_out_tensors[weight_info.output_space].append(weight_info)
            elif not weight_info.is_vector:
                logging.warning(f"Weight {weight_info.name} has no output space")

        res_refs: DefaultDict[str, List[ProceduralSpaceInfo]] = DefaultDict(list)
        for ps in procs_spaces:
            for i in ps.inputs:
                res_refs[i].append(ps.name)

        reverse_res_refs: DefaultDict[str, List[str]] = DefaultDict(list)
        for k, v in res_refs.items():
            for ps in v:
                reverse_res_refs[ps].append(k)

    def construct_subgraph(self, n):
        # pseudo
        # what is the stopping condition ?
        subgraph = Graph()

        if isinstance(n, str):
            subgraph.add_node(n, pseudo_node=True)
        else:
            subgraph.add_node(n.name, pseudo_node=False)

        if n in self.res_refs:
            # get all the nodes that are connected to this node
            for input_edge in self.res_refs[n]:
                for node in self.space_out_tensors[input_edge]:
                    subgraph.add_node(node)
                    subgraph.add_edge(n, node)
                    subgraph += construct_subgraph(node)

        if not isinstance(n, str):
            for output_edge in n.output_space:
                for node in self.space_in_tensors[output_edge]:
                    # going upstream
                    subgraph.add_node(node, zip_type=ZipType.UNMERGE)
                    subgraph.add_edge(node, n)
                    if node.is_vector:
                        subgraph += construct_subgraph(node)

            for input_edge in n.input_space:
                for node in self.space_out_tensors[input_edge]:
                    subgraph.add_node(node)
                    subgraph.add_edge(n, node)
                    if node.is_vector:
                        subgraph += construct_subgraph(node)

        return subgraph


@click.command("zipit")
@click.argument("model_path", type=str)
@click.argument("secondary_model_path", type=str)
@click.argument("activations", type=str)
@click.option("--out-path", "-o", required=True, type=str, help="Output model path")
@click.option(
    "--dtype",
    type=str,
    default=None,
    help="Data type to convert weights to",
)
@add_merge_options
def main(
    model_path: str,
    secondary_model_path,
    merge_unmerge_dictionary: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    out_path: str,
    merge_options: MergeOptions,
):
    model = ModelReference.model_validate(model_path)
    secondary_model = ModelReference.model_validate(merge_options.secondary_model_path)

    cache = LoaderCache()
    cache.lazy_unpickle = merge_options.lazy_unpickle
    cache.lora_cache_dir = merge_options.lora_merge_cache
    cache.hf_cache_dir = merge_options.transformers_cache

    for m in tqdm.tqdm([model, secondary_model_path], desc="Preparing models"):
        cache.get(m)

    model_config = model.config(trust_remote_code=merge_options.trust_remote_code)
    model_arch_info = get_architecture_info(
        model.config(trust_remote_code=merge_options.trust_remote_code)
    )
    if not model_arch_info.has_defined_spaces():
        raise RuntimeError(f"Model {model} does not have defined spaces - cannot align")

    # two pass model merging
    # assume merge/unmerge is calculated elsewhere and you are given it

    # create structure for the graph and using keys in the M/U dict create subgraphs and create tasks at each of these nodes
    all_weights = model_arch_info.all_weights(config=model_config)

    for named_res_conn, (m, u) in merge_unmerge_dictionary.items():
        # recursively make directed subgraph starting from named_res_conn
        # and ends in nodes that have rectangular weights
        subgraph = Graph()
