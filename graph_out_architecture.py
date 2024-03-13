import argparse
import string
from dataclasses import dataclass
from typing import List

from graphviz import Digraph
from transformers import AutoConfig

from mergekit.architecture import ConfiguredArchitectureInfo, get_architecture_info


@dataclass
class Edge:
    start: str
    end: str
    name: str


@dataclass
class Graph:
    nodes: List[str]
    edges: List[Edge]


def create_graph(data: ConfiguredArchitectureInfo):
    input_to_node = {}  #  edge -> [node]
    # TODO: change name
    # Guarantees: the node is only mentioned once
    node_to_output = {}  #  node -> [edge]

    # TODO: worry about the name later
    residual_connections = {}

    node_list = data.all_weights()

    for weight in node_list:
        if weight.input_space:
            if weight.input_space not in input_to_node:
                input_to_node[weight.input_space] = set()

            input_to_node[weight.input_space].add(weight.name)

        if weight.output_space:
            if weight.output_space not in node_to_output:
                node_to_output[weight.name] = set()

            node_to_output[weight.name].add(weight.output_space)

    # TODO: figure out guarantee on this always containing two weights
    for layer in data.procedural_spaces():
        residual_connections[layer.inputs[0]] = (layer.inputs[1], layer.name)

    edges_list = []

    for node in node_to_output.keys():
        for connected_edge in node_to_output[node]:
            # find connected edge
            if connected_edge in residual_connections:
                transfer_edge, name = residual_connections[connected_edge]
                connected_nodes = input_to_node[transfer_edge]
                for connected_node in connected_nodes:
                    edges_list.append(Edge(node, connected_node, name))

            if connected_edge in input_to_node:
                connected_nodes = input_to_node[connected_edge]
                for connected_node in connected_nodes:
                    edges_list.append(Edge(node, connected_node, connected_edge))

    return Graph(nodes=[n.name for n in node_list], edges=edges_list)


def create_graphviz_diagram(graph):
    dot = Digraph(comment="Model Architecture")

    for node in graph.nodes:
        dot.node(node, node)

    for edge in graph.edges:
        print(edge)
        dot.edge(edge.start, edge.end, label=edge.name)

    return dot


if __name__ == "__main__":
    # Load the JSON data
    parser = argparse.ArgumentParser(
        description="Create a graphviz diagram of the model architecture."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name whose architecture is defined in Json architecture",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="The name of the output file (without the file extension).",
    )
    args = parser.parse_args()

    model_config = AutoConfig.from_pretrained(args.model)

    arch_info = get_architecture_info(model_config)
    configured_arch_info = ConfiguredArchitectureInfo(
        info=arch_info, config=model_config
    )

    g = create_graph(configured_arch_info)

    diagram = create_graphviz_diagram(g)

    diagram.render(args.output, format="png")
    print("Diagram created.")
