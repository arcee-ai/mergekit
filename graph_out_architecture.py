import argparse
import string
from dataclasses import dataclass
from typing import Optional

import pydantic
from graphviz import Digraph


class Weight(pydantic.BaseModel):
    name: str
    output_space: Optional[str] = None
    input_space: Optional[str] = None


class LayerTemplate(pydantic.BaseModel):
    normal_weights: list[Weight]
    residual_weights: list[Weight]


class Model(pydantic.BaseModel):
    pre_weights: list[Weight]
    post_weights: list[Weight]
    layer_template: LayerTemplate


class TemplateWithArithmetic(string.Template):
    idpattern = r"(?a:[_a-z][_a-z0-9]*([+-]1)?)"


# Assuming the JSON data is loaded into a variable named `data`


@dataclass
class Edge:
    start: str
    end: str
    name: str


@dataclass
class Graph(object):
    nodes: list[str]
    edges: list[Edge]


def _template_substitution(template: str, layer_idx: int) -> str:
    if "{" not in template:
        return template

    substitutions = {
        "layer": layer_idx,
        "layer+1": layer_idx + 1,
        "layer-1": layer_idx - 1,
    }

    return TemplateWithArithmetic(template).substitute(substitutions)


def create_graph(data):
    input_to_node = {}  #  edge -> [node]
    # TODO: change name
    # Guarantees: the node is only mentioned once
    node_to_output = {}  #  node -> [edge]

    # TODO: worry about the name later
    residual_connections = {}

    node_list = (
        data.pre_weights + data.post_weights + data.layer_template.normal_weights
    )

    for weight in node_list:
        if weight.input_space:
            if weight.input_space not in input_to_node:
                input_to_node[weight.input_space] = set()

            input_to_node[weight.input_space].add(weight.name)

        if weight.output_space:
            if weight.output_space not in node_to_output:
                node_to_output[weight.name] = set()

            node_to_output[weight.name].add(weight.output_space)

    for layer in data.layer_template.residual_weights:
        residual_connections[layer.input_space] = (layer.output_space, layer.name)

    edges_list = []
    # TODO: filter for Nones

    # simplify this
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


def expand_template(model: Model, num_layers: int) -> str:
    # TODO: figure out the non-deprecated view
    # TODO: should model be set to frozen

    new_normal_weights = []
    new_residual_weights = []

    # render template
    for weight in model.layer_template.normal_weights:
        weight_template = weight.model_dump_json()
        for i in range(args.num_layers + 1):
            punched_weight = _template_substitution(weight_template, i)
            new_normal_weights.append(Weight.parse_raw(punched_weight))

    for weight in model.layer_template.residual_weights:
        # stringify the weight and apply the template
        weight_template = weight.model_dump_json()
        for i in range(1, args.num_layers + 1):
            punched_weight = _template_substitution(weight_template, i)
            # convert the string back to a dictionary
            new_residual_weights.append(Weight.parse_raw(punched_weight))

    model.layer_template.normal_weights = new_normal_weights
    model.layer_template.residual_weights = new_residual_weights

    return model


if __name__ == "__main__":
    # Load the JSON data
    parser = argparse.ArgumentParser(
        description="Create a graphviz diagram of the model architecture."
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="The JSON file containing the model architecture data.",
    )

    # add argument for number of layers
    parser.add_argument(
        "--num_layers", type=int, help="The number of layers in the model."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="The name of the output file (without the file extension).",
    )
    args = parser.parse_args()

    json_definition = open(args.json_file).read()

    model = Model.model_validate_json(json_definition)

    expanded_model = expand_template(model, args.num_layers)

    g = create_graph(model)

    diagram = create_graphviz_diagram(g)

    diagram.render(args.output, format="png")
    print("Diagram created.")
