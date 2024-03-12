import argparse
import json
import pprint
import string
from typing import Optional

import pydantic
from graphviz import Digraph


class Weight(pydantic.BaseModel):
    name: str
    output_space: Optional[str]
    input_space: Optional[str]


class LayerTemplate(pydantic.BaseModel):
    normal_weights: list[Weight]
    residual_weights: list[Weight]


class TemplateWithArithmetic(string.Template):
    idpattern = r"(?a:[_a-z][_a-z0-9]*([+-]1)?)"


def _template_substitution(template: str, layer_idx: int) -> str:
    if "{" not in template:
        return template

    substitutions = {
        "layer": layer_idx,
        "layer+1": layer_idx + 1,
        "layer-1": layer_idx - 1,
    }

    return TemplateWithArithmetic(template).substitute(substitutions)


# Assuming the JSON data is loaded into a variable named `data`


def create_graphviz_diagram(data):
    dot = Digraph(comment="Model Architecture")

    # Add pre_weights and post_weights as nodes
    for weight in data["pre_weights"]:
        print(weight["name"])
        dot.node(weight["name"], weight["name"])

    for weight in data["post_weights"]:
        dot.node(weight["name"], weight["name"])

    # Add layer weights as nodes and connect them
    for layer in data["layers_template"]["normal_weights"]:
        print(layer)
        dot.node(layer.name, layer.name)
        if layer.input_space:
            dot.edge(layer.input_space, layer.name)
        if layer.output_space:
            dot.edge(layer.name, layer.output_space)

    # Add procedural spaces and their connections
    for space in data["layers_template"]["residual_weights"]:
        # dot.node(space.name, space.name)
        if space.input_space:
            dot.edge(space.input_space, space.name)
        if space.output_space:
            dot.edge(space.name, space.output_space)

    return dot


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

    load_json = json.load(open(args.json_file))
    pprint.pprint(load_json)
    template_normal_weights = [
        Weight(**weight) for weight in load_json["layers_template"]["normal_weights"]
    ]
    template_residual_weights = [
        Weight(**weight) for weight in load_json["layers_template"]["residual_weights"]
    ]

    new_normal_weights = []
    new_residual_weights = []

    # render template
    for weight in template_normal_weights:
        # stringify the weight and apply the template
        weight_template = weight.model_dump_json()
        for i in range(1, args.num_layers + 1):
            punched_weight = _template_substitution(weight_template, i)
            new_normal_weights.append(Weight.parse_raw(punched_weight))

    for weight in template_residual_weights:
        # stringify the weight and apply the template
        weight_template = weight.model_dump_json()
        for i in range(1, args.num_layers + 1):
            punched_weight = _template_substitution(weight_template, i)
            # convert the string back to a dictionary
            new_residual_weights.append(Weight.parse_raw(punched_weight))

    load_json["layers_template"]["normal_weights"] = new_normal_weights
    load_json["layers_template"]["residual_weights"] = new_residual_weights

    pprint.pprint(load_json)

    diagram = create_graphviz_diagram(load_json)

    # Render the diagram to a file (e.g., 'model_architecture.gv')
    diagram.render(args.output, format="png")
    print("Diagram created.")
