#!/usr/bin/env python3
"""
Merges multiple models and their dependencies into a single model
using multiple merge yaml documents in a single yaml file as the input
"""

import logging
import os
import sys
from pathlib import Path

import click
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
from mergekit.options import add_merge_options

merges = {}


def has_circular_dependency(nodes):
    """
    Detects circular in merges dependencies using DFS
    Returns the node where the cycle is detected
    """

    def dfs(node, visited, stack):
        """
        Returns True if a cycle is detected
        """
        visited[node] = True
        stack[node] = True

        for dependency in nodes[node]["deps"]:
            if not visited[dependency]:
                if dfs(dependency, visited, stack):
                    return True
            elif stack[dependency]:
                return True

        stack[node] = False
        return False

    visited = {key: False for key in nodes}
    stack = {key: False for key in nodes}

    for node in nodes:
        if not visited[node]:
            if dfs(node, visited, stack):
                return node

    return None


def merge(m: str, merge_options: MergeOptions, force: bool, out_path: Path):
    """
    Merges a model and its dependencies

    Params:
        m: name of the model to merge
        merge_options: MergeOptions
        force: overwrite existing merge results
        out_path: output path
    """
    # check if output_path exists
    if os.path.exists(out_path / m):
        if not force:
            logging.info("Skipping %s as it already exists", m)
            del merges[m]
            return
        logging.info("Overwriting %s as --force was specified", m)

    if len(merges[m]["deps"]) != 0:
        for dep in merges[m]["deps"]:
            if dep in merges:
                merge(dep, merge_options, force, out_path)

    logging.info("Merging model %s", m)
    merge_config: MergeConfiguration = MergeConfiguration.model_validate(merges[m])
    run_merge(
        merge_config,
        str(out_path / merges[m]["name"]),
        options=merge_options,
    )
    del merges[m]


def add_model_deps(model: str, name: str, out_path: Path):
    """
    Adds a model to `name`s dependencies if it is not already there and is a merge
    """
    model_lora = model.split("+")
    # name must not have a slash to avoid path traversal
    # therefore, we can use it to check if its a merge from the config
    if "/" not in model_lora[0]:
        # avoid duplicate deps
        if model_lora[0] not in merges[name]["deps"]:
            merges[name]["deps"].append(model_lora[0])
        model = str(out_path / model_lora[0])
        if len(model_lora) == 2:
            model += "+" + model_lora[1]

    return model


@click.command("mergekit-mega")
@click.argument("config_file")
@click.argument("out_path")
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@click.option(
    "--force",
    "-f",
    type=bool,
    default=False,
    is_flag=True,
    help="Overwrite existing merge results instead of skipping them",
)
@click.option(
    "--require-nameless",
    "-R",
    type=bool,
    default=False,
    is_flag=True,
    help="Enforces exactly one unnamed merge in the YAML, which will inherit the input file's name.",
)
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
    force: bool,
    verbose: bool,
    require_nameless: bool,
):
    """
    Main entrypoint for mergekit-mega command see module docstring for more info
    Params are supplied by click decorators
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    out_path = Path(out_path)
    final_found = False

    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.load_all(f, Loader=yaml.FullLoader)

        for d in data:
            if "name" not in d:
                if final_found:
                    logging.error("Only one merge must not have a name")
                    sys.exit(1)
                # this sets the name of the final merge to the config file name without the extension
                d["name"] = os.path.basename(config_file).rsplit(".", maxsplit=1)[0]
                final_found = True

            if "/" in d["name"]:
                logging.error("name must not contain a slash")
                sys.exit(1)

            merges[d["name"]] = d
            merges[d["name"]]["deps"] = []
            if "base_model" in d:
                d["base_model"] = add_model_deps(d["base_model"], d["name"], out_path)
            if "slices" in d:
                for slc in d["slices"]:
                    for src in slc["sources"]:
                        src["model"] = add_model_deps(src["model"], d["name"], out_path)
            if "models" in d:
                for mdl in d["models"]:
                    mdl["model"] = add_model_deps(mdl["model"], d["name"], out_path)

    if require_nameless and not final_found:
        logging.error("No final merge found")
        sys.exit(1)

    logging.info("Merging: %s", ", ".join(merges))

    if (dep := has_circular_dependency(merges)) is not None:
        logging.error("Circular dependency detected: %s", dep)
        sys.exit(1)

    while len(merges) != 0:
        m = list(merges.keys())[0]
        merge(m, merge_options, force, out_path)


if __name__ == "__main__":
    main()
