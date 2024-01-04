#!/usr/bin/env python3

import yaml
import os
import re
import click
import logging
import os
from pathlib import Path

from mergekit.merge import MergeOptions, run_merge
from mergekit.config import MergeConfiguration
from mergekit.options import add_merge_options

merges = {}

def has_circular_dependency(nodes):
    def dfs(node, visited, stack):
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

def merge(m, merge_options, force, out_path):
    # check if output_path exists
    if os.path.exists(out_path / m):
        if not force:
            logging.info(f"Skipping {m} as it already exists")
            del merges[m]
            return
        else:
            logging.info(f"Overwriting {m} as --force was specified")

    if len(merges[m]["deps"]) != 0:
        for dep in merges[m]["deps"]:
            if dep in merges:
                merge(dep, merge_options, force, out_path)

    logging.info(f"Merging model {m}")
    merge_config: MergeConfiguration = MergeConfiguration.model_validate(merges[m])
    run_merge(
        merge_config,
        str(out_path / merges[m]["name"]),
        options=merge_options,
    )
    del merges[m]

@click.command("mergekit-mega")
@click.argument("config_file")
@click.argument("out_path")
@click.option(
    "--verbose", "-v", type=bool, default=False, is_flag=True, help="Verbose logging"
)
@click.option(
    "--force", "-f", type=bool, default=False, is_flag=True, help="overwrite existing merge results instead of skipping them"
)
@add_merge_options
def main(
    merge_options: MergeOptions,
    config_file: str,
    out_path: str,
    force: bool,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    out_path = Path(out_path)
    with open(config_file, "r") as f:
        data = yaml.load_all(f, Loader=yaml.FullLoader)

        for d in data:
            merges[d["name"]] = d
            merges[d["name"]]["deps"] = []
            if "slices" in d:
                for slc in d["slices"]:
                    for src in slc["sources"]:
                        if "model" in src and src["model"] is not None:
                            model_lora = src["model"].split("+")
                            # name must not have a slash to avoid path traversal
                            # therefore, we can use it to check if its a merge from the config
                            if "/" not in model_lora[0]:
                                # avoid duplicate deps
                                if model_lora[0] not in merges[d["name"]]["deps"]:
                                    merges[d["name"]]["deps"].append(model_lora[0])
                                src["model"] = str(out_path / model_lora[0])
                                if len(model_lora) == 2:
                                    src["model"] += "+" + model_lora[1]
            if "models" in d:
                for mdl in d["models"]:
                    if "model" in mdl and mdl["model"] is not None:
                        model_lora = mdl["model"].split("+")
                        # name must not have a slash to avoid path traversal
                        # therefore, we can use it to check if its a merge from the config
                        if "/" not in model_lora[0]:
                            # avoid duplicate deps
                            if model_lora[0] not in merges[d["name"]]["deps"]:
                                merges[d["name"]]["deps"].append(model_lora[0])
                            mdl["model"] = str(out_path / model_lora[0])
                            if len(model_lora) == 2:
                                mdl["model"] += "+" + model_lora[1]

    logging.info("Merging: " + ', '.join(merges))

    if (dep := has_circular_dependency(merges)) is not None:
        logging.error(f"Circular dependency detected: {dep}")
        exit(1)

    while len(merges) != 0:
        m = list(merges.keys())[0]
        merge(m, merge_options, force)

if __name__ == "__main__":
    main()