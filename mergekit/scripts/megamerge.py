#!/usr/bin/env python3

import yaml
import os
import re
import click
import logging
import os

from mergekit.merge import MergeOptions, run_merge
from mergekit.config import MergeConfiguration
from mergekit.options import add_merge_options

# Regex that matches huggingface path
hf_path = r"^[a-zA-Z0-9\-]+/[a-zA-Z0-9\-\._]+(?:\+.+)$"
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

def merge(m, options, force):
    # check if out_path exists
    if not force and os.path.exists(m):
        logging.info(f"Skipping {m} as it already exists")
        del merges[m]
        return
    elif force and os.path.exists(m):
        logging.info(f"Overwriting {m} as --force was specified")

    if len(merges[m]["deps"]) != 0:
        for dep in merges[m]["deps"]:
            if dep in merges:
                merge(dep, options, force)

    logging.info(f"Merging model {m}")
    merge_config: MergeConfiguration = MergeConfiguration.model_validate(merges[m])
    run_merge(
        merge_config,
        merges[m]["out_path"],
        options=options,
    )
    del merges[m]

@click.command("mergekit-mega")
@click.argument("config_file")
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
    force: bool,
    verbose: bool,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    with open(config_file, "r") as f:
        data = yaml.load_all(f, Loader=yaml.FullLoader)

        for d in data:
            merges[d["out_path"]] = d
            merges[d["out_path"]]["deps"] = []
            if "slices" in d:
                for slc in d["slices"]:
                    for src in slc["sources"]:
                        if "model" in src and src["model"] is not None:
                            # if the model is a huggingface model, skip it
                            if not re.match(hf_path, src["model"]):
                                merges[d["out_path"]]["deps"].append(src["model"])
            if "models" in d:
                for mdl in d["models"]:
                    if "model" in mdl and mdl["model"] is not None:
                        # if the model is a huggingface model, skip it
                        if not re.match(hf_path, mdl["model"]):
                            merges[d["out_path"]]["deps"].append(mdl["model"])

    logging.info("Merging: " + ', '.join(merges))

    if (dep := has_circular_dependency(merges)) is not None:
        logging.error(f"Circular dependency detected: {dep}")
        exit(1)

    while len(merges) != 0:
        m = list(merges.keys())[0]
        merge(m, merge_options, force)

if __name__ == "__main__":
    main()