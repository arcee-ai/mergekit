#!/usr/bin/env python3

import yaml
import os
import re
from typing import Optional
from typing_extensions import Annotated
import typer
import logging
import os

from mergekit.merge import MergeOptions, run_merge
from mergekit.common import parse_kmb
from mergekit.config import MergeConfiguration

# Regex that matches huggingface path
hf_path = r"^[a-zA-Z0-9\-]+/[a-zA-Z0-9\-\._]+(?:\+.+)$"
merges = {}

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

def main(
    config_file: Annotated[str, typer.Argument(help="YAML configuration file")],
    lora_merge_cache: Annotated[
        Optional[str],
        typer.Option(help="Path to store merged LORA models", metavar="PATH"),
    ] = None,
    transformers_cache: Annotated[
        Optional[str],
        typer.Option(
            help="Override storage path for downloaded models", metavar="PATH"
        ),
    ] = None,
    cuda: Annotated[
        bool, typer.Option(help="Perform matrix arithmetic on GPU")
    ] = False,
    low_cpu_memory: Annotated[
        bool,
        typer.Option(
            help="Store results and intermediate values on GPU. Useful if VRAM > RAM"
        ),
    ] = False,
    copy_tokenizer: Annotated[
        bool, typer.Option(help="Copy a tokenizer to the output")
    ] = True,
    allow_crimes: Annotated[
        bool, typer.Option(help="Allow mixing architectures")
    ] = False,
    out_shard_size: Annotated[
        Optional[int],
        typer.Option(
            help="Number of parameters per output shard  [default: 5B]",
            parser=parse_kmb,
            show_default=False,
            metavar="NUM",
        ),
    ] = parse_kmb("5B"),
    verbose: Annotated[bool, typer.Option("-v", help="Verbose logging")] = False,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Trust remote code when merging LoRAs")
    ] = False,
    clone_tensors: Annotated[
        bool,
        typer.Option(
            help="Clone tensors before saving, to allow multiple occurrences of the same layer"
        ),
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Force overwrite of existing output")
    ] = False,
    lazy_unpickle: Annotated[
        bool, typer.Option(help="Experimental lazy unpickler for lower memory usage")
    ] = False,
):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    with open(config_file, "r") as f:
        data = yaml.load_all(f, Loader=yaml.FullLoader)

        # find leaf merges, the ones that don't have a local path specified in slices[].sources[].model or models[].model
        leaf_merges = []
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

    options = MergeOptions(
            lora_merge_cache=lora_merge_cache,
            transformers_cache=transformers_cache,
            cuda=cuda,
            low_cpu_memory=low_cpu_memory,
            copy_tokenizer=copy_tokenizer,
            allow_crimes=allow_crimes,
            out_shard_size=out_shard_size,
            trust_remote_code=trust_remote_code,
            clone_tensors=clone_tensors,
            lazy_unpickle=lazy_unpickle,
    )

    print("Merging:\n" + '\n'.join(merges))

    while len(merges) != 0:
        m = list(merges.keys())[0]
        merge(m, options, force)



def _main():
    # just a wee li'l stub for setuptools
    typer.run(main)


if __name__ == "__main__":
    _main()