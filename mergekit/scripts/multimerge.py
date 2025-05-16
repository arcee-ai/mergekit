# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1

import logging
import os
from typing import Dict, Optional, Set, Tuple, Union

import click
import yaml

from mergekit.common import ImmutableMap, ModelReference
from mergekit.config import MergeConfiguration
from mergekit.graph import Executor, Task
from mergekit.merge import run_merge
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options

LOG = logging.getLogger("multimerge")


MODEL_CHECK_FILENAMES = [
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
]


class MergeModelTask(Task[str]):
    config_yaml: str
    name: str
    input_merges: ImmutableMap[str, "MergeModelTask"]
    options: MergeOptions
    out_path: str
    lazy: bool = True

    def arguments(self):
        return {str(key): self.input_merges[key] for key in self.input_merges}

    def execute(self, **kwargs):
        if (
            self.lazy
            and os.path.exists(os.path.join(self.out_path, "config.json"))
            and any(
                os.path.exists(os.path.join(self.out_path, filename))
                for filename in MODEL_CHECK_FILENAMES
            )
        ):
            LOG.info(f"Model already exists at {self.out_path}, skipping")
            return self.out_path

        LOG.info(f"Running merge for {self.name}")
        cfg = MergeConfiguration.model_validate(yaml.safe_load(self.config_yaml))

        run_merge(
            cfg,
            self.out_path,
            options=self.options,
        )
        LOG.info(f"Merge complete for {self.name}")
        return self.out_path


@click.command("mergekit-multimerge", cls=PrettyPrintHelp)
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--out-path",
    type=click.Path(),
    required=False,  # validated later
    help="Path to save the final merged model",
)
@click.option(
    "--intermediate-dir",
    "-I",
    type=click.Path(),
    required=True,
    help="Directory to store intermediate merges",
)
@click.option(
    "--lazy/--no-lazy",
    default=True,
    help="Skip merges that already exist",
)
@add_merge_options
def main(
    config_file: str,
    intermediate_dir: str,
    out_path: Optional[str],
    lazy: bool,
    merge_options: MergeOptions,
):
    """Execute a set of potentially interdependent merge recipes.

    The configuration file should be a YAML file containing multiple
    documents, each of which is a merge configuration with the addition
    of a `name` field.

    The `intermediate_dir` is used to store intermediate merge results.
    Any merge configuration with a `name` field will be saved to this
    directory. If an unnamed merge configuration is present, it will be
    saved to `out_path` (which is required in this case)."""
    merge_options.apply_global_options()
    os.makedirs(intermediate_dir, exist_ok=True)

    with open(config_file, "r", encoding="utf-8") as file:
        config_source = file.read()

    merge_configs, dependencies = load_config(config_source, intermediate_dir)

    # Validate out_path requirement
    if None in merge_configs and not out_path:
        raise click.UsageError(
            "--out-path is required when configuration contains an unnamed final merge"
        )
    tasks = make_tasks(
        merge_configs, dependencies, merge_options, intermediate_dir, out_path, lazy
    )

    executor = Executor(
        tasks, math_device="cpu", storage_device="cpu"
    )  # inner executors will handle accelerator
    executor.execute(desc="Merging models")


def patched_config(config: MergeConfiguration, merge_names: Set[str], working_dir: str):
    """Replace instances of intermediate merge names with actual paths.

    Also returns the set of intermediate merge names that were used.

    Args:
        config: The configuration to patch
        merge_names: The set of all merge names
        working_dir: The directory to use as the base for relative paths
    """
    used = set()

    def _patch_mr(value: Union[dict, list, str, int, None]):
        nonlocal used
        if isinstance(value, list):
            return [_patch_mr(x) for x in value]
        elif isinstance(value, dict):
            if set(value.keys()) == {"model", "lora", "override_architecture"}:
                # is a ModelReference
                base = value["model"]["path"]
                if base in merge_names:
                    value["model"] = value["model"].copy()
                    value["model"]["path"] = os.path.join(working_dir, base)
                    used.add(base)
                return value
            return {k: _patch_mr(v) for k, v in value.items()}
        elif isinstance(value, str):
            try:
                mr = ModelReference.model_validate(value)
                if mr.model.path in merge_names:
                    used.add(mr.model.path)
                    return ModelReference(
                        model={
                            "path": os.path.join(working_dir, mr.model.path),
                            "revision": mr.model.revision,
                        },
                        lora=mr.lora,
                        override_architecture=mr.override_architecture,
                    ).model_dump()
            except ValueError:
                pass
        return value

    new_dict = _patch_mr(config.model_dump())
    return MergeConfiguration.model_validate(new_dict), used


def make_tasks(
    merge_configs: Dict[str, MergeConfiguration],
    dependencies: Dict[str, Set[str]],
    merge_options: MergeOptions,
    intermediate_dir: str,
    out_path: Optional[str],
    lazy: bool,
):
    """Build the task dependency graph for the merge recipes."""
    touched = set()
    tasks = {}

    def _make_task(name: str):
        nonlocal touched, tasks, out_path
        if name in tasks:
            return tasks[name]
        elif name in touched:
            raise ValueError(f"Circular dependency detected involving {name}")
        touched.add(name)
        if name is None:
            # out_path validation happens earlier in main()
            merge_out_path = out_path
        else:
            merge_out_path = os.path.join(intermediate_dir, name)
        tasks[name] = MergeModelTask(
            config_yaml=merge_configs[name].to_yaml(),
            name=name or "final merge",
            input_merges=ImmutableMap(
                {dep: _make_task(dep) for dep in dependencies[name]}
            ),
            options=merge_options,
            out_path=merge_out_path,
            lazy=lazy,
        )
        return tasks[name]

    # Only create tasks that exist in the config (allow missing None)
    tasks_to_create = [
        name for name in merge_configs.keys() if name is not None or out_path
    ]
    tasks = [_make_task(name) for name in tasks_to_create]
    return tasks


def load_config(
    config_source: str, intermediate_dir: str
) -> Tuple[Dict[str, MergeConfiguration], Dict[str, Set[str]]]:
    """Load the merge configurations from the YAML source.

    Args:
        config_source: The YAML source to load
        intermediate_dir: The directory to use for intermediate merges

    Returns:
        A tuple containing:
        - A dictionary of merge configurations keyed by name
        - A dictionary of dependencies keyed by name
    """
    docs = list(yaml.safe_load_all(config_source))
    merge_configs = {}
    for doc in docs:
        if "name" in doc:
            merge_name = doc.pop("name")
        else:
            merge_name = None
        if merge_name in merge_configs:
            if merge_name is not None:
                raise ValueError(f"Duplicate merge name {merge_name}")
            else:
                raise ValueError(
                    "Multiple unnamed merge configurations are not supported"
                )
        merge_configs[merge_name] = MergeConfiguration.model_validate(doc)

    merge_names = set(merge_configs.keys())
    dependencies = {}
    for merge_name in merge_names:
        merge_config, used_names = patched_config(
            merge_configs[merge_name], merge_names, intermediate_dir
        )
        merge_configs[merge_name] = merge_config
        dependencies[merge_name] = used_names
    return merge_configs, dependencies


if __name__ == "__main__":
    main()
