from __future__ import annotations

from pathlib import Path

import click

from mergekit.atlascloud import (
    DEFAULT_ENV_FILE,
    chat_completion,
    AtlasCloudConfig,
    extract_message_text,
)


@click.command("mergekit-atlascloud-test")
@click.option(
    "--env-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_ENV_FILE,
    show_default=True,
    help="Path to the local AtlasCloud credentials file.",
)
@click.option(
    "--prompt",
    type=str,
    default="Reply with the single word: pong",
    show_default=True,
    help="Prompt used for the smoke test.",
)
@click.option("--timeout", type=int, default=60, show_default=True)
def main(
    env_file: Path,
    prompt: str,
    timeout: int,
):
    config = AtlasCloudConfig.from_env(env_file=str(env_file))

    response = chat_completion(
        config=config,
        prompt=prompt,
        timeout=timeout,
    )
    click.echo(extract_message_text(response))


if __name__ == "__main__":
    main()
