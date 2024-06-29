import click
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.merge import run_merge
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler

@click.command()
@click.option('--output_path', default="./merged", help='folder to store the result in.')
@click.option('--config_yml', default="./examples/metrics-small.yml", help='merge configuration file.')
@click.option('--copy_tokenizer', default=True, help='')
@click.option('--lazy_unpickle', default=False, help='experimental low-memory model loader.')
@click.option('--low_cpu_memory', default=False, help='enable if you somehow have more VRAM than RAM+swap')
def main(output_path, config_yml, copy_tokenizer, lazy_unpickle, low_cpu_memory):
    with open(config_yml, "r", encoding="utf-8") as fp:
        metric_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

    metrics_results = run_merge(
        metric_config,
        out_path=output_path,
        options=MergeOptions(
            cuda=torch.cuda.is_available(),
            copy_tokenizer=copy_tokenizer,
            lazy_unpickle=lazy_unpickle,
            low_cpu_memory=low_cpu_memory,
        ),
    )

    handler = ResultsHandler(metrics_results)

    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    main()