import click
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.merge import run_merge
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler
from mergekit.metric_methods.base import Results

def create_temp_config(config_yml, **kwargs):
    with open(config_yml, "r", encoding="utf-8") as config:
        config = yaml.safe_load(config)
        config.update(kwargs)
        return MergeConfiguration.model_validate(config)

@click.command()
@click.option('--output_path', default="./merged", help='folder to store the result in.')
@click.option('--config_yml', default="./examples/metrics-small.yml", help='merge configuration file.')
@click.option('--copy_tokenizer', default=True, help='')
@click.option('--lazy_unpickle', default=False, help='experimental low-memory model loader.')
@click.option('--low_cpu_memory', default=False, help='enable if you somehow have more VRAM than RAM+swap')
def main(output_path, config_yml, copy_tokenizer, lazy_unpickle, low_cpu_memory):
    with open(config_yml, "r", encoding="utf-8") as config:
        config = yaml.safe_load(config)
        metric_config = MergeConfiguration.model_validate(config)

        models = metric_config.models
        intra_model = config['parameters']['intra_model_metrics']
        inter_model = config['parameters']['inter_model_metrics']

    intra_results = {}
    inter_results = None
    if intra_model:
        print(f"Running intra-model metrics for {len(models)} models: {models}")
        for model in models:
            temp_config = create_temp_config(config_yml, models=[{'model':model.model.model.path}], parameters=({'intra_model_metrics':True, 'inter_model_metrics':False}))
            print(f"  {model}")
            metrics_out = run_merge(
                temp_config,
                out_path=output_path,
                options=MergeOptions(
                    cuda=torch.cuda.is_available(),
                    copy_tokenizer=copy_tokenizer,
                    lazy_unpickle=lazy_unpickle,
                    low_cpu_memory=low_cpu_memory,
                ),
            )
            intra_results[model.model] = Results().load_metrics(metrics_out, model_refs=[model.model])
        
    if inter_model:
        assert len(models) == 2, "Inter-model metrics require exactly 2 models"
        print(f"Running inter-model metrics for {models}")
        temp_config = create_temp_config(config_yml, parameters=({'intra_model_metrics':False, 'inter_model_metrics':True}))

        print(f"  {models}")
        metrics_out = run_merge(
            temp_config,
            out_path=output_path,
            options=MergeOptions(
                cuda=torch.cuda.is_available(),
                copy_tokenizer=copy_tokenizer,
                lazy_unpickle=lazy_unpickle,
                low_cpu_memory=low_cpu_memory,
            ),
        )
        inter_results = Results().load_metrics(metrics_out, model_refs=models)


    handler = ResultsHandler()

    handler.load_results(inter_results)
    for result in intra_results.values():
        handler.load_results(result)


    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    main()