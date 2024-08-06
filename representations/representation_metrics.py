from pathlib import Path

import click
import yaml

from experiment_setup import Configuration, ExperimentFactory

def run(config_yml: str = "config"):
    mergekit_root = Path(__file__).parent.parent

    config = yaml.safe_load(open((mergekit_root / 'representations' / 'configs' / config_yml).with_suffix('.yml'), 'r'))
    config['out_dir'] = mergekit_root / 'representations' / 'results_out'
    config['representations_to_analyse'] = mergekit_root / 'representations' / 'representations_to_analyse'
    config = Configuration.from_dict(config)

    experiment = ExperimentFactory.create(config.comparison_type.name.lower())
    experiment.run(config)

@click.command()
@click.option('--config_yml', default="config_i_block", help='path to the configuration file.')
def main(config_yml: str = "config"):
    run(config_yml)

if __name__ == "__main__":
    main()
