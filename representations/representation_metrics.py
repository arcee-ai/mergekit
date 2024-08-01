from pathlib import Path

import click
import yaml

from experiment_setup import Configuration, ExperimentFactory

@click.command()
@click.option('--config_yml', default="config_i_block.yml", help='path to the configuration file.')
def main(config_yml: str = "config.yml"):
    mergekit_root = Path(__file__).parent.parent
    config = yaml.safe_load(open(mergekit_root / 'representations' / 'configs' / config_yml, 'r'))
    config['out_dir'] = mergekit_root / 'representations' / 'stored_results'
    config['stored_representations'] = mergekit_root / 'representations' / 'stored_representations'
    config = Configuration.from_dict(config)

    experiment = ExperimentFactory.create(config.comparison_type.name.lower())
    experiment.run(config)

if __name__ == "__main__":
    main()
