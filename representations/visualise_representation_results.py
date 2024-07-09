import click
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler
from mergekit.metric_methods.base import Results

@click.command()
@click.option('--results_path', 
              default="./representations/results.pkl", 
              help="path to load the results from.")
def main(results_path):
    results = Results()
    results = results.load(results_path)

    handler = ResultsHandler()
    handler.load_results(results)

    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    main()