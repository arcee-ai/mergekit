import click
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler
from mergekit.metric_methods.base import Results

@click.command()
@click.option('--input_dir', 
              default="/Users/elliotstein/Documents/Arcee/mergekit/representations/results_to_visualise", 
              help="path to load the results from.")
def main(input_dir):
    handler = ResultsHandler()
    for res in Path(input_dir).iterdir():
        results = Results()
        results = results.load(res.absolute())

        handler.load_results(results)

    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    main()