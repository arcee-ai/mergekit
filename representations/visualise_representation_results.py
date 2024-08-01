import click
from mergekit.plot_tools.plot_tools import create_app, ResultsHandler
from mergekit.metric_methods.base import Results
from pathlib import Path

def main(input_dir):
    handler = ResultsHandler()
    for res in Path(input_dir).iterdir():
        results = Results()
        results = results.load(res.absolute())

        handler.load_results(results)

    app = create_app(results_handler=handler)
    app.run_server()

if __name__ == '__main__':
    mergekit_root = Path(__file__).parent.parent
    input_dir = mergekit_root / 'representations' / 'stored_results'

    main(input_dir)