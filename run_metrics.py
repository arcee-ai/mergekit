#%%
OUTPUT_PATH = "./merged"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "./examples/linear_small.yml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions
from mergekit.measure import run_measure
from mergekit.plot_tools.plot_tools import ModelGraph
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

out = run_measure(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
    ),
)

nn_graph = ModelGraph([pair for pair in out if pair[1] is not None])
nn_graph.construct_graph()

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Label('Model Architecture Graph | Hover for node stats | Click for node plots', style={'font-family': 'Arial'}),
    dcc.Graph(
        id='graph',
        figure=nn_graph.plot_graph(),
    ),
    dcc.Graph(id='node-details'),
    html.Label('Select Metric:', style={'font-family': 'Arial'}),
    # Add a dropdown menu to select the metric
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'SMAPE', 'value': 'SMAPE_full'},
            {'label': 'Cossim', 'value': 'cossim_full'},
            {'label': 'Scale', 'value': 'scale_full'}
        ],
        value='cossim_full',
        style={'font-family': 'Arial'}
    )
])


@app.callback(
    Output('node-details', 'figure'),
    [Input('graph', 'clickData'),
     Input('metric-dropdown', 'value')])
def display_node_data(clickData, selected_metric):
    if clickData is None:
        print("No clickData received")
        return go.Figure()
    
    try:
        node_name = clickData['points'][0]['text']
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error processing clickData: {e}")
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(
        nn_graph.metric_handler.plot_node_hist(node_name, stat=selected_metric)
        )
    fig.update_layout(title=f"Metrics for {node_name} | {selected_metric}",
                      xaxis_title="Metric",
                      yaxis_title="Value")

    return fig

if __name__ == '__main__':
    app.run_server()