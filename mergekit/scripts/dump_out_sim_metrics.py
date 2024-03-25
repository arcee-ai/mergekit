import click
import torch
import safetensors.torch
from typing import Optional, List
from zipit_metrics import CovarianceMetric

@click.command()
@click.argument("model1_path", type=str, default='gpt2_features_activation.bin')
@click.argument("model2_path", type=str, default='lvwerra_gpt2-imdb_features_activation.bin')
@click.option("--out-path", "-o", required=True, type=str, help="Output path for metric tensors")
@click.option("--metric", "-m", type=str, default='covariance', help="Metric to calculate (default: covariance)")
@click.option("--device", "-d", type=str, default="cpu", help="Device to compute on (default: cpu)")
def main(model1_path, model2_path, out_path, metric, device):
    metric_classes = {'covariance': CovarianceMetric()}
    if metric not in metric_classes:
        raise ValueError(f"Unsupported metric: {metric}")

    model1_layers = safetensors.torch.load_file(model1_path)
    model2_layers = safetensors.torch.load_file(model2_path)

    metrics_results = {}

    for layer_name in model1_layers.keys():
        layer1_data = model1_layers[layer_name].float().to(device)
        layer2_data = model2_layers[layer_name].float().to(device)

        concatenated_layer = torch.cat((layer1_data, layer2_data), dim=-1)
        concatenated_layer = concatenated_layer.view(-1, concatenated_layer.shape[-1])
        
        metric_instance = metric_classes[metric]()
        metric_instance.update(concatenated_layer)
        final_metric = metric_instance.finalize()
        identifier = f"{layer_name}_{metric}"
        metrics_results[identifier] = final_metric

    # Saving the metrics results as SafeTensors
    for identifier, tensor in metrics_results.items():
        safetensors.torch.save_file({identifier: tensor}, f"{out_path}/{identifier}_metric.safetensor")

if __name__ == "__main__":
    main()