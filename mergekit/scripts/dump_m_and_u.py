import click
import torch
import safetensors.torch
from typing import Optional, List
from mergekit.scripts.zipit_utils import CovarianceMetric, remove_pads

@click.command()
@click.argument("model1-ft", type=str, required=True)
@click.argument("model2-ft", type=str, required=True)
@click.option("--out_path", required=True, type=str, help="Output path for metric tensors")
@click.option("--metric", "-m", type=str, default='covariance', help="Metric to calculate (default: covariance)")
@click.option("--device", "-d", type=str, default="cpu", help="Device to compute on (default: cpu)")

def main(model1_ft, model2_ft, out_path, metric, device):
    metric_classes = {'covariance': CovarianceMetric()}
    if metric not in metric_classes:
        raise ValueError(f"Unsupported metric: {metric}")

    model1_features = safetensors.torch.load_file(model1_ft)
    model2_features = safetensors.torch.load_file(model2_ft)

    model_1_attention_mask = model1_features.pop("attention_mask")
    model_2_attention_mask = model2_features.pop("attention_mask")

    metrics_results = {}

    for layer_name in model1_features.keys():

        model1_feature = model1_features[layer_name].float().to(device)
        model2_feature = model2_features[layer_name].float().to(device)

        model1_filtered_feature = remove_pads(model_1_attention_mask, model1_feature)
        model2_filtered_feature = remove_pads(model_2_attention_mask, model1_feature)

        print(model2_filtered_feature.shape)
        exit()

        concatenated_feature = torch.cat((model1_filtered_feature, model2_filtered_feature), dim=-1)

        if "attention" in layer_name:
            # how to compute the metrics here?
            print(layer_name)
        else:
            print(layer_name)

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

# python dump_m_and_u.py /Users/gayalshamane/Documents/mergekit/mergekit/scripts/dump_output/TinyLlama_TinyLlama-1.1B-Chat-v0.6_features.bin /Users/gayalshamane/Documents/mergekit/mergekit/scripts/dump_output/TinyLlama_TinyLlama-1.1B-Chat-v1.0_features.bin --out_path /Users/gayalshamane/Documents/mergekit/mergekit/scripts/m_v_out