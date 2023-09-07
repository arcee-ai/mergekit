## ties-merge

Implementation of the merging method described in [Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708). Lazily loads sharded models so you don't need enough RAM/VRAM to hold all N models at the same time.

This can be used to merge an arbitrary number of models that share a common base model with minimal degradation of performance.

Example usage:
```python main.py ./PlatypusWizard-13b --base-model TheBloke/Llama-2-13B-fp16 --merge WizardLM/WizardLM-13B-V1.2 --merge garage-bAInd/Platypus2-13B --cuda```

Per-model weights can be set, in addition to densities:
```
python main.py ./PlatypusWizard-13b --base-model TheBloke/Llama-2-13B-fp16 --cuda \
    --merge WizardLM/WizardLM-13B-V1.2 --weight 0.3 --density 0.5 \
    --merge garage-bAInd/Platypus2-13B --weight 0.5 --density 0.5
```

Also implements linear and SLERP merging.

```
python main.py ./WizardPlatypusHermes-13b --cuda --method linear --merge garage-bAInd/Platypus2-13B --weight 0.6 --merge WizardLM/WizardLM-13B-V1.2 --weight 0.2 --merge NousResearch/Nous-Hermes-Llama2-13b --weight 0.5
```

```
python main.py ./PlatypusWizard-13b-slerp --cuda --method slerp --base-model garage-bAInd/Platypus2-13B --merge WizardLM/WizardLM-13B-V1.2 --weight 0.5
```


`bakllama.py` can be used to assemble a model piecewise from layers taken from others.
Example usage:
`python bakllama.py examples/orcamini-platy-44layer.yml ./orcamini-platy-44layer`