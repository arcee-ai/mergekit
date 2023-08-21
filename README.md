## ties-merge

Implementation of the merging method described in [Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708). Lazily loads sharded models so you don't need enough RAM/VRAM to hold all N models at the same time.

This can be used to merge an arbitrary number of models that share a common base model with minimal degradation of performance.

Example usage:
```python ties_merge.py TheBloke/Llama-2-13B-fp16 ./PlatypusWizard-13b --merge WizardLM/WizardLM-13B-V1.2 --merge garage-bAInd/Platypus2-13B --cuda```