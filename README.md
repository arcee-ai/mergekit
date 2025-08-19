# LLMTailor

LLMTailor is an enhanced fork of [mergekit](https://github.com/arcee-ai/mergekit), designed for **layer-wise merging of large language models (LLMs)** with extended support for:

- ✅ Compatible with our new checkpoint system StreamCheck
- ✅ Layer-wise model merging & selection  
- ✅ Optimizer state reconstruction (supports ZeRO-3 shards)  
- ✅ Tokenizer & embedding adaptation: these auxiliary layers in LLMs could also be selected and merged now
- ✅ Backward compatibility with most `mergekit` plans  

> **Note:** LLMTailor retains most of `mergekit`’s original merging capabilities while adding extensions (`llmtailor.*` fields in YAML) for training-oriented scenarios.

---

## Installation

### From GitHub (development version)
```bash
git clone https://github.com/SunMinqiu/LLMTailor.git
cd LLMTailor
pip install -e .
```

## Quick Start
- The example can be found in the folder.x