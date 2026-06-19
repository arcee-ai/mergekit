# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml
from transformers import AutoTokenizer

from mergekit.common import ModelReference
from mergekit.config import MergeConfiguration
from mergekit.options import MergeOptions, PrettyPrintHelp, add_merge_options

LOG = logging.getLogger(__name__)

FIM_TOKENS = ["<PRE>", "<SUF>", "<MID>", "<EOT>"]


@dataclass
class Issue:
    severity: str  # ERROR, WARNING, INFO
    message: str


def _short_name(ref: ModelReference) -> str:
    return ref.model.path.rstrip("/").split("/")[-1]


def _resolve_config(cfg: Any) -> Any:
    """For multimodal configs, unwrap to text_config to access architecture params."""
    if getattr(cfg, "hidden_size", None) is None and hasattr(cfg, "text_config"):
        return cfg.text_config
    return cfg


def _safe_get(cfg: Any, attr: str, default: Any = None) -> Any:
    return getattr(cfg, attr, default)


def _get_rope_theta(cfg: Any) -> Optional[float]:
    if hasattr(cfg, "rope_theta"):
        return cfg.rope_theta
    if hasattr(cfg, "rope_parameters") and isinstance(cfg.rope_parameters, dict):
        return cfg.rope_parameters.get("rope_theta")
    # Some configs (e.g. Ministral3) embed rope_theta inside rope_scaling
    scaling = getattr(cfg, "rope_scaling", None)
    if isinstance(scaling, dict) and "rope_theta" in scaling:
        return scaling["rope_theta"]
    return None


def check_architecture(
    configs: Dict[str, Any],
) -> Tuple[Dict[str, Dict], List[Issue]]:
    shape_params = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
    ]
    table: Dict[str, Dict] = {}
    issues: List[Issue] = []
    resolved = {name: _resolve_config(cfg) for name, cfg in configs.items()}

    for param in shape_params:
        vals = {name: _safe_get(cfg, param) for name, cfg in resolved.items()}
        table[param] = vals
        unique = {v for v in vals.values() if v is not None}
        if len(unique) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in vals.items())
            issues.append(
                Issue(
                    "ERROR",
                    f"Tensor shape mismatch in `{param}` ({detail}). "
                    "Merge will fail or produce garbage output.",
                )
            )

    model_types = {name: _safe_get(cfg, "model_type") for name, cfg in resolved.items()}
    table["model_type"] = model_types
    unique_types = {v for v in model_types.values() if v is not None}
    if len(unique_types) > 1:
        detail = ", ".join(f"{n}={v}" for n, v in model_types.items())
        issues.append(
            Issue(
                "WARNING",
                f"Model type mismatch ({detail}). Models may differ in activation "
                "functions, attention variants, or normalization — merged weights may "
                "be semantically incoherent even if tensor shapes match.",
            )
        )

    return table, issues


def check_rope(configs: Dict[str, Any]) -> Tuple[Dict[str, Dict], List[Issue]]:
    table: Dict[str, Dict] = {}
    issues: List[Issue] = []
    resolved = {name: _resolve_config(cfg) for name, cfg in configs.items()}

    thetas = {name: _get_rope_theta(cfg) for name, cfg in resolved.items()}
    table["rope_theta"] = thetas
    unique_thetas = {v for v in thetas.values() if v is not None}
    if len(unique_thetas) > 1:
        detail = ", ".join(f"{n}={v}" for n, v in thetas.items())
        issues.append(
            Issue(
                "WARNING",
                f"RoPE theta mismatch ({detail}). Positional encodings are "
                "incompatible; merged model will likely have degraded coherence, "
                "especially for longer outputs.",
            )
        )

    scalings = {name: _safe_get(cfg, "rope_scaling") for name, cfg in resolved.items()}
    table["rope_scaling"] = {
        n: (s.get("type", "?") if isinstance(s, dict) else str(s) if s else "none")
        for n, s in scalings.items()
    }
    scaling_types = {
        n: (s.get("type") if isinstance(s, dict) else None) for n, s in scalings.items()
    }
    if any(v is not None for v in scaling_types.values()):
        shown = {n: (v or "none") for n, v in scaling_types.items()}
        if len(set(shown.values())) > 1:
            issues.append(
                Issue(
                    "WARNING",
                    f"RoPE scaling mismatch: {shown}. The output config inherits the "
                    "base model's RoPE settings. If the base model lacks rope_scaling, "
                    "the merged model may produce garbled output at longer sequence "
                    "lengths. Manually add rope_scaling to the output config.json if needed.",
                )
            )

    return table, issues


def check_vocab(tokenizers: Dict[str, Any]) -> Tuple[Dict[str, Dict], List[Issue]]:
    table: Dict[str, Dict] = {}
    issues: List[Issue] = []

    vocab_sizes = {name: tok.vocab_size for name, tok in tokenizers.items()}
    table["vocab_size"] = vocab_sizes
    unique_sizes = set(vocab_sizes.values())
    if len(unique_sizes) > 1:
        detail = ", ".join(f"{n}={v}" for n, v in vocab_sizes.items())
        larger_has_fim = any(
            vocab_sizes[n] > min(unique_sizes)
            and any(tok in tokenizers[n].get_vocab() for tok in FIM_TOKENS)
            for n in tokenizers
        )
        msg = (
            f"Vocabulary size mismatch ({detail}). MergeKit will truncate to the "
            "base model's vocab. "
        )
        if larger_has_fim:
            msg += (
                "FIM tokens (<PRE>, <MID>, <SUF>, <EOT>) in the larger-vocab model "
                "will be dropped. Consider: resize_tok_vocab.py or set "
                "`tokenizer_source: union` in config."
            )
        issues.append(Issue("WARNING", msg))

    return table, issues


def check_fim_tokens(
    tokenizers: Dict[str, Any],
) -> Tuple[Dict[str, Dict], List[Issue]]:
    table: Dict[str, Dict] = {}
    issues: List[Issue] = []

    fim_presence = {
        name: any(tok in tokenizer.get_vocab() for tok in FIM_TOKENS)
        for name, tokenizer in tokenizers.items()
    }
    table["FIM tokens"] = {n: "Yes" if v else "No" for n, v in fim_presence.items()}

    if len(set(fim_presence.values())) > 1:
        fim_models = [n for n, v in fim_presence.items() if v]
        issues.append(
            Issue(
                "WARNING",
                f"{', '.join(fim_models)} contains FIM tokens (<PRE>, <MID>, <SUF>, "
                "<EOT>) but other models do not. FIM tokens may appear in natural "
                "language completions after merging. Consider adding them to a "
                "bad_words list at inference time.",
            )
        )

    return table, issues


def check_chat_template(
    tokenizers: Dict[str, Any],
) -> Tuple[Dict[str, Dict], List[Issue]]:
    table: Dict[str, Dict] = {}
    issues: List[Issue] = []

    has_template = {
        name: tok.chat_template is not None for name, tok in tokenizers.items()
    }
    table["chat_template"] = {n: "Yes" if v else "No" for n, v in has_template.items()}

    if all(has_template.values()):
        issues.append(
            Issue(
                "INFO",
                "All models have chat templates. Merging two instruct models may "
                "cause [INST]/[/INST] token bleed in outputs.",
            )
        )
    elif any(has_template.values()):
        instruct_models = [n for n, v in has_template.items() if v]
        issues.append(
            Issue(
                "INFO",
                f"{', '.join(instruct_models)} has a chat template but other models "
                "do not. Output format may be unpredictable; consider using the base "
                "model's tokenizer.",
            )
        )

    return table, issues


_STATUS = {"match": "✓", "warn": "⚠", "error": "✗"}


def _format_table(rows: Dict[str, Dict], model_names: List[str]) -> str:
    col_w = max(max(len(n) for n in model_names) + 2, 18)
    label_w = max(max(len(k) for k in rows) + 2, 20)

    lines = [" " * label_w + "".join(n.ljust(col_w) for n in model_names)]
    for param, vals in rows.items():
        row = param.ljust(label_w)
        unique = {str(v) for v in vals.values() if v is not None}
        for name in model_names:
            v = vals.get(name)
            cell = "N/A" if v is None else str(v)
            row += cell.ljust(col_w)
        row += _STATUS["match"] if len(unique) <= 1 else _STATUS["warn"]
        lines.append(row)
    return "\n".join(lines)


def _format_issues(issues: List[Issue]) -> str:
    lines = []
    for issue in issues:
        prefix = f"[{issue.severity}] "
        wrapped = textwrap.fill(
            issue.message,
            width=78,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        )
        lines.append(wrapped)
        lines.append("")
    return "\n".join(lines)


def _verdict(issues: List[Issue]) -> Tuple[str, int]:
    severities = {i.severity for i in issues}
    if "ERROR" in severities:
        return "MERGE WILL FAIL", 1
    if "WARNING" in severities:
        return "MERGE POSSIBLE WITH WARNINGS", 0
    return "MERGE LOOKS COMPATIBLE", 0


@click.command("mergekit-check-compat", cls=PrettyPrintHelp)
@click.argument("config_file")
@add_merge_options
def main(config_file: str, merge_options: MergeOptions):
    """Check model compatibility before merging.

    Loads configs and tokenizers (no weights) for all models in CONFIG_FILE and
    reports structural mismatches, RoPE incompatibilities, vocabulary differences,
    and FIM/chat-template issues before any merge work begins.
    """
    merge_options.apply_global_options()

    with open(config_file, "r", encoding="utf-8") as f:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(f))

    models = merge_config.referenced_models()
    if not models:
        print("No models found in config.")
        sys.exit(1)

    names = [_short_name(ref) for ref in models]
    seen: Dict[str, int] = {}
    for i, n in enumerate(names):
        count = seen.get(n, 0) + 1
        seen[n] = count
        if count > 1:
            names[i] = f"{n}_{count}"

    trc = merge_options.trust_remote_code
    print("Loading model configs and tokenizers (no weights)...")
    arch_configs: Dict[str, Any] = {}
    tokenizers: Dict[str, Any] = {}
    for ref, name in zip(models, names):
        try:
            arch_configs[name] = ref.config(trust_remote_code=trc)
        except Exception as exc:
            print(f"ERROR: Could not load config for {name}: {exc}")
            sys.exit(1)
        try:
            tokenizers[name] = AutoTokenizer.from_pretrained(
                ref.model.path,
                revision=ref.model.revision,
                trust_remote_code=trc,
            )
        except Exception as exc:
            print(f"ERROR: Could not load tokenizer for {name}: {exc}")
            sys.exit(1)

    all_rows: Dict[str, Dict] = {}
    all_issues: List[Issue] = []

    for fn in (check_architecture, check_rope):
        rows, issues = fn(arch_configs)
        all_rows.update(rows)
        all_issues.extend(issues)

    for fn in (check_vocab, check_fim_tokens, check_chat_template):
        rows, issues = fn(tokenizers)
        all_rows.update(rows)
        all_issues.extend(issues)

    print("\nModel Compatibility Report")
    print("=" * 60)
    print(_format_table(all_rows, names))

    if all_issues:
        print("\nIssues")
        print("-" * 40)
        print(_format_issues(all_issues))

    verdict, exit_code = _verdict(all_issues)
    print(f"Verdict: {verdict}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
