"""Record the source facts that M0 of the Flash-Next design must confirm before recipes freeze.

    python3 -m tools.flash_next.inspect_checkpoint /path/to/Qwen3.8-Flash-Next-NVFP4 \
        --json out/flash_next_source_facts.json

Reads ``config.json``, ``hf_quant_config.json`` and the Safetensors headers, plus the FP32 scalar
scales of every quantized expert matrix (a few hundred KiB); bulk tensor data is never read.
It answers docs/maintainer/qwen3_8-flash-next-design.md §6.1 and §19 M0:

- the declared quantization (algorithm, group size, exclusions, per-layer overrides);
- whether every NVFP4 expert matrix has exactly one ``weight_scale_2`` and one ``input_scale``;
- whether gate and up share ``input_scale`` (one A4 quantization of x per expert, or two);
- how many ``weight_scale_2`` are powers of two (the only case Infernix's divisor ``nvfp4`` is exact);
- the MTP experts' storage and activation scheme (FP8 blocks with or without ``input_scale``);
- the PLE n-gram shards' dtype and scale shape;
- bytes per tensor class and dtype, to check the §4 byte budget.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from math import prod
from pathlib import Path
import re
import struct
import sys

from tools.convert.sources.safetensors import SafetensorsSource

_EXPERT = re.compile(
    r"^(?P<bank>.*\.experts)\.(?P<expert>\d+)\.(?P<role>gate|up|down)_proj\.(?P<suffix>[A-Za-z0-9_]+)$"
)
_INDEX = re.compile(r"\.\d+\.")
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "I8": 1, "U8": 1, "F8_E4M3": 1}


def tensor_class(name: str) -> str:
    """The name with layer, expert and shard indices replaced by ``*``."""

    previous = None
    while previous != name:
        previous, name = name, _INDEX.sub(".*.", name)
    return re.sub(r"shard_\d+", "shard_*", name)


def _scalar_word(store: SafetensorsSource, name: str) -> tuple[int | None, str]:
    """The FP32 word of a scalar scale, or None with the reason it is not one."""

    info = store.describe(name)
    if prod(info.shape) != 1:
        return None, f"shape {list(info.shape)}"
    if info.dtype != "F32":
        return None, f"dtype {info.dtype}"
    raw = store.read_flat(name).numpy().tobytes()
    return struct.unpack("<I", raw)[0], ""


def _word_value(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def _is_power_of_two(word: int) -> bool:
    exponent = (word >> 23) & 0xFF
    return (word & 0x7FFFFF) == 0 and 0 < exponent < 0xFF and not (word >> 31)


def _quant_config(root: Path, config: dict) -> dict:
    path = root / "hf_quant_config.json"
    if path.is_file():
        return {"source": "hf_quant_config.json", **json.loads(path.read_text())}
    for owner in (config, config.get("text_config", {})):
        if "quantization_config" in owner:
            return {"source": "config.json", "quantization": owner["quantization_config"]}
    return {"source": None}


def _model_shape(config: dict) -> dict:
    text = config.get("text_config", config)
    keys = (
        "architectures", "model_type", "num_hidden_layers", "hidden_size", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
        "max_position_embeddings", "mtp_num_hidden_layers", "layer_types",
        # PLE n-gram hash (row ids depend on every one of these)
        "ngram_size", "heads_per_ngram", "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by",
        "seed", "eos_token_id",
    )
    found = {k: (config if k in config else text)[k] for k in keys if k in config or k in text}
    if "layer_types" in found:
        counts: dict[str, int] = defaultdict(int)
        for kind in found["layer_types"]:
            counts[kind] += 1
        found["layer_types"] = dict(counts)
    return found


def inspect(path: str | Path) -> dict:
    with SafetensorsSource(path) as store:
        config = store.config
        names = sorted(store.weight_map)
        classes: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        banks: dict[str, dict[tuple[int, str], dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for name in names:
            info = store.describe(name)
            entry = classes[(tensor_class(name), info.dtype)]
            entry[0] += 1
            entry[1] += info.bytes
            match = _EXPERT.match(name)
            if match:
                key = (int(match["expert"]), match["role"])
                banks[match["bank"]][key][match["suffix"]] = name

        bank_facts = {}
        for bank, matrices in sorted(banks.items()):
            bank_facts[bank] = _bank_facts(store, matrices)

        ngram = {
            name: {"dtype": store.describe(name).dtype, "shape": list(store.describe(name).shape)}
            for name in names
            if "ngram" in name and not _EXPERT.match(name)
        }
        ngram_summary = defaultdict(lambda: {"count": 0, "bytes": 0, "dtypes": set(), "shapes": set()})
        for name, info in ngram.items():
            s = ngram_summary[tensor_class(name)]
            s["count"] += 1
            s["bytes"] += store.describe(name).bytes
            s["dtypes"].add(info["dtype"])
            s["shapes"].add(tuple(info["shape"]))

        return {
            "path": str(path),
            "model": _model_shape(config),
            "quantization": _quant_config(store.root, config),
            "tensors": len(names),
            "bytes": sum(v[1] for v in classes.values()),
            "classes": [
                {"class": c, "dtype": d, "count": v[0], "bytes": v[1]}
                for (c, d), v in sorted(classes.items(), key=lambda kv: -kv[1][1])
            ],
            "expert_banks": bank_facts,
            "ngram": {
                k: {"count": v["count"], "bytes": v["bytes"], "dtypes": sorted(v["dtypes"]),
                    "shapes": sorted(list(s) for s in v["shapes"])}
                for k, v in sorted(ngram_summary.items())
            },
        }


def _bank_facts(store: SafetensorsSource, matrices: dict[tuple[int, str], dict[str, str]]) -> dict:
    experts = sorted({e for e, _ in matrices})
    suffix_sets: dict[str, int] = defaultdict(int)
    dtypes: dict[str, set[str]] = defaultdict(set)
    for leaves in matrices.values():
        suffix_sets[",".join(sorted(leaves))] += 1
        for suffix, name in leaves.items():
            dtypes[suffix].add(store.describe(name).dtype)
    facts: dict = {
        "experts": len(experts),
        "expert_ids_contiguous": experts == list(range(len(experts))),
        "matrix_suffix_sets": dict(suffix_sets),
        "dtypes": {k: sorted(v) for k, v in sorted(dtypes.items())},
    }
    scalar_suffixes = [s for s in ("weight_scale_2", "input_scale") if s in dtypes]
    for suffix in scalar_suffixes:
        words, problems = {}, defaultdict(int)
        for (expert, role), leaves in matrices.items():
            if suffix not in leaves:
                problems["missing"] += 1
                continue
            word, reason = _scalar_word(store, leaves[suffix])
            if word is None:
                problems[reason] += 1
            else:
                words[(expert, role)] = word
        values = [_word_value(w) for w in words.values()]
        facts[suffix] = {
            "scalar_fp32_matrices": len(words),
            "non_scalar_or_non_fp32": dict(problems),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "powers_of_two": sum(_is_power_of_two(w) for w in words.values()),
            "nonpositive_or_nonfinite": sum(not (v > 0 and v < float("inf")) for v in values),
        }
        if suffix == "input_scale":
            shared = differ = 0
            examples = []
            for expert in experts:
                g, u = words.get((expert, "gate")), words.get((expert, "up"))
                if g is None or u is None:
                    continue
                if g == u:
                    shared += 1
                else:
                    differ += 1
                    if len(examples) < 5:
                        examples.append({"expert": expert, "gate": _word_value(g), "up": _word_value(u)})
            facts["gate_up_input_scale"] = {"equal_words": shared, "different": differ, "examples": examples}
    return facts


def _summary(facts: dict) -> str:
    lines = [f"{facts['path']}: {facts['tensors']} tensors, {facts['bytes'] / 1e9:.2f} GB"]
    q = facts["quantization"]
    lines.append(f"quantization source: {q.get('source')}")
    if "quantization" in q:
        qq = q["quantization"]
        lines.append("  " + json.dumps({k: qq[k] for k in qq if k not in ("exclude_modules", "quantized_layers")}))
        if "exclude_modules" in qq:
            lines.append(f"  exclude_modules ({len(qq['exclude_modules'])}): {qq['exclude_modules'][:12]}")
        if "quantized_layers" in qq:
            algos: dict[str, int] = defaultdict(int)
            for layer in qq["quantized_layers"].values():
                algos[json.dumps(layer, sort_keys=True)] += 1
            lines.append(f"  quantized_layers schemes: {dict(algos)}")
    groups: dict[str, list[dict]] = defaultdict(list)
    for bank, b in facts["expert_banks"].items():
        groups[tensor_class(bank + ".")].append(b)
    for cls, banks in groups.items():
        experts = sorted({b["experts"] for b in banks})
        dtypes = {k: sorted({d for b in banks for d in b["dtypes"].get(k, [])}) for k in banks[0]["dtypes"]}
        lines.append(f"banks {cls.rstrip('.')}: {len(banks)} banks of {experts} experts, dtypes {dtypes}")
        for suffix in ("weight_scale_2", "input_scale"):
            stats = [b[suffix] for b in banks if suffix in b]
            if not stats:
                continue
            other: dict[str, int] = defaultdict(int)
            for st in stats:
                for k, v in st["non_scalar_or_non_fp32"].items():
                    other[k] += v
            lines.append(
                f"  {suffix}: {sum(st['scalar_fp32_matrices'] for st in stats)} scalar FP32, other {dict(other)}, "
                f"range [{min(st['min'] for st in stats)}, {max(st['max'] for st in stats)}], "
                f"powers of two {sum(st['powers_of_two'] for st in stats)}"
            )
        shared = [b["gate_up_input_scale"] for b in banks if "gate_up_input_scale" in b]
        if shared:
            lines.append(f"  gate/up input_scale equal {sum(g['equal_words'] for g in shared)}, "
                         f"different {sum(g['different'] for g in shared)}")
    for cls, n in facts["ngram"].items():
        lines.append(f"ngram {cls}: {n['count']} x {n['dtypes']} {n['shapes'][:2]}, {n['bytes'] / 1e9:.2f} GB")
    lines.append("largest classes:")
    for c in facts["classes"][:25]:
        lines.append(f"  {c['bytes'] / 1e9:9.3f} GB  {c['dtype']:8} x{c['count']:<6} {c['class']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="checkpoint directory (or index/safetensors file)")
    parser.add_argument("--json", type=Path, help="write the full facts as JSON")
    args = parser.parse_args(argv)
    facts = inspect(args.checkpoint)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(facts, indent=1) + "\n")
    print(_summary(facts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
