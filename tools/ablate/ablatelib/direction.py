"""Direction: the refusal direction at every layer, from the capture's recordings.

heretic's recipe, which produced refusal-cut-v1 (the defaults): the LAST prompt token (where the answer starts, after
the template's empty thinking block); the hidden state each layer hands the next (Flash-Next has several
hyper-connection residual streams, so the closest stand-in for heretic's per-layer hidden state is what the next layer
reads: layer L+1's attention input, and for the last layer its own MoE input); mean(harmful) - mean(harmless) per
layer; orthogonalized against the harmless mean (heretic's projected abliteration); one unit row per layer.

The output is a base sidecar with no cut. The fit chooses, every trial, which direction to apply (one layer's row,
interpolated, at every layer; or each layer's own) and how hard (per-lane bands).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from . import sidecar as SC

TAP = {"mixer_in": 0, "mixer_out": 1, "moe_in": 2, "moe_out": 3}


def bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def layer_vectors(path: Path, n_layers: int, hidden: int, n_tokens: int, measure_at: str, token: str) -> np.ndarray:
    """One [layers, hidden] vector for one prompt's recording."""
    want = n_layers * len(TAP) * n_tokens * hidden * 2
    if path.stat().st_size != want:
        raise SystemExit(f"{path.name}: {path.stat().st_size} bytes, expected {want}: a truncated or differently "
                         "shaped recording would silently bias the means")
    view = bf16_to_f32(path.read_bytes()).reshape(n_layers, len(TAP), n_tokens, hidden)
    if measure_at == "next_layer_input":
        x = np.concatenate([view[1:, TAP["mixer_in"]], view[-1:, TAP["moe_in"]]], axis=0)
    elif measure_at == "lane_output":
        x = view[:, TAP["mixer_out"]]
    else:
        raise SystemExit(f"unknown measure_at {measure_at!r}")
    # x is [layers, tokens, hidden]: the token axis is 1. Averaging axis 2 instead returns a plausible-looking
    # [layers, tokens] non-direction.
    out = x[:, -1, :] if token == "last" else x.mean(axis=1)
    return np.ascontiguousarray(out, dtype=np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise SystemExit("a layer's direction has zero norm: the two classes are indistinguishable there, which "
                         "usually means a capture or labelling fault")
    return (v / n).astype(np.float32)


def build(run, index: dict, out_dir: Path, *, measure_at: str = "next_layer_input", token: str = "last",
          orthogonalize: bool = True, source: str = "") -> tuple[Path, dict]:
    cap = run.dir / "capture"
    if "shape" not in index:
        raise SystemExit("capture/index.json records no model shape (written by an older version): rerun `capture`, "
                         "which keeps every recording and rewrites the index")
    n_layers, hidden = int(index["shape"]["layers"]), int(index["shape"]["hidden"])
    classes: dict[str, list[np.ndarray]] = {"harmful": [], "harmless": []}
    for key, entry in index["prompts"].items():
        kind = key.split(":", 1)[0]
        classes[kind].append(layer_vectors(cap / f"{entry['fingerprint']}.blocks.bin", n_layers, hidden,
                                           int(entry["n_tokens"]), measure_at, token))
    if min(len(v) for v in classes.values()) < 20:
        raise SystemExit(f"too few prompts for a direction (harmful {len(classes['harmful'])}, "
                         f"harmless {len(classes['harmless'])}); the docs suggest 150-200 per list")
    bad, good = np.stack(classes["harmful"]), np.stack(classes["harmless"])
    diff = bad.mean(axis=0) - good.mean(axis=0)
    if orthogonalize:
        gm = good.mean(axis=0)
        gm = gm / np.maximum(np.linalg.norm(gm, axis=1, keepdims=True), 1e-12)
        diff = diff - (diff * gm).sum(axis=1, keepdims=True) * gm
    rows = np.stack([unit(r) for r in diff])
    recipe = (f"measure_at={measure_at}, token={token}, "
              + ("orthogonalized against the harmless mean (heretic's projected abliteration), " if orthogonalize else "")
              + f"{len(bad)} harmful / {len(good)} harmless prompts")
    path = SC.write_base(out_dir, rows, source=source or recipe, notes=f"direction: {recipe}")
    digest = hashlib.sha256((out_dir / "d.f32.bin").read_bytes()).hexdigest()
    report = {"d_sha256": digest, "recipe": recipe, "harmful": len(bad), "harmless": len(good),
              "raw_norm_per_layer": [round(float(np.linalg.norm(r)), 5) for r in diff]}
    return path, report
