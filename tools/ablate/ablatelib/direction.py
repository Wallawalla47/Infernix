"""Direction: the refusal direction at every layer, from the capture's recordings.

heretic's recipe (projected abliteration): the hidden state each layer hands on at the LAST prompt token (where
the answer starts, after the template's empty thinking block; capture.py); mean(harmful) - mean(harmless) per layer;
orthogonalized against the harmless mean (heretic's projected abliteration); one unit row per layer.

The output is a base sidecar with no cut. The fit chooses, every trial, which direction to apply (one layer's row,
interpolated, at every layer; or each layer's own) and how hard (per-lane bands).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from . import sidecar as SC


def recording(path: Path, n_layers: int, hidden: int) -> np.ndarray:
    """One prompt's [layers, hidden] hidden states."""
    want = n_layers * hidden * 4
    if not path.is_file() or path.stat().st_size != want:
        raise SystemExit(f"{path.name}: missing, or not the {want} bytes the capture's shape needs: a truncated or "
                         "differently shaped recording would silently bias the means; rerun `capture`")
    return np.fromfile(path, dtype="<f4").reshape(n_layers, hidden)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise SystemExit("a layer's direction has zero norm: the two classes are indistinguishable there, which "
                         "usually means a capture or labelling fault")
    return (v / n).astype(np.float32)


def build(run, index: dict, out_dir: Path, *, orthogonalize: bool = True, source: str = "") -> tuple[Path, dict]:
    cap = run.dir / "capture"
    if not index.get("shape"):
        raise SystemExit("capture/index.json records no model shape: run `capture`")
    n_layers, hidden = int(index["shape"]["layers"]), int(index["shape"]["hidden"])
    classes: dict[str, list[np.ndarray]] = {"harmful": [], "harmless": []}
    for key, entry in index["prompts"].items():
        kind = key.split(":", 1)[0]
        classes[kind].append(recording(cap / f"{entry['fingerprint']}.f32", n_layers, hidden))
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
    recipe = ("the hidden state each layer hands on at the last prompt token, "
              + ("orthogonalized against the harmless mean (heretic's projected abliteration), " if orthogonalize else "")
              + f"{len(bad)} harmful / {len(good)} harmless prompts")
    path = SC.write_base(out_dir, rows, source=source or recipe, notes=f"direction: {recipe}")
    digest = hashlib.sha256((out_dir / "d.f32.bin").read_bytes()).hexdigest()
    report = {"d_sha256": digest, "recipe": recipe, "harmful": len(bad), "harmless": len(good),
              "raw_norm_per_layer": [round(float(np.linalg.norm(r)), 5) for r in diff]}
    return path, report
