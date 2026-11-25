"""v2 sidecars: the base direction, one trial's cut, and the export a user shares.

Band shape (heretic's, per lane): a peak position, a peak weight, a reach and an edge weight; the per-layer alpha
falls linearly from the peak to the edge weight at `reach` layers away, and is 0 beyond. Two of heretic's rules are
load-bearing: every parameter is sampled every trial, and the edge weight is sampled as a fraction of the peak
(multivariate TPE refuses conditional or variable-range parameters). Layers outside a band are absent from the
table, which the engine reads as alpha 0.

heretic's direction scope is sampled every trial too: 'global' applies one direction, interpolated between two
neighbouring layers' rows at direction_index (0.4-0.9 of the depth), at every layer; 'per layer' cuts each layer
along its own row.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "ablate-direction-v2"
LANES = ("attn", "mlp")
ALPHA_MAX = 4.0      # the engine's kAlphaMax


@dataclass(frozen=True)
class Band:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float

    def alpha_at(self, layer: int) -> float:
        distance = abs(layer - self.max_weight_position)
        if distance > self.min_weight_distance:
            return 0.0
        if self.min_weight_distance == 0.0:
            return self.max_weight
        return self.max_weight + (distance / self.min_weight_distance) * (self.min_weight - self.max_weight)


def search_space(n_layers: int) -> dict[str, tuple[float, float]]:
    """heretic's ranges for every sampled parameter: the direction index and each lane's band."""
    last = n_layers - 1
    space = {"direction_index": (0.4 * last, 0.9 * last)}
    for lane in LANES:
        space.update({f"{lane}.max_weight": (0.8, 1.5), f"{lane}.max_weight_position": (0.6 * last, 1.0 * last),
                      f"{lane}.min_weight": (0.0, 1.0), f"{lane}.min_weight_distance": (1.0, 0.6 * last)})
    return space


def sample_band(trial, n_layers: int, lane: str) -> Band:
    space = search_space(n_layers)
    max_weight = trial.suggest_float(f"{lane}.max_weight", *space[f"{lane}.max_weight"])
    position = trial.suggest_float(f"{lane}.max_weight_position", *space[f"{lane}.max_weight_position"])
    min_weight = trial.suggest_float(f"{lane}.min_weight", *space[f"{lane}.min_weight"]) * max_weight   # a fraction
    reach = trial.suggest_float(f"{lane}.min_weight_distance", *space[f"{lane}.min_weight_distance"])
    return Band(max_weight, position, min_weight, reach)


def lane_tables(bands: dict[str, Band], n_layers: int) -> dict[str, dict[str, float]]:
    return {f"alpha_{lane}_per_layer": {str(layer): band.alpha_at(layer) for layer in range(n_layers)
                                        if band.alpha_at(layer) > 0.0}
            for lane, band in bands.items()}


def band_summary(bands: dict[str, Band]) -> str:
    return " | ".join(f"{lane} L{b.max_weight_position:.0f} x{b.max_weight:.2f} edge {b.min_weight:.2f} "
                      f"reach {b.min_weight_distance:.0f}" for lane, b in bands.items())


def global_direction(rows: np.ndarray, direction_index: float) -> np.ndarray:
    """heretic's global direction: normalize(lerp(rows[i], rows[i+1], frac(direction_index)))."""
    weight, index = math.modf(direction_index)
    i = int(index)
    if not 0 <= i < len(rows) - 1:
        raise ValueError(f"direction_index {direction_index} outside 0..{len(rows) - 1}")
    v = rows[i] + (rows[i + 1] - rows[i]) * weight
    return (v / np.linalg.norm(v)).astype(np.float32)


def read(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("format") != FORMAT:
        raise SystemExit(f"{path}: format {cfg.get('format')!r}; this tool writes and reads {FORMAT}")
    return cfg


def rows_of(path: Path) -> np.ndarray:
    """The direction rows a sidecar names, [rows, direction_dim]."""
    cfg = read(path)
    d = np.fromfile(Path(path).parent / cfg["direction_file"], dtype=np.float32)
    return d.reshape(-1, int(cfg["direction_dim"]))


def write_base(out_dir: Path, rows: np.ndarray, source: str, notes: str) -> Path:
    """The direction step's output: per-layer unit rows and no cut (default_alpha 0, empty tables)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.astype(np.float32).tofile(out_dir / "d.f32.bin")
    cfg = {"format": FORMAT, "direction_dtype": "float32", "direction_dim": int(rows.shape[1]),
           "direction_layout": "per_layer" if rows.shape[0] > 1 else "shared", "direction_file": "d.f32.bin",
           "default_alpha": 0.0, "alpha_attn_per_layer": {}, "alpha_mlp_per_layer": {},
           "generation": {"source": source, "notes": notes}}
    (out_dir / "sidecar.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    return out_dir / "sidecar.json"


def write_trial(base: Path, out_dir: Path, tables: dict[str, dict[str, float]],
                direction: np.ndarray | None, notes: str) -> Path:
    """One candidate: the base's direction (or one global row), the trial's per-lane tables."""
    cfg = read(base)
    for field_name, table in tables.items():
        if field_name not in ("alpha_attn_per_layer", "alpha_mlp_per_layer"):
            raise ValueError(field_name)
        if any(not 0.0 <= a <= ALPHA_MAX for a in table.values()):
            raise ValueError(f"{field_name}: alpha outside [0, {ALPHA_MAX}]")
        cfg[field_name] = {k: float(v) for k, v in table.items()}
    cfg.pop("alpha_per_layer", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    if direction is None:
        shutil.copy2(Path(base).parent / cfg["direction_file"], out_dir / cfg["direction_file"])
    else:
        row = np.asarray(direction, dtype=np.float32).reshape(-1)
        if row.size != int(cfg["direction_dim"]) or abs(float(np.linalg.norm(row)) - 1.0) > 1e-3:
            raise ValueError("a global direction must be one unit row of direction_dim floats")
        cfg["direction_file"], cfg["direction_layout"] = "d.f32.bin", "shared"
        row.tofile(out_dir / "d.f32.bin")
    cfg["generation"] = dict(cfg.get("generation") or {}, notes=notes)
    (out_dir / "sidecar.json").write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    return out_dir / "sidecar.json"


def copy(src: Path, dest_dir: Path) -> Path:
    cfg = read(src)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(src).parent / cfg["direction_file"], dest_dir / cfg["direction_file"])
    shutil.copy2(src, dest_dir / "sidecar.json")
    return dest_dir / "sidecar.json"
