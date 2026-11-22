"""The run directory: one production setup, one prompt set, every stage's output and the record of what made it.

    <run>/run.json          the setup (written by `init`, read by every command)
    <run>/prompts/          the prompt sets as the fit uses them (sets.json)
    <run>/capture/          recordings, one per prompt, named by fingerprint
    <run>/direction/        the base sidecar (per-layer refusal directions, no cut)
    <run>/fit/<study>/      the study ledger (sqlite), one directory per trial
    <run>/candidates/<name> pinned trials
    <run>/eval/             evaluation results; the stock model's are cached per setup
    <run>/export/<name>     a shareable sidecar
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .production import Production

SCHEMA = "infernix-ablate-run-1"
TOOL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOL_DIR.parent.parent


@dataclass
class Run:
    dir: Path
    production: Production
    serve_binary: Path
    capture_binary: Path | None
    context: int = 16384
    created: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.dir / "run.json"

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {"schema": SCHEMA, "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "serve_binary": str(self.serve_binary),
                "capture_binary": str(self.capture_binary) if self.capture_binary else None,
                "context": self.context, "production": self.production.to_json(), "notes": self.notes}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    @classmethod
    def load(cls, run_dir: Path) -> Run:
        path = Path(run_dir) / "run.json"
        if not path.is_file():
            raise SystemExit(f"{run_dir} is not a run directory (no run.json); start one with `ablate.py init`")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: schema {data.get('schema')!r}, this tool reads {SCHEMA}")
        return cls(dir=Path(run_dir).resolve(), production=Production.from_json(data["production"]),
                   serve_binary=Path(data["serve_binary"]),
                   capture_binary=Path(data["capture_binary"]) if data.get("capture_binary") else None,
                   context=int(data.get("context", 16384)), created=data.get("created", ""),
                   notes=data.get("notes") or {})

    def sub(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def sha256_file(path: Path, limit: int | None = None) -> str:
    """sha256 of a file; with limit, of its first `limit` bytes plus its size (artifacts are tens of GB)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        if limit is None:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
            return h.hexdigest()
        h.update(fh.read(limit))
    h.update(str(Path(path).stat().st_size).encode())
    return f"head{limit}+size:{h.hexdigest()[:16]}"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def as_dict(obj) -> dict:
    return asdict(obj)
