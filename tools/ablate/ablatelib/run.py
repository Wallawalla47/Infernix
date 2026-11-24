"""The run directory: one production setup, one prompt set, every stage's output and the record of what made it.

    <run>/run.json          the setup (written by `init`, read by every command)
    <run>/prompts/          the prompt sets as the fit uses them (sets.json)
    <run>/capture/          each direction prompt's hidden states, one recording per prompt, named by fingerprint
    <run>/direction/        the base sidecar (per-layer refusal directions, no cut)
    <run>/fit/<study>/      the study ledger (sqlite), one directory per trial
    <run>/candidates/<name> pinned trials
    <run>/eval/             evaluation results; the stock model's are cached per setup
    <run>/export/<name>     a shareable sidecar
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .production import Production

SCHEMA = "infernix-ablate-run-2"
TOOL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOL_DIR.parent.parent


@dataclass
class Run:
    dir: Path
    production: Production
    serve_binary: Path
    context: int = 16384
    created: str = ""
    notes: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.dir / "run.json"

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {"schema": SCHEMA, "created": self.created or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "serve_binary": str(self.serve_binary), "context": self.context, "production": self.production.to_json(), "notes": self.notes}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        replace_file(tmp, self.path)

    @classmethod
    def load(cls, run_dir: Path) -> Run:
        path = Path(run_dir) / "run.json"
        if not path.is_file():
            raise SystemExit(f"{run_dir} is not a run directory (no run.json); start one with `ablate.py init`")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: schema {data.get('schema')!r}, this tool reads {SCHEMA}")
        return cls(dir=Path(run_dir).resolve(), production=Production.from_json(data["production"]),
                   serve_binary=Path(data["serve_binary"]), context=int(data.get("context", 16384)), created=data.get("created", ""),
                   notes=data.get("notes") or {})

    def sub(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def files_stamp(value: str | None) -> list | None:
    """The size and modification time of the file an option names, or of each file directly in the folder it names (a
    LoRA adapter's PEFT folder): an option's string says where its file is, not what it holds, so a file replaced in
    place is another setup. None when the value names no existing path."""
    if not value:
        return None
    path = Path(value)
    try:
        if path.is_file():
            st = path.stat()
            return [path.name, st.st_size, st.st_mtime_ns]
        if path.is_dir():
            return sorted([f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in path.iterdir() if f.is_file())
    except OSError:
        return None
    return None


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


# A v3 entry file's header (tools/artifact/framing.py): magic, directory JSON length, artifact id.
ENTRY_HEADER = struct.Struct("<8sQ16s")
ENTRY_MAGIC = b"NINFER\x00\x03"


def artifact_id(artifact: Path) -> str:
    """The artifact's id as the engine prints it (32 lowercase hex digits), from its entry file's header: what a
    sidecar's artifact_id names, so a start on another artifact warns."""
    with Path(artifact).open("rb") as fh:
        raw = fh.read(ENTRY_HEADER.size)
    if len(raw) != ENTRY_HEADER.size or ENTRY_HEADER.unpack(raw)[0] != ENTRY_MAGIC:
        raise SystemExit(f"{artifact}: not an Infernix v3 artifact (its entry header does not read)")
    return ENTRY_HEADER.unpack(raw)[2].hex()


def text_model_type(artifact: Path) -> str:
    """The model_type of the artifact's text component (its directory JSON, after the entry header), or "" when it
    cannot be read: tells a fit which model's rendering rules apply, nothing more."""
    try:
        with Path(artifact).open("rb") as fh:
            magic, length, _ = ENTRY_HEADER.unpack(fh.read(ENTRY_HEADER.size))
            if magic != ENTRY_MAGIC or length > (1 << 30):
                return ""
            directory = json.loads(fh.read(length).decode("utf-8"))
        return str(directory["components"]["text"]["config"]["model_type"])
    except (OSError, ValueError, KeyError, TypeError, struct.error):
        return ""


def replace_file(tmp: Path, path: Path) -> None:
    """tmp.replace(path), tried again for a few seconds while Windows refuses it: a rename over a file another program
    has open fails there (the launcher reads progress.json every second; a virus scanner opens new files)."""
    for attempt in range(60):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(0.05)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    replace_file(tmp, path)


def as_dict(obj) -> dict:
    return asdict(obj)
