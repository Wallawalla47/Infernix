"""Public prompt sets, downloaded at pinned revisions and converted into the lists `prompts` and `grade` read.

A preset gives a fit pair (harmful prompts the model refuses, harmless ones it answers) and a test pair the fit never
sees (JailbreakBench's harmful and benign behaviors). Every source is pinned to a commit and checked against its
sha256, so a preset always gives the same lists; each prompt is stripped, prompts with line breaks are dropped (the
tool refuses them), repeats are dropped, and a test prompt that is also a fit prompt is dropped from the test list
(the sets borrow from each other: JBB-Behaviors takes some behaviors from AdvBench and HarmBench).

The licences are the sets' own (from each repository): check them before sharing a sidecar fitted on them. Alpaca's
data is CC BY-NC 4.0 and its README limits it to research use.

    ablate.py datasets list [--json]
    ablate.py datasets fetch PRESET --out DIR        writes DIR/PRESET/{harmful,harmless,test-harmful,test-benign}.txt
                                                      and DIR/PRESET/sets.json (sources, licences, counts)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .run import write_json


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    sha256: str
    licence: str
    licence_note: str          # what the licence asks of you, in a few words
    home: str                  # where to read about it
    column: str                # the prompt's column (a CSV column, or a JSON object's key)
    keep: dict = field(default_factory=dict)   # rows kept: column -> required value (CSV); for JSON, keys that must be empty


SOURCES: dict[str, Source] = {
    "advbench": Source(
        "AdvBench harmful behaviors (520)",
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/098262edf85f807224e70ecd87b9d83716bf6b73/"
        "data/advbench/harmful_behaviors.csv",
        "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
        "MIT", "keep the copyright notice", "https://github.com/llm-attacks/llm-attacks", "goal"),
    "harmbench": Source(
        "HarmBench standard behaviors (200 of 400)",
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/8e1604d1171fe8a48d8febecd22f600e462bdcdd/"
        "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
        "MIT", "keep the copyright notice", "https://github.com/centerforaisafety/HarmBench", "Behavior",
        {"FunctionalCategory": "standard"}),
    "jbb-harmful": Source(
        "JailbreakBench JBB-Behaviors, harmful (100)",
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/886acc352a31533ffbcf4ef22c744658688086fc/"
        "data/harmful-behaviors.csv",
        "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
        "MIT", "keep the copyright notice", "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors", "Goal"),
    "jbb-benign": Source(
        "JailbreakBench JBB-Behaviors, benign (100)",
        "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/886acc352a31533ffbcf4ef22c744658688086fc/"
        "data/benign-behaviors.csv",
        "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
        "MIT", "keep the copyright notice", "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors", "Goal"),
    "alpaca": Source(
        "Stanford Alpaca instructions without an input",
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/761dc5bfbdeeffa89b8bff5d038781a4055f796a/"
        "alpaca_data.json",
        "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
        "CC BY-NC 4.0", "non-commercial use only; its README limits it to research use",
        "https://github.com/tatsu-lab/stanford_alpaca", "instruction", {"input": ""}),
}


@dataclass(frozen=True)
class Preset:
    title: str
    description: str
    harmful: tuple[str, int]        # (source, prompts kept)
    harmless: tuple[str, int]
    test_harmful: str = "jbb-harmful"
    test_benign: str = "jbb-benign"


PRESETS: dict[str, Preset] = {
    "advbench-alpaca": Preset(
        "AdvBench + Alpaca (heretic's defaults)",
        "heretic's default prompts from their sources: 400 AdvBench behaviors against 400 Alpaca instructions; "
        "JBB-Behaviors as the test pair.",
        ("advbench", 400), ("alpaca", 400)),
    "harmbench-alpaca": Preset(
        "HarmBench + Alpaca",
        "HarmBench's 200 standard behaviors (refusal-cut-v1's harmful prompts) against 200 Alpaca instructions; "
        "JBB-Behaviors as the test pair.",
        ("harmbench", 200), ("alpaca", 200)),
}


def download(url: str, timeout: float = 300.0) -> bytes:
    request = urllib.request.Request(url, headers={"user-agent": "infernix-ablate"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return reply.read()


def fetch_source(name: str, cache: Path, fetch=download) -> bytes:
    """A source's bytes, from the cache or downloaded, checked against its pinned sha256."""
    source = SOURCES[name]
    path = cache / f"{name}{Path(source.url).suffix}"
    data = path.read_bytes() if path.is_file() else None
    if data is None or hashlib.sha256(data).hexdigest() != source.sha256:
        data = fetch(source.url)
        digest = hashlib.sha256(data).hexdigest()
        if digest != source.sha256:
            raise SystemExit(f"{source.url}: sha256 {digest}, expected {source.sha256} (the pinned revision changed?)")
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def prompts_of(name: str, data: bytes) -> list[str]:
    """A source's usable prompts in file order: kept rows, stripped, one line each, each once."""
    source = SOURCES[name]
    if source.url.endswith(".json"):
        rows = [row for row in json.loads(data) if all(not row.get(key) for key in source.keep)]
    else:
        rows = [row for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
                if all(row.get(column) == value for column, value in source.keep.items())]
    seen: set[str] = set()
    out = []
    for row in rows:
        text = str(row.get(source.column) or "").strip()
        if text and "\n" not in text and "\r" not in text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def fetch_preset(name: str, out: Path, cache: Path | None = None, fetch=download, log=print) -> Path:
    """Downloads and converts a preset into out/name; returns that folder."""
    if name not in PRESETS:
        raise SystemExit(f"unknown prompt set {name!r} (known: {', '.join(PRESETS)})")
    preset = PRESETS[name]
    cache = cache or out / ".cache"
    lists: dict[str, list[str]] = {}
    for part, (source, count) in (("harmful", preset.harmful), ("harmless", preset.harmless)):
        prompts = prompts_of(source, fetch_source(source, cache, fetch))
        if len(prompts) < count:
            raise SystemExit(f"{source}: {len(prompts)} usable prompts, the preset takes {count}")
        lists[part] = prompts[:count]
    fit = set(lists["harmful"]) | set(lists["harmless"])
    dropped: dict[str, int] = {}
    for part, source in (("test-harmful", preset.test_harmful), ("test-benign", preset.test_benign)):
        prompts = prompts_of(source, fetch_source(source, cache, fetch))
        lists[part] = [p for p in prompts if p not in fit]
        dropped[part] = len(prompts) - len(lists[part])
    # A benign test prompt that is also a harmful one (or the reverse) cannot grade both ways.
    both = set(lists["test-harmful"]) & set(lists["test-benign"])
    lists["test-benign"] = [p for p in lists["test-benign"] if p not in both]
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    for part, prompts in lists.items():
        (folder / f"{part}.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    used = sorted({preset.harmful[0], preset.harmless[0], preset.test_harmful, preset.test_benign})
    write_json(folder / "sets.json", {
        "preset": name, "title": preset.title, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {part: len(prompts) for part, prompts in lists.items()},
        "dropped_from_test_as_fit_prompts": dropped,
        "sources": {s: {"title": SOURCES[s].title, "url": SOURCES[s].url, "sha256": SOURCES[s].sha256,
                        "licence": SOURCES[s].licence, "licence_note": SOURCES[s].licence_note,
                        "home": SOURCES[s].home} for s in used},
        "source_text": describe(name)})
    log(f"{name}: harmful {len(lists['harmful'])}, harmless {len(lists['harmless'])}, test harmful "
        f"{len(lists['test-harmful'])}, test benign {len(lists['test-benign'])} (in {folder})")
    return folder


def describe(name: str) -> str:
    """A preset in words, with its licences: a sidecar's generation.source."""
    preset = PRESETS[name]
    h, g = SOURCES[preset.harmful[0]], SOURCES[preset.harmless[0]]
    return (f"{h.title.split(' (')[0]} ({h.licence}), {preset.harmful[1]} prompts, against "
            f"{g.title.split(' (')[0]} ({g.licence}), {preset.harmless[1]} prompts")


def catalogue() -> list[dict]:
    """The presets with their sources and licences (the launcher's Ablation page lists this)."""
    out = []
    for name, preset in PRESETS.items():
        used = [preset.harmful[0], preset.harmless[0], preset.test_harmful, preset.test_benign]
        out.append({"name": name, "title": preset.title, "description": preset.description,
                    "harmful": {"source": preset.harmful[0], "count": preset.harmful[1]},
                    "harmless": {"source": preset.harmless[0], "count": preset.harmless[1]},
                    "test": {"harmful": preset.test_harmful, "benign": preset.test_benign},
                    "sources": [{"name": s, "title": SOURCES[s].title, "licence": SOURCES[s].licence,
                                 "licence_note": SOURCES[s].licence_note, "home": SOURCES[s].home}
                                for s in dict.fromkeys(used)]})
    return out
