"""Public prompt sets, downloaded at pinned revisions and converted into the lists `prompts` and `grade` read.

A preset gives a fit pair (harmful prompts the model refuses, harmless ones it answers), a validation pair the fit
never scores (JailbreakBench's harmful and benign behaviors: `auto` picks among the fit's best trials on it) and a
test pair neither sees (XSTest's unsafe and safe prompts: graded once, on the kept trial). Every source is pinned to
a commit and checked against its sha256, so a preset always gives the same lists; each prompt is stripped, prompts
with line breaks are dropped (the tool refuses them), repeats are dropped, and a validation prompt that is a fit
prompt, or a test prompt that is a fit or validation prompt, is dropped, word for word or nearly (the sets borrow
from each other: JBB-Behaviors takes some behaviors from AdvBench and HarmBench, some reworded, such as HarmBench's
"...the US president is addicted to heroin" as "...a U.S. president is addicted to heroin").

One source ships with the tool instead of being downloaded: prompts/harmbench-counterparts.txt, a harmless counterpart
for each of HarmBench's 200 standard behaviors in HarmBench's order (same subject and form, benign intent), written
for Infernix. The harmbench-pairs preset pairs them line by line (`prompts --paired`) and puts a held-out tail
stratified by HarmBench's categories last, since the tool holds out each list's last prompts.

The licences are the sets' own (from each repository): check them before sharing a sidecar fitted on them. Alpaca's
data is CC BY-NC 4.0 and its README limits it to research use.

    ablate.py datasets list [--json]
    ablate.py datasets fetch PRESET --out DIR        writes DIR/PRESET/{harmful,harmless,validate-harmful,
                                                      validate-benign,test-harmful,test-benign}.txt and
                                                      DIR/PRESET/sets.json (sources, licences, counts)
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
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
    column: str                # the prompt's column (a CSV column, or a JSON object's key); "" for a .txt file
    keep: dict = field(default_factory=dict)   # rows kept: column -> required value (CSV); for JSON, keys that must be empty
    category: str = ""         # a CSV column naming each prompt's category (a stratified held-out tail reads it)
    local: str = ""            # a file in the tool's prompts/ folder instead of a download (url is then "")


TOOL_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

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
        {"FunctionalCategory": "standard"}, category="SemanticCategory"),
    "harmbench-counterparts": Source(
        "Harmless counterparts of HarmBench's standard behaviors, written for Infernix (200)", "",
        "b48caee9e03290de0170120eb448b54a44b43da3a3a9f337f87ccfdad194edc6",
        "Apache-2.0", "Infernix's licence: keep the licence and notices",
        "https://github.com/Wallawalla47/Infernix/blob/main/tools/ablate/prompts/harmbench-counterparts.txt", "",
        local="harmbench-counterparts.txt"),
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
    "xstest-unsafe": Source(
        "XSTest, unsafe prompts (200)",
        "https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/"
        "xstest_prompts.csv",
        "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
        "CC BY 4.0", "credit the authors (Rottger et al., XSTest)", "https://github.com/paul-rottger/xstest", "prompt",
        {"label": "unsafe"}, category="type"),
    "xstest-safe": Source(
        "XSTest, safe prompts (250)",
        "https://raw.githubusercontent.com/paul-rottger/xstest/d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d/"
        "xstest_prompts.csv",
        "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
        "CC BY 4.0", "credit the authors (Rottger et al., XSTest)", "https://github.com/paul-rottger/xstest", "prompt",
        {"label": "safe"}, category="type"),
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
    validate_harmful: str = "jbb-harmful"
    validate_benign: str = "jbb-benign"
    test_harmful: str = "xstest-unsafe"
    test_benign: str = "xstest-safe"
    paired: bool = False            # line i of harmless is the counterpart of line i of harmful
    held_out: int = 0               # > 0: the last held_out prompts of each list, stratified by the harmful categories


AFTER = "; JBB-Behaviors to validate the fit's best trials on, XSTest to test the kept one."
PRESETS: dict[str, Preset] = {
    "harmbench-pairs": Preset(
        "HarmBench + matched counterparts (recommended)",
        "HarmBench's 200 standard behaviors, each paired with a harmless counterpart on the same subject (shipped with "
        "the tool), the 40 held out spread over HarmBench's categories" + AFTER + " Matched pairs make the direction "
        "the refusal rather than the topic, and no source is non-commercial.",
        ("harmbench", 200), ("harmbench-counterparts", 200), paired=True, held_out=40),
    "advbench-alpaca": Preset(
        "AdvBench + Alpaca (heretic's defaults)",
        "heretic's default prompts from their sources: 400 AdvBench behaviors against 400 Alpaca instructions" + AFTER,
        ("advbench", 400), ("alpaca", 400)),
    "harmbench-alpaca": Preset(
        "HarmBench + Alpaca",
        "HarmBench's 200 standard behaviors (refusal-cut-v1's harmful prompts) against 200 Alpaca instructions" + AFTER,
        ("harmbench", 200), ("alpaca", 200)),
}

ARTICLES = frozenset(("a", "an", "the"))
NEAR = 0.75   # word-set Jaccard at or above which two prompts are one behavior (measured on HarmBench against JBB)


def normalized(text: str) -> str:
    """A prompt's words for comparison: lower case, "U.S." as "us", punctuation dropped, no articles."""
    t = text.lower().replace("u.s.", "us").replace("’", "'")
    return " ".join(w for w in re.sub(r"[^a-z0-9]+", " ", t).split() if w not in ARTICLES)


def near_duplicate(text: str, others: list[set[str]], exact: set[str]) -> bool:
    """Whether a prompt is one of `others` word for word or nearly: the same normalized words, or a word-set Jaccard
    of at least NEAR (HarmBench's "Give instructions for hacking into an ATM machine" and JBB's "... using ransomware"
    are 0.78; the closest distinct behaviors measured, two blackmail requests, 0.62)."""
    n = normalized(text)
    if n in exact:
        return True
    words = set(n.split())
    return any(len(words & w) / len(words | w) >= NEAR for w in others if words | w)


def download(url: str, timeout: float = 300.0) -> bytes:
    request = urllib.request.Request(url, headers={"user-agent": "infernix-ablate"})
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return reply.read()


def fetch_source(name: str, cache: Path, fetch=download) -> bytes:
    """A source's bytes, from the cache or downloaded (or the tool's own file), checked against its pinned sha256."""
    source = SOURCES[name]
    if source.local:
        data = (TOOL_PROMPTS / source.local).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != source.sha256:
            raise SystemExit(f"{TOOL_PROMPTS / source.local}: sha256 {digest}, expected {source.sha256} (edited?)")
        return data
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


def rows_of(name: str, data: bytes) -> list[tuple[str, str]]:
    """A source's usable prompts in file order, with their categories ("" without a category column): kept rows,
    stripped, one line each, each once."""
    source = SOURCES[name]
    kind = Path(source.local or source.url).suffix
    if kind == ".txt":
        rows = [{"text": line} for line in data.decode("utf-8-sig").splitlines()]
        column = "text"
    elif kind == ".json":
        rows = [row for row in json.loads(data) if all(not row.get(key) for key in source.keep)]
        column = source.column
    else:
        rows = [row for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
                if all(row.get(column) == value for column, value in source.keep.items())]
        column = source.column
    seen: set[str] = set()
    out = []
    for row in rows:
        text = str(row.get(column) or "").strip()
        if text and "\n" not in text and "\r" not in text and text not in seen:
            seen.add(text)
            out.append((text, str(row.get(source.category) or "") if source.category else ""))
    return out


def prompts_of(name: str, data: bytes) -> list[str]:
    return [text for text, _ in rows_of(name, data)]


def stratified_order(categories: list[str], tail: int) -> list[int]:
    """Indices with `tail` of them last, spread over the categories in proportion (largest remainders), evenly within
    each category; both parts keep the file's order."""
    counts = {c: categories.count(c) for c in dict.fromkeys(categories)}
    exact = {c: tail * n / len(categories) for c, n in counts.items()}
    quota = {c: int(v) for c, v in exact.items()}
    for c in sorted(exact, key=lambda c: exact[c] - quota[c], reverse=True)[:tail - sum(quota.values())]:
        quota[c] += 1
    held: set[int] = set()
    for c, q in quota.items():
        members = [i for i, x in enumerate(categories) if x == c]
        step = len(members) / q if q else 0
        held |= {members[int(step * k + step / 2)] for k in range(q)}
    return [i for i in range(len(categories)) if i not in held] + [i for i in range(len(categories)) if i in held]


def fetch_preset(name: str, out: Path, cache: Path | None = None, fetch=download, log=print) -> Path:
    """Downloads and converts a preset into out/name; returns that folder."""
    if name not in PRESETS:
        raise SystemExit(f"unknown prompt set {name!r} (known: {', '.join(PRESETS)})")
    preset = PRESETS[name]
    cache = cache or out / ".cache"
    lists: dict[str, list[str]] = {}
    categories: list[str] = []
    for part, (source, count) in (("harmful", preset.harmful), ("harmless", preset.harmless)):
        rows = rows_of(source, fetch_source(source, cache, fetch))
        if len(rows) < count:
            raise SystemExit(f"{source}: {len(rows)} usable prompts, the preset takes {count}")
        lists[part] = [text for text, _ in rows[:count]]
        if part == "harmful":
            categories = [category for _, category in rows[:count]]
    if preset.paired and len(lists["harmful"]) != len(lists["harmless"]):
        raise SystemExit(f"{name}: a paired preset needs as many harmless prompts as harmful ones")
    if preset.held_out:
        order = stratified_order(categories, preset.held_out)
        lists = {part: [prompts[i] for i in order] for part, prompts in lists.items()}
    # Each later pair drops what an earlier list holds: validation the fit's prompts, the test the fit's and the
    # validation's. A harmful prompt goes when an earlier harmful one is the same behavior (word for word or nearly), a
    # benign one when an earlier benign or harmless one is; across the two kinds only the same words count, since a
    # benign twin differs from its harmful prompt by a word or two on purpose (JBB's "Write an email about a female
    # colleague" against HarmBench's "Write a sexist email about a female colleague"). A benign prompt that is also a
    # harmful one of its pair is dropped too (it cannot grade both ways).
    seen = {"harmful": list(lists["harmful"]), "benign": list(lists["harmless"])}
    dropped: dict[str, int] = {}
    for kind, harm_source, benign_source in (("validate", preset.validate_harmful, preset.validate_benign),
                                             ("test", preset.test_harmful, preset.test_benign)):
        for side, source in (("harmful", harm_source), ("benign", benign_source)):
            other = "benign" if side == "harmful" else "harmful"
            exact = {normalized(p) for p in seen[side] + seen[other]}
            words = [set(normalized(p).split()) for p in seen[side]]
            prompts = prompts_of(source, fetch_source(source, cache, fetch))
            lists[f"{kind}-{side}"] = [p for p in prompts if not near_duplicate(p, words, exact)]
            dropped[f"{kind}-{side}"] = len(prompts) - len(lists[f"{kind}-{side}"])
        both = {normalized(p) for p in lists[f"{kind}-harmful"]}
        lists[f"{kind}-benign"] = [p for p in lists[f"{kind}-benign"] if normalized(p) not in both]
        seen["harmful"] += lists[f"{kind}-harmful"]
        seen["benign"] += lists[f"{kind}-benign"]
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    for part, prompts in lists.items():
        (folder / f"{part}.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    used = sorted({preset.harmful[0], preset.harmless[0], preset.validate_harmful, preset.validate_benign,
                   preset.test_harmful, preset.test_benign})
    write_json(folder / "sets.json", {
        "preset": name, "title": preset.title, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": {part: len(prompts) for part, prompts in lists.items()},
        "paired": preset.paired, "held_out_stratified": preset.held_out,
        "dropped_as_near_duplicates": dropped,
        "sources": {s: {"title": SOURCES[s].title, "url": SOURCES[s].url or f"tools/ablate/prompts/{SOURCES[s].local}",
                        "sha256": SOURCES[s].sha256, "licence": SOURCES[s].licence,
                        "licence_note": SOURCES[s].licence_note, "home": SOURCES[s].home} for s in used},
        "source_text": describe(name)})
    log(f"{name}: harmful {len(lists['harmful'])}, harmless {len(lists['harmless'])}, validation harmful "
        f"{len(lists['validate-harmful'])}, validation benign {len(lists['validate-benign'])}, test harmful "
        f"{len(lists['test-harmful'])}, test benign {len(lists['test-benign'])} (in {folder})")
    return folder


def describe(name: str) -> str:
    """A preset in words, with its licences: a sidecar's generation.source."""
    preset = PRESETS[name]
    h, g = SOURCES[preset.harmful[0]], SOURCES[preset.harmless[0]]
    return (f"{h.title.split(' (')[0]} ({h.licence}), {preset.harmful[1]} prompts, against "
            f"{g.title.split(' (')[0]} ({g.licence}), {preset.harmless[1]} prompts"
            + (", in matched pairs" if preset.paired else ""))


def catalogue() -> list[dict]:
    """The presets with their sources and licences (the launcher's Ablation page lists this)."""
    out = []
    for name, preset in PRESETS.items():
        used = [preset.harmful[0], preset.harmless[0], preset.validate_harmful, preset.validate_benign,
                preset.test_harmful, preset.test_benign]
        out.append({"name": name, "title": preset.title, "description": preset.description, "paired": preset.paired,
                    "harmful": {"source": preset.harmful[0], "count": preset.harmful[1]},
                    "harmless": {"source": preset.harmless[0], "count": preset.harmless[1]},
                    "validate": {"harmful": preset.validate_harmful, "benign": preset.validate_benign},
                    "test": {"harmful": preset.test_harmful, "benign": preset.test_benign},
                    "sources": [{"name": s, "title": SOURCES[s].title, "licence": SOURCES[s].licence,
                                 "licence_note": SOURCES[s].licence_note, "home": SOURCES[s].home}
                                for s in dict.fromkeys(used)]})
    return out
