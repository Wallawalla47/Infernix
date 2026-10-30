#!/usr/bin/env python3
"""Join stored infernix_bench JSON reports into one table, across PCIe links or across builds.

A measurement run leaves a directory of reports: `<run>/report/<arm>.json` (what a run script writes)
or one report per process beside its `summary.json` (what `abba.py --out` writes). This tool reads any
number of those directories and prints the same arm's metric beside itself in every group, so one
table answers "what did the wider link buy?" (`--group-by link`, the default: each report joins on the
PCIe gen/width its own probe recorded) or "what did this build change?" (`--group-by run`).

The guards are the reason this is a tool rather than a one-off script:

* a report's link is what `infernix_bench` probed right after its measured runs, while the device was
  still loaded — an idle GPU drops its link generation, and a host-side probe reads the slot, not the
  run;
* a report whose probe recorded no `pcie_width` cannot be placed on the link axis, so it is dropped
  and counted rather than guessed (an old schema, or a probe that could not allocate);
* a run whose reports disagree on width is a card that moved, or a link that retrained, part-way
  through: its arms did not share one link, so the whole run is dropped unless `--allow-mixed-link`
  says the split is what you meant to measure;
* a ratio is only a link effect if nothing else differs, so every field that differs between the runs
  is listed beside the tables. A wide link next to a different CPU, OS, driver or page-cache state is
  not one experiment, and the reader, not the tool, decides whether the difference is acceptable.

Per metric, per arm/test/concurrency key and per group it prints the mean of the joined reports and
the spread across them (a group with one report can only offer that report's own repetition stddev).
There is no paired test here: separate processes share no block, so this table places runs side by
side and keeps every ratio descriptive. To decide a small difference, run `abba.py` — one process set,
one link, interleaved arms, Wilcoxon with Holm's correction — under `--expect-width`.

Usage:

    join_reports.py profiles/bench/<root> --out profiles/bench/<root>/join
    join_reports.py --run x8=profiles/.../20261028-095535Z-x8 \\
                    --run x16=profiles/.../20261028-103019Z-x16 --metric decode_output_tok_s

With a positional ROOT, its subdirectories become the runs in sorted order; repeated
`--run LABEL=DIR` is the explicit form and wins when both are given. --out writes join.md, join.csv
and join.json there; the tables are always printed.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# The link reading and its wording come from abba.py, so the two tools cannot disagree about what a
# report's probe says: one guard, read at two moments (per process, and per stored report).
import abba

REPORT_ARTIFACT_TYPE = "infernix_bench_report"
DEFAULT_METRICS = ("decode_output_tok_s", "output_tok_s", "prefill_tok_s", "load_seconds")
# The machine probes are the link evidence itself, so they ride along with the arm tables.
PROBE_METRICS = ("h2d_gbps", "ram_read_gbps")
LOWER_IS_BETTER = frozenset({"load_seconds", "upload_seconds", "total_seconds", "prefill_seconds",
                             "decode_seconds", "prepare_seconds", "workspace_peak_bytes"})
ABBA_TAG = re.compile(r"^b\d{2}-\d+-")  # abba.py's "<block>-<position>-<arm>" report tags
DEFAULT_EXCLUDES = ("warmup", "000_*")  # a discarded warm-up pass is not a measurement
# Fields compared between runs for the comparability note. The expert-state path, the corpus offset
# and the trace/constraint paths are excluded: they differ by construction (one state file per run
# directory, another prompt per block). Paths are never compared, only the artifact's size.
CONDITION_FIELDS = (
    ("model", ("load", "name")), ("architecture", ("load", "architecture")),
    ("artifact bytes", ("artifact", "file_size_bytes")), ("gpu", ("environment", "gpu_name")),
    ("cuda runtime", ("environment", "cuda_runtime_version")),
    ("cuda driver", ("environment", "cuda_driver_version")),
    ("kv cache", ("config", "kv_cache")), ("max context", ("config", "max_context")),
    ("prefill chunk", ("config", "prefill_chunk")), ("repetitions", ("config", "repetitions")),
    ("warmup", ("config", "warmup")), ("sampling", ("config", "sampling")),
    ("cuda graph", ("config", "use_cuda_graph")), ("decode path", ("config", "decode_path")),
    ("proposal head", ("config", "proposal_head")),
    ("spec backend", ("config", "speculative_backend")), ("draft tokens", ("config", "draft_tokens")),
    ("n-gram window", ("config", "ngram_draft_tokens")), ("n-gram match", ("config", "ngram_min_match")),
    ("ram free MiB", ("config", "ram_free_mib")), ("expert RAM MiB", ("config", "expert_ram_mib")),
    ("expert stripe", ("config", "expert_stripe")), ("logits dtype", ("config", "logits_dtype")),
    ("fast paths", ("config", "fast_paths")), ("temporary switches", ("config", "temporary_switches")),
)
VOLATILE_FIELDS = frozenset({"expert_state", "corpus_offset", "route_trace_path", "constraint_file"})


@dataclass
class Report:
    """One stored report file, reduced to what the join needs."""

    run: str
    path: Path
    arm: str
    schema: Any
    link: abba.Link
    concurrency: int | None
    conditions: dict[str, str] = field(default_factory=dict)
    # (metric, test label ("" for a run-level value), value, that report's own repetition stddev)
    values: list[tuple[str, str, float, float | None]] = field(default_factory=list)


@dataclass(frozen=True)
class Row:
    metric: str
    key: tuple[str, str, int | None]  # arm, test label, concurrency
    group: str
    value: float
    spread: float | None
    run: str


@dataclass
class Cell:
    """The joined measurements of one key in one group."""

    values: list[float] = field(default_factory=list)
    spreads: list[float | None] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values)

    @property
    def spread(self) -> float | None:
        if len(self.values) > 1:
            return statistics.stdev(self.values)
        return next((s for s in self.spreads if s is not None), None)

    @property
    def n(self) -> int:
        return len(self.values)


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def arm_name(path: Path) -> str:
    """The report's arm: its file name less the block/position tag abba.py prefixes it with."""
    return ABBA_TAG.sub("", path.stem)


def is_report(payload: Any) -> bool:
    return isinstance(payload, dict) and payload.get("artifact_type") == REPORT_ARTIFACT_TYPE


def report_entries(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """One run directory's reports as (path, payload), in a fixed order.

    A `report/` subdirectory (a run script's layout) is the report set when it exists; otherwise the
    directory itself is (an `abba.py --out` directory, whose summary.json is not a report). Files that
    do not parse or come from another tool are skipped, and unreadable ones are named on stderr."""
    nested = directory / "report"
    root = nested if nested.is_dir() else directory
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            print(f"join_reports: {path}: unreadable, skipped ({error})", file=sys.stderr)
            continue
        if is_report(payload):
            found.append((path, payload))
    return found


def discover(directory: Path) -> list[Path]:
    return [path for path, _ in report_entries(directory)]


def read_report(path: Path, run: str, payload: dict[str, Any], metrics: Sequence[str]) -> Report:
    """One parsed report file as a Report."""
    tests = payload.get("tests") or []
    values: list[tuple[str, str, float, float | None]] = []
    for metric in metrics:
        measured = False
        for test in tests:
            value = number(test.get(f"{metric}_mean"))
            if value is None:
                continue
            measured = True
            values.append((metric, str(test.get("label", "")), value,
                           number(test.get(f"{metric}_stddev"))))
        if measured:
            continue
        # Not a per-test metric: a run-level one (load's seconds, the machine block's probes).
        for section in ("load", "machine"):
            value = number((payload.get(section) or {}).get(metric))
            if value is not None:
                values.append((metric, "", value, None))
                break
    conditions = {}
    for label, (section, key) in CONDITION_FIELDS:
        raw = (payload.get(section) or {}).get(key)
        if raw is not None and key not in VOLATILE_FIELDS:
            conditions[label] = str(raw)
    concurrency = number((payload.get("config") or {}).get("concurrency"))
    return Report(run=run, path=path, arm=arm_name(path), schema=payload.get("schema_version"),
                  link=abba.link_of(payload), concurrency=int(concurrency) if concurrency else None,
                  conditions=conditions, values=values)


def resolve_runs(root: Path | None, run_args: Sequence[str]) -> list[tuple[str, Path]]:
    """The runs to join: the explicit `--run LABEL=DIR` entries, or ROOT's subdirectories in order."""
    if run_args:
        runs: list[tuple[str, Path]] = []
        for entry in run_args:
            label, separator, directory = entry.partition("=")
            if not separator or not label or not directory:
                raise SystemExit(f"--run wants LABEL=DIR, got {entry!r}")
            path = Path(directory)
            if not path.is_dir():
                raise SystemExit(f"run {label!r} has no directory {path}")
            if any(existing == label for existing, _ in runs):
                raise SystemExit(f"run label {label!r} given twice")
            runs.append((label, path))
        return runs
    if root is None:
        raise SystemExit("give a ROOT directory or at least one --run LABEL=DIR")
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    if discover(root):  # ROOT is itself one run's report directory
        return [(root.name, root)]
    runs = [(child.name, child) for child in sorted(root.iterdir())
            if child.is_dir() and discover(child)]
    if not runs:
        raise SystemExit(f"no {REPORT_ARTIFACT_TYPE} report under {root} (in it or its subdirectories)")
    return runs


def load_runs(runs: Sequence[tuple[str, Path]], metrics: Sequence[str],
              excludes: Sequence[str]) -> tuple[list[Report], list[str]]:
    """Read every report of every run, drop the arms that are not measurements, and say what was."""
    reports: list[Report] = []
    notes: list[str] = []
    excluded_arms: set[str] = set()
    for label, directory in runs:
        kept = 0
        for path, payload in report_entries(directory):
            report = read_report(path, label, payload, metrics)
            if not any(fnmatch.fnmatch(report.arm, pattern) for pattern in excludes):
                reports.append(report)
                kept += 1
                continue
            if not any(fnmatch.fnmatch(label, pattern) for pattern in excludes):
                excluded_arms.add(report.arm)
        if not kept:
            notes.append(f"run {label} ({directory}) contributed no measurement")
    if excluded_arms:
        notes.append("excluded arms as not measurements (" + ", ".join(excludes) + "): "
                     + ", ".join(sorted(excluded_arms)))
    return reports, notes


def guard_links(reports: Sequence[Report], allow_mixed: bool) -> tuple[list[Report], list[str]]:
    """Drop what cannot be placed on the link axis, naming every dropped report and why."""
    notes: list[str] = []
    kept: list[Report] = []
    by_run: dict[str, list[Report]] = {}
    for report in reports:
        by_run.setdefault(report.run, []).append(report)
    for label in sorted(by_run):
        group = by_run[label]
        widths = sorted({r.link.width for r in group if r.link.width is not None})
        if len(widths) > 1:
            message = (f"run {label} recorded {', '.join(f'x{w}' for w in widths)} across its "
                       f"{len(group)} reports: the card moved, or the link retrained, mid-run")
            if not allow_mixed:
                notes.append(message + " — the whole run is dropped (--allow-mixed-link to keep it)")
                continue
            notes.append(message + " — kept, each report joining on its own probe")
        unprobed = [r for r in group if r.link.width is None]
        if unprobed:
            notes.append(f"run {label}: {len(unprobed)} of {len(group)} report(s) recorded no PCIe "
                         "link width, so they cannot be placed on the link axis: "
                         + ", ".join(r.arm for r in unprobed[:5])
                         + (" ..." if len(unprobed) > 5 else ""))
            group = [r for r in group if r.link.width is not None]
        kept.extend(group)
    return kept, notes


def group_of(report: Report, by: str) -> str:
    return report.run if by == "run" else str(report.link)


def group_order(reports: Sequence[Report], by: str) -> list[str]:
    """The table's group columns. Link groups come narrowest-first, so a ratio reads as the
    better link over the worse one."""
    if by == "run":
        return sorted({r.run for r in reports})
    links = {str(r.link): r.link for r in reports}
    return sorted(links, key=lambda text: links[text].order)


def rows_of(reports: Sequence[Report], by: str, skip: Sequence[str] = ()) -> list[Row]:
    """One row per measurement, less the run-level metrics that carry no arm (the machine probes)."""
    return [Row(metric=metric, key=(report.arm, label, report.concurrency),
                group=group_of(report, by), value=value, spread=spread, run=report.run)
            for report in reports for metric, label, value, spread in report.values
            if metric not in skip]


def cells(rows: Sequence[Row]) -> dict[str, dict[tuple[str, str, int | None], dict[str, Cell]]]:
    table: dict[str, dict[tuple[str, str, int | None], dict[str, Cell]]] = {}
    for row in rows:
        cell = table.setdefault(row.metric, {}).setdefault(row.key, {}).setdefault(row.group, Cell())
        cell.values.append(row.value)
        cell.spreads.append(row.spread)
        cell.runs.append(row.run)
    return table


def fmt(value: float | None) -> str:
    """Four significant digits: enough to read a 1 % difference, few enough to scan a column."""
    if value is None:
        return "-"
    if abs(value) >= 1e5:  # past the point where four significant digits would use an exponent
        return "{:.0f}".format(value)
    return "{:.4g}".format(value)


def fmt_spread(value: float | None) -> str:
    return "-" if value is None else "{:.3g}".format(value)


def render_metric(metric: str, groups: Sequence[str], base: str,
                  by_key: dict[tuple[str, str, int | None], dict[str, Cell]],
                  lower: bool) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """One metric's markdown table, its flat rows, and the keys some groups never measured."""
    lines = [f"## {metric} ({'lower' if lower else 'higher'} is better; base `{base}`)", ""]
    header = ["arm", "test", "C"]
    for group in groups:
        header += [f"n({group})", group, f"± {group}"]
    header += [f"{group}/{base}" for group in groups if group != base]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    flat: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in sorted(by_key, key=lambda k: (k[0], k[1], k[2] is None, k[2] or 0)):
        here = by_key[key]
        arm, label, concurrency = key
        reference = here.get(base)
        columns = [arm, label or "-", "-" if concurrency is None else str(concurrency)]
        for group in groups:
            cell = here.get(group)
            columns += ["-" if cell is None else str(cell.n), fmt(cell.mean) if cell else "-",
                        fmt_spread(cell.spread) if cell else "-"]
        for group in groups:
            if group == base:
                continue
            cell = here.get(group)
            ratio = cell.mean / reference.mean if cell and reference and reference.mean else None
            columns.append(f"{ratio:.3f}x" if ratio is not None else "-")
        lines.append("| " + " | ".join(columns) + " |")
        absent = [g for g in groups if g not in here]
        if absent:
            missing.append(f"`{arm}` {label or ''} C{concurrency}: no " + ", ".join(absent)
                           + f" measurement of {metric}")
        for group in groups:
            cell = here.get(group)
            if cell is None:
                continue
            ratio = cell.mean / reference.mean if reference and reference.mean else None
            flat.append({"metric": metric, "arm": arm, "test": label, "concurrency": concurrency,
                         "group": group, "base": base, "reports": cell.n, "mean": cell.mean,
                         "spread": cell.spread, "ratio_vs_base": ratio,
                         "delta_pct": None if ratio is None else 100.0 * (ratio - 1.0),
                         "runs": sorted(set(cell.runs))})
    return lines, flat, missing


def render_probes(reports: Sequence[Report], groups: Sequence[str], by: str,
                  probes: Sequence[str]) -> list[str]:
    """The machine probes: the link evidence, kept apart from the arms' results."""
    lines: list[str] = []
    for metric in probes:
        means: dict[str, list[float]] = {}
        for report in reports:
            for name, label, value, _ in report.values:
                if name == metric and label == "":
                    means.setdefault(group_of(report, by), []).append(value)
        if len(means) < 2:
            continue
        ordered = [g for g in groups if g in means]
        unit = " GB/s" if metric.endswith("_gbps") else ""
        first = statistics.fmean(means[ordered[0]])
        readings = []
        for group in ordered:
            mean = statistics.fmean(means[group])
            ratio = "" if group == ordered[0] else f" ({mean / first:.2f}x)"
            readings.append(f"`{group}` {fmt(mean)}{unit}{ratio}")
        lines.append(f"- {metric}: " + ", ".join(readings))
    if not lines:
        return []
    return ["", "## Machine probes", "",
            "Every report probes its own link and rates right after its measured runs; these are the",
            "link's own numbers, not an arm's result.", ""] + lines


def brief(value: str) -> str:
    """A condition value short enough to read in a list. A switch record is counted, not dumped."""
    if " | " in value:
        return f"{len(value.split(' | '))} switches"
    return value if len(value) <= 56 else value[:56] + " ..."


def differing_conditions(reports: Sequence[Report]) -> list[tuple[str, dict[str, str]]]:
    """The fields that differ between the runs, ignoring those that vary between arms inside one run."""
    per_run: dict[str, dict[str, set[str]]] = {}
    for report in reports:
        seen = per_run.setdefault(report.run, {})
        for name, value in report.conditions.items():
            seen.setdefault(name, set()).add(value)
    found: list[tuple[str, dict[str, str]]] = []
    for name in sorted({present for seen in per_run.values() for present in seen}):
        if any(len(seen.get(name, set())) > 1 for seen in per_run.values()):
            continue  # it varies between arms inside a run, so it says nothing about the runs
        across = {run: next(iter(seen[name])) for run, seen in per_run.items() if name in seen}
        if len(set(across.values())) > 1:
            found.append((name, across))
    return found


def conditions_differ(reports: Sequence[Report]) -> list[str]:
    """Those fields as lines, with a record-shaped field's entry names spelled out."""
    lines = []
    for name, across in differing_conditions(reports):
        lines.append(f"- {name}: " + ", ".join(f"{run} `{brief(across[run])}`" for run in sorted(across)))
        if not any(" | " in value for value in across.values()):
            continue
        ordered = sorted(across)
        base = set(abba.parse_switches(across[ordered[0]]))

        def names(sample: list[str]) -> str:
            return "(" + ", ".join(sample[:3]) + (", ...)" if len(sample) > 3 else ")")

        for run in ordered[1:]:
            entries = set(abba.parse_switches(across[run]))
            added, dropped = sorted(entries - base), sorted(base - entries)
            if not (added or dropped):
                continue
            detail = []
            if dropped:
                detail.append(f"lacks {len(dropped)} {names(dropped)}")
            if added:
                detail.append(f"adds {len(added)} {names(added)}")
            lines.append(f"  - {name} in {run} against {ordered[0]}: " + " and ".join(detail)
                         + " — the engine's switch list changed, so these runs are different builds")
    return lines


def group_pooling_notes(reports: Sequence[Report], groups: Sequence[str], by: str) -> list[str]:
    """Warn when one group's column averages reports from runs that are not the same thing."""
    notes = []
    for group in groups:
        runs = sorted({r.run for r in reports if group_of(r, by) == group})
        if len(runs) < 2:
            continue
        fields = sorted({name for name, across in differing_conditions(reports)
                         if len({across.get(run) for run in runs}) > 1})
        if fields:
            notes.append(f"group `{group}` pools {len(runs)} runs that differ in "
                         + ", ".join(fields)
                         + ": its ± is a difference between those runs, not measurement spread. "
                           "Join one build at a time with --run LABEL=DIR to read a ratio.")
    return notes


def render(reports: Sequence[Report], table: dict[str, dict[tuple, dict[str, Cell]]],
           groups: Sequence[str], base: str, by: str, metrics: Sequence[str], probes: Sequence[str],
           lower: set[str], notes: Sequence[str],
           guard_notes: Sequence[str]) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    lines = ["# Report join", ""]
    lines.append("Runs: " + ", ".join(f"`{run}`" for run in sorted({r.run for r in reports})) + ".")
    lines.append(f"Grouped by {'the PCIe link each report probed' if by == 'link' else 'the run'}; "
                 f"ratios are group / `{base}`.")
    links = sorted({r.link.full for r in reports})
    lines.append("Links recorded: " + (", ".join(f"`{link}`" for link in links) if links else "none")
                 + ".")
    if guard_notes or notes:
        lines += ["", "## Guards and notes", ""]
        lines += [f"- {note}" for note in [*guard_notes, *notes]]
    if len(groups) < 2:
        need = ("a link comparison needs reports that probed two different widths"
                if by == "link" else "a comparison needs at least two runs")
        lines += ["", f"Only one group (`{groups[0] if groups else 'none'}`) has reports, so no ratio "
                     f"can be formed: {need}."]
        return lines, [], []
    lines += ["", "## Runs", "", "| run | links | reports | report schemas | model |",
              "|---|---|---|---|---|"]
    for run in sorted({r.run for r in reports}):
        here = [r for r in reports if r.run == run]
        lines.append(f"| {run} | " +
                     ", ".join(f"`{link}`" for link in sorted({str(r.link) for r in here})) +
                     f" | {len(here)} | " + ", ".join(sorted({str(r.schema) for r in here})) + " | " +
                     ", ".join(sorted({r.conditions.get("model", "?") for r in here})) + " |")
    differences = conditions_differ(reports)
    if differences:
        lines += ["", "Fields that differ between the runs. Some may be the point of the join; every "
                      "ratio below has to be read with them in mind.", ""] + differences
    lines += ["", "`±` is the spread across the joined reports of that group; a group with one report "
              "shows that", "report's own repetition stddev instead. Nothing here is paired — separate "
              "processes", "share no block — so a small difference is `abba.py`'s question, not this "
              "table's. Load and", "prefill also follow host page-cache and drive state, not only the "
              "link."]
    flat: list[dict[str, Any]] = []
    missing: list[str] = []
    for metric in metrics:
        if metric not in table:
            continue
        metric_lines, metric_flat, metric_missing = render_metric(
            metric, groups, base, table[metric], metric in lower)
        lines += [""] + metric_lines
        flat += metric_flat
        missing += metric_missing
    lines += render_probes(reports, groups, by, probes)
    if missing:
        lines += ["", "## Not measured in every group", ""] + [f"- {note}" for note in missing]
    return lines, flat, missing


def write_outputs(out: Path, lines: Sequence[str], flat: Sequence[dict[str, Any]],
                  missing: Sequence[str], meta: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "join.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = list(flat[0]) if flat else []
    with (out / "join.csv").open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in flat:
                writer.writerow({k: "|".join(v) if isinstance(v, list) else v
                                 for k, v in row.items()})
    (out / "join.json").write_text(json.dumps({"meta": meta, "rows": flat,
                                              "not_measured_in_every_group": missing},
                                             indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", nargs="?", type=Path, default=None,
                        help="directory whose subdirectories are the runs to join")
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                        help="one run to join under this label (repeatable); wins over ROOT")
    parser.add_argument("--group-by", choices=("link", "run"), default="link",
                        help="group reports by the PCIe link they probed (default) or by their run")
    parser.add_argument("--base", default=None, metavar="GROUP",
                        help="the group the ratios divide by (default: the narrowest link, or the "
                             "first run)")
    parser.add_argument("--metric", action="append", default=[],
                        help=f"a metric to join, repeatable; default: {', '.join(DEFAULT_METRICS)}")
    parser.add_argument("--lower-is-better", action="append", default=[],
                        help="mark another metric as lower is better (repeatable)")
    parser.add_argument("--higher-is-better", action="append", default=[],
                        help="mark a metric as higher is better, overriding the default (repeatable)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="arm-name pattern that is not a measurement, repeatable; default: "
                             + ", ".join(DEFAULT_EXCLUDES))
    parser.add_argument("--allow-mixed-link", action="store_true",
                        help="keep a run whose reports disagree on link width")
    parser.add_argument("--no-probes", action="store_true",
                        help=f"omit the {', '.join(PROBE_METRICS)} section")
    parser.add_argument("--out", type=Path, default=None,
                        help="write join.md, join.csv and join.json here")
    args = parser.parse_args(argv)

    metrics = list(args.metric) or list(DEFAULT_METRICS)
    # The machine probes are the link's own numbers rather than an arm's result, so they go in their
    # own section instead of repeating once per arm.
    probes = [] if args.no_probes else [m for m in PROBE_METRICS if m not in metrics]
    loaded = metrics + probes
    lower = (set(LOWER_IS_BETTER) | set(args.lower_is_better)) - set(args.higher_is_better)

    runs = resolve_runs(args.root, args.run)
    excludes = list(args.exclude) if args.exclude else list(DEFAULT_EXCLUDES)
    reports, notes = load_runs(runs, loaded, excludes)
    if not reports:
        raise SystemExit("; ".join(notes) or "no readable reports in those runs")
    if args.group_by == "link":
        reports, guard_notes = guard_links(reports, args.allow_mixed_link)
    else:
        unprobed = sum(1 for r in reports if r.link.width is None)
        guard_notes = ["grouped by run: each report's link is listed but not enforced"
                       + (f"; {unprobed} report(s) carried no link probe" if unprobed else "")]
    if not reports:
        raise SystemExit("; ".join(guard_notes) or "the link guard dropped every report")

    groups = group_order(reports, args.group_by)
    base = args.base or groups[0]
    if base not in groups:
        raise SystemExit(f"--base {base!r} is not one of the groups: {', '.join(groups)}")
    guard_notes += group_pooling_notes(reports, groups, args.group_by)
    lines, flat, missing = render(reports, cells(rows_of(reports, args.group_by, skip=probes)),
                                  groups, base, args.group_by, metrics, probes, lower, notes,
                                  guard_notes)
    if args.out:
        meta = {"group_by": args.group_by, "base": base, "groups": groups, "metrics": metrics,
                "probes": probes,
                "lower_is_better": sorted(lower), "excludes": excludes,
                "runs": [{"run": run, "dir": str(path)} for run, path in runs],
                "guards": guard_notes, "notes": notes,
                "links": sorted({r.link.full for r in reports})}
        write_outputs(args.out, lines, flat, missing, meta)
    print("\n".join(lines))
    if args.out:
        print(f"\nwritten: {args.out}/join.md, join.csv, join.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
