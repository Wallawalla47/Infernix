"""IQS-2 certification run (docs/maintainer/model-quality.md, section 6): the whole regime as one ordered list of
steps on one machine, decided into one certificate.

  python -m tools.quality.certify plan --artifacts LOCAL.json --build BUILD --store STORE --out RUN [--mode M]
  python -m tools.quality.certify run RUN [--dry-run] [--blocks C0,C1,...]
  python -m tools.quality.certify certificate RUN

`plan` writes RUN/steps.json (every command in run order, its outputs and watchdog) and RUN/plan.json (what the
certificate decides); `run` executes the steps not yet done, in order, logging each to RUN/logs and RUN/status.jsonl,
then writes RUN/certificate/certificate.{json,html}. A run resumes where it stopped: a step whose outputs exist is
skipped. References and noise twins live in STORE, keyed by artifact, set, route and KV format; the lossless route,
the design reference and the noise twin never change their bits, so later runs reuse them. The main set is stored in
layers of 32 windows (the corpus grows a layer at a time): each layer of a reference or twin is its own part, so a
larger corpus makes only its new layers, and the arms, which measure the shipped default route, run on every window.
The reproducible route has no quality arms: its promise (bit-exact, request-invariant) is tested bitwise (block F).
--mode release adds the C0 spot check (re-score window 0 of every stored
reference, bit for bit); a missing reference is made in either mode.

LOCAL.json maps each artifact key to this machine's files:
  {"q27": {"path": "E:/.../qwen3_8_27b_nvfp4-official.ninfer"},
   "rc": {"path": ".../Uncensored-NVFP4-Infernix-00001-of-00003.infernix", "ngram_volume": ".../x.ngram"},
   "d8": {..., "draft_model": ".../Qwen3.8-Flash-Next-DFlash2-Infernix.infernix"}, ...}
Keys absent from LOCAL.json (or excluded by --only) are left out of the plan, and the certificate says so.

G1 (block B): the run captures the lossless route's logits (STORE/<artifact>/conformance); the CPU comparison
  python -m tools.quality.shadow STORE/<artifact>/conformance --artifact ART --out RUN/checks/shadow-<artifact>.json
runs beside or after the GPU blocks, before the certificate.

Blocks (standard section 5.2): C1 references (lossless; design reference where the artifact declares activation
quantization), C2 noise twins (every main-set window; for a KV format the twin of its design), C3 exact-codec
references for the KV formats on the family representatives, C4 long-set references, D BF16-KV arms of every
configuration, E1 KV-format arms, E2 long-set arms, F functional and end-to-end checks, G the certificate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "eval" / "corpora" / "iqs-1" / "manifest.json"

KV_FORMATS = ("int8", "fp8", "nvfp4", "k8v4", "vq2", "k4v2")
KV_REPRESENTATIVES = ("q27gw", "rc")          # one per attention family (Qwen3.5, Qwen3.8-Flash-Next)
LONG_REPRESENTATIVES = ("q27gw", "q35", "rc")  # dense Qwen3.5, MoE Qwen3.5, Flash-Next
CONFORMANCE = ("q27gw", "q35", "fp8")          # G1: one artifact per architecture (Flash-Next: the official FP8)
CAPTURE = ("--windows", "0,8,16,24", "--tokens", "512", "--every", "16")  # 4 sequences, 31 positions each
LAYER = 32  # main-set windows per corpus layer (tools/quality/iqs_corpus.py)
TWIN_SEED = 20261022
# The one-request decode arms: a window's segments run in order, each resuming from the previous segment's
# cached prompt (the prefix cache), so a window prefills about 3,840 tokens instead of 7,168.
CHAINED = ("--concurrency", "1", "--prefix-reuse", "--warm-tokens", "200")
# The prefill arm: deterministic for a fixed build, so with --reuse (an earlier run whose engine routes are
# unchanged) its earlier windows are reused and only the new layers run.
SINGLE_REQUEST_ARMS = ("prefill-default",)
# The stress arm: four requests per window resuming from a 200-token warm prefix (off the prefill chunk grid) in a
# KV pool too small for all four (7,424 tokens of segments in 6,144), so requests are preempted and replayed.
STRESS = ("--concurrency", "4", "--prefix-reuse", "--warm-tokens", "200", "--kv-tokens", "6144")

FN_MTP = ("--spec", "mtp", "--draft-tokens", "auto", "--ngram-draft-tokens", "7")


@dataclass(frozen=True)
class Config:
    """One certified configuration: an artifact with the engine options that define its declared design."""
    key: str
    artifact: str                 # LOCAL.json key
    family: str                   # qwen3_5 | qwen4_exp
    declared: bool                # declares activation quantization: R(A, bf16) is the design reference
    stress: tuple[str, ...]       # the stress arm's speculation: the artifact's default drafter
    extra: tuple[str, ...] = ()   # engine options of the configuration itself
    tree: tuple[str, ...] = ()    # a one-request DFlash2 tree-verification arm
    specdist: bool = False        # end-to-end speculative-sampling distribution check
    representative: bool = False  # long generation and task audits


CONFIGS = (
    Config("q27", "q27", "qwen3_5", True,
           ("--spec", "dflash2", "--draft-tokens", "auto", "--ngram-draft-tokens", "auto"),
           tree=("--spec", "dflash2", "--draft-tokens", "7", "--draft-tree-nodes", "16"), specdist=True,
           representative=True),
    Config("q27gw", "q27gw", "qwen3_5", False, ("--spec", "dflash2", "--draft-tokens", "7")),
    Config("q35", "q35", "qwen3_5", False, ("--spec", "mtp", "--draft-tokens", "auto"), specdist=True,
           representative=True),
    Config("rc", "rc", "qwen4_exp", False, FN_MTP, specdist=True, representative=True),
    Config("iq3", "iq3", "qwen4_exp", False, FN_MTP),
    Config("fp8", "fp8", "qwen4_exp", False, FN_MTP),
    Config("fp4l", "fp4l", "qwen4_exp", False, FN_MTP),
    Config("fp4l-a4", "fp4l", "qwen4_exp", True, FN_MTP, extra=("--expert-a4-activations",)),
    Config("d8", "d8", "qwen4_exp", True,
           ("--spec", "dflash2", "--draft-model", "{draft_model}", "--draft-tokens", "auto"), specdist=True),
    Config("ra", "ra", "qwen4_exp", True, FN_MTP),
)


@dataclass
class Step:
    block: str
    name: str
    argv: list[str]
    outputs: list[str]                # done when every output exists (a .json output must be complete)
    minutes: int                      # watchdog
    env: dict[str, str] = field(default_factory=dict)
    check: str | None = None          # a pass/fail step: the result JSON the orchestrator writes from its exit code


def compared(design: Path) -> list[Path]:
    """A design reference made with --reference: its own metadata and its design-error arm."""
    return [Path(str(design) + ".compare.json"), Path(str(design) + ".json")]


def main_windows() -> int:
    manifest = json.loads(CORPUS.read_text(encoding="utf-8"))
    count = len(manifest["sets"]["main"]["documents"])
    if count % LAYER:
        raise SystemExit(f"{CORPUS}: {count} main-set windows is not a whole number of {LAYER}-window layers")
    return count


def layer_path(path: Path, layer: int, legacy: bool = True) -> Path:
    """Layer k of a stored main-set item: windows 32k .. 32k+31. Layer 0 of a reference keeps the name it had
    before the corpus grew (legacy); twins and arms always carry their window range."""
    if layer == 0 and legacy:
        return path
    suffix = f".w{layer * LAYER}-{(layer + 1) * LAYER - 1}"
    return path.with_name(path.stem + suffix + path.suffix) if path.suffix == ".ref" else Path(str(path) + suffix)


def window_list(layer: int) -> str:
    return ",".join(str(w) for w in range(layer * LAYER, (layer + 1) * LAYER))


def exe(build: Path, name: str) -> str:
    for candidate in (build / "apps" / "Release" / f"{name}.exe", build / "apps" / name, build / "apps" / f"{name}.exe"):
        if candidate.is_file():
            return str(candidate)
    return str(build / "apps" / "Release" / f"{name}.exe")


def ctest_exe() -> str:
    found = shutil.which("ctest")
    if found:
        return found
    default = Path("C:/Program Files/CMake/bin/ctest.exe")
    return str(default)


class Planner:
    def __init__(self, local: dict, build: Path, store: Path, run: Path, mode: str, only: set[str] | None,
                 kv_cells: list[str] = (), reuse: Path | None = None):
        self.local, self.build, self.store, self.run, self.mode = local, build, store, run, mode
        # An earlier run on a build whose serving, default and reproducible routes this build leaves unchanged:
        # its end-to-end, functional and conformance results and its single-request default arms are reused.
        self.reuse = reuse
        self.certify = exe(build, "infernix-certify")
        self.serve = exe(build, "infernix-serve")
        self.configs = [c for c in CONFIGS if c.artifact in local and (only is None or c.key in only)]
        self.left_out = [c.key for c in CONFIGS if c not in self.configs]
        # The KV formats measured beyond BF16, per configuration: every format on the representatives, and
        # the release's rotation cells (KEY:FORMAT).
        self.kv_cells: dict[str, list[str]] = {key: list(KV_FORMATS) for key in KV_REPRESENTATIVES}
        for cell in kv_cells:
            key, _, kv = cell.partition(":")
            if key not in {c.key for c in CONFIGS} or kv not in KV_FORMATS:
                raise SystemExit(f"--kv-cell {cell}: expected KEY:FORMAT, FORMAT one of {', '.join(KV_FORMATS)}")
            if kv not in self.kv_cells.setdefault(key, []):
                self.kv_cells[key].append(kv)
        self.layers = main_windows() // LAYER
        self.steps: list[Step] = []
        self.decisions: list[dict] = []
        self.disclosures: list[dict] = []
        self.checks: list[dict] = []

    # Paths -------------------------------------------------------------------------------------------------
    def lossless(self, artifact: str, set_name: str = "main") -> Path:
        return self.store / artifact / f"{set_name}-lossless.ref"

    def design(self, config: Config, kv: str) -> Path:
        return self.store / config.key / f"main-design-{kv}.ref"

    def reference(self, config: Config, kv: str) -> Path:
        """R(A, K): the lossless route for a 16-bit configuration on BF16 KV, else the design reference."""
        if kv == "bf16" and not config.declared:
            return self.lossless(config.artifact)
        return self.design(config, kv)

    def twin(self, config: Config, kv: str = "bf16") -> Path:
        """The noise twin of R(A, K): the lossless route's for a 16-bit configuration on BF16 KV, else the design's."""
        return self.store / config.key / ("main-twin" if kv == "bf16" else f"main-twin-{kv}")

    def parts(self, path: Path, legacy: bool = True) -> list[Path]:
        """A stored main-set item's layers, in window order."""
        return [layer_path(path, k, legacy) for k in range(self.layers)]

    def reference_args(self, path: Path, set_name: str = "main") -> list[str]:
        out = []
        for part in (self.parts(path) if set_name == "main" else [path]):
            out += ["--reference", str(part)]
        return out

    def scored(self, config: Config, set_name: str = "main") -> Path:
        """The positions the artifact's windows score: the tokens the model writes (see infernix-certify mask)."""
        return self.store / config.artifact / f"{set_name}-scored.json"

    def arm(self, config: Config, kv: str, name: str, set_name: str = "main") -> Path:
        return self.run / "arms" / f"{config.key}-{set_name}-{kv}-{name}"

    # Commands ----------------------------------------------------------------------------------------------
    def artifact_args(self, config: Config) -> list[str]:
        entry = self.local[config.artifact]
        out = [entry["path"]]
        if entry.get("ngram_volume"):
            out += ["--ngram-volume", entry["ngram_volume"]]
        return out + list(config.extra)

    def speculation(self, config: Config, flags: tuple[str, ...]) -> list[str]:
        draft = self.local[config.artifact].get("draft_model", "")
        return [draft if f == "{draft_model}" else f for f in flags]

    def add(self, step: Step) -> None:
        if any(s.name == step.name for s in self.steps):
            return  # a shared reference (fp4l and fp4l-a4 share the lossless one)
        self.steps.append(step)

    def certify_step(self, block: str, name: str, command: str, config: Config, args: list[str], out: Path,
                     minutes: int, set_name: str = "main", outputs: list[Path] | None = None) -> None:
        argv = [self.certify, command, *self.artifact_args(config), "--corpus", str(CORPUS), "--set", set_name,
                "--out", str(out), *args]
        self.add(Step(block, name, argv, [str(p) for p in (outputs or [Path(str(out) + ".json")])], minutes))

    def decide(self, label: str, arm: list[Path], config: Config, kv: str, set_name: str = "main") -> None:
        # A long-set arm takes the BF16 main-set twin: no twin runs at 32K.
        twin = self.twin(config, kv if set_name == "main" else "bf16")
        self.decisions.append({"label": label, "arm": [str(p) for p in arm],
                               "twin": [str(p) for p in self.parts(twin, legacy=False)],
                               "scored": str(self.scored(config, set_name)),
                               "twin_scored": str(self.scored(config, "main"))})

    def mask(self, c: Config, set_name: str) -> None:
        out = self.scored(c, set_name)
        self.certify_step("C0", f"mask-{c.artifact}-{set_name}", "mask", c, [], out, 20, set_name=set_name,
                          outputs=[out])

    # The regime --------------------------------------------------------------------------------------------
    def plan(self) -> None:
        for c in self.configs:
            self.mask(c, "main")
            if c.key in LONG_REPRESENTATIVES:
                self.mask(c, "long")
        for c in self.configs:
            self.references(c)
        for c in self.configs:
            self.twin_step(c, "bf16")
        for c in self.configs:
            self.kv_references(c)
        for c in self.configs:
            for kv in self.kv_cells.get(c.key, ()):
                self.twin_step(c, kv)
        for c in self.configs:
            if c.key in LONG_REPRESENTATIVES:
                self.certify_step("C4", f"ref-{c.artifact}-long-lossless", "reference", c, [],
                                  self.lossless(c.artifact, "long"), 180, set_name="long")
        for c in self.configs:
            if c.key in CONFORMANCE:
                self.capture(c)
        if self.mode == "release":
            self.spot_checks()
        for c in self.configs:
            self.bf16_arms(c)
        for c in self.configs:
            for kv in self.kv_cells.get(c.key, ()):
                self.kv_arms(c, kv)
        for c in self.configs:
            if c.key in LONG_REPRESENTATIVES:
                self.long_arms(c)
        self.functional()
        for c in self.configs:
            self.end_to_end(c)
        self.sensitivity()

    @staticmethod
    def layer_name(name: str, layer: int) -> str:
        return name if layer == 0 else f"{name}-L{layer}"

    def design_reference(self, c: Config, kv: str, block: str) -> None:
        """A design reference, layer by layer, measured against the lossless one in the same pass."""
        design = self.design(c, kv)
        for k, part in enumerate(self.parts(design)):
            self.certify_step(block, self.layer_name(f"ref-{c.key}-design-{kv}", k), "reference", c,
                              ["--design", "--kv-dtype", kv, *self.reference_args(self.lossless(c.artifact)),
                               "--windows", window_list(k)], part, 120, outputs=compared(part))

    def references(self, c: Config) -> None:
        for k, part in enumerate(self.parts(self.lossless(c.artifact))):
            self.certify_step("C1", self.layer_name(f"ref-{c.artifact}-lossless", k), "reference", c,
                              ["--windows", window_list(k)], part, 120)
        if c.declared:
            self.design_reference(c, "bf16", "C1")
            self.disclosures.append({"label": f"{c.key} declared design (activation quantization, bf16 KV)",
                                     "arm": [str(p) + ".compare" for p in self.parts(self.design(c, "bf16"))],
                                     "scored": str(self.scored(c))})

    def twin_step(self, c: Config, kv: str) -> None:
        """The noise twin of R(A, K) on every main-set window, a layer per part."""
        for k, part in enumerate(self.parts(self.twin(c, kv), legacy=False)):
            route = ["--lossless", "--noise-seed", str(TWIN_SEED), "--windows", window_list(k),
                     *self.reference_args(self.reference(c, kv))]
            if c.declared or kv != "bf16":
                route[1:1] = ["--design", "--kv-dtype", kv]
            name = f"twin-{c.key}" if kv == "bf16" else f"twin-{c.key}-{kv}"
            self.certify_step("C2", self.layer_name(name, k), "prefill", c, route, part, 60)

    def kv_references(self, c: Config) -> None:
        for kv in self.kv_cells.get(c.key, ()):
            self.design_reference(c, kv, "C3")
            self.disclosures.append({"label": f"{c.key} KV {kv} design error",
                                     "arm": [str(p) + ".compare" for p in self.parts(self.design(c, kv))],
                                     "kv": kv, "scored": str(self.scored(c))})

    def capture(self, c: Config) -> None:
        """G1: the lossless route's whole logits rows, which tools/quality/shadow compares bit for bit
        with the publisher's model code in exact arithmetic (on the CPU, after the GPU blocks)."""
        out = self.store / c.artifact / "conformance"
        self.certify_step("B", f"capture-{c.artifact}", "capture", c, list(CAPTURE), out, 30)
        if earlier := self.reused(f"shadow-{c.artifact}"):
            self.checks.append({"name": f"B {c.artifact} reference conformance (G1, from {self.reuse.name})",
                                "result": str(earlier)})
            return
        self.checks.append({"name": f"B {c.artifact} reference conformance (G1)",
                            "result": str(self.run / "checks" / f"shadow-{c.artifact}.json")})

    def spot_checks(self) -> None:
        """C0: window 0 of the lossless reference of each attention family's representative, re-scored bit for bit
        (every lossless kernel serves every artifact of its family)."""
        refs = [(c, self.lossless(c.artifact)) for c in self.configs if c.key in KV_REPRESENTATIVES]
        for c, ref in refs:
            name = f"verify-{ref.parent.name}-{ref.stem}"
            result = self.run / "checks" / f"{name}.json"
            argv = [self.certify, "verify", *self.artifact_args(c), *self.reference_args(ref), "--windows", "0"]
            self.add(Step("C0", name, argv, [str(result)], 30, check=str(result)))
            self.checks.append({"name": f"C0 {ref.parent.name} {ref.stem} unchanged", "result": str(result)})

    def arm_steps(self, c: Config, kv: str, block: str, arms: list[tuple[str, str, list[str]]],
                  set_name: str = "main", decide: bool = True) -> list[list[Path]]:
        """Each arm's parts: with --reuse, a single-request main-set arm takes its earlier windows from the earlier run
        and runs its later layers here; every other arm runs on all its set's windows in this run."""
        reference = self.lossless(c.artifact, set_name) if set_name != "main" else self.reference(c, kv)
        common = [*self.reference_args(reference, set_name), "--kv-dtype", kv]
        minutes = 90 if set_name == "main" else 120
        outs = []
        for name, command, args in arms:
            step = f"arm-{c.key}-{set_name}-{kv}-{name}"
            earlier = self.reuse / "arms" / self.arm(c, kv, name, set_name).name if self.reuse else None
            if set_name == "main" and name in SINGLE_REQUEST_ARMS and earlier and Path(str(earlier) + ".json").is_file():
                # windows 0-31 from the earlier run, the later layers in this one
                parts = [earlier] + [layer_path(self.arm(c, kv, name, set_name), k, legacy=False)
                                     for k in range(1, self.layers)]
                for k, part in enumerate(parts[1:], start=1):
                    self.certify_step(block, self.layer_name(step, k), command, c,
                                      [*common, *args, "--windows", window_list(k)], part, minutes)
            else:
                parts = [self.arm(c, kv, name, set_name)]
                self.certify_step(block, step, command, c, [*common, *args], parts[0], minutes, set_name=set_name)
            if decide:
                self.decide(f"{c.key} {set_name} {kv} {name}", parts, c, kv, set_name)
            outs.append(parts)
        return outs

    def bf16_arms(self, c: Config) -> None:
        arms = [("prefill-default", "prefill", []),
                ("decode-default-one", "decode", list(CHAINED)),
                ("decode-default-stress", "decode", [*STRESS, *self.speculation(c, c.stress)])]
        if c.tree:
            arms.append(("decode-default-tree", "decode", [*CHAINED, *self.speculation(c, c.tree)]))
        self.arm_steps(c, "bf16", "D", arms)
        if c.declared:
            self.total_error(c)

    def total_error(self, c: Config) -> None:
        """A declared design's shipped route against the lossless route: engine and design error together, what a
        user of the artifact gets against the model itself (disclosed beside the design error, never gated)."""
        name = "prefill-default-vs-lossless"
        part = self.arm(c, "bf16", name)
        self.certify_step("D", f"arm-{c.key}-main-bf16-{name}", "prefill", c,
                          [*self.reference_args(self.lossless(c.artifact)), "--kv-dtype", "bf16"], part, 90)
        self.disclosures.append({"label": f"{c.key} default route against lossless (engine and design error)",
                                 "arm": [str(part)], "scored": str(self.scored(c))})

    def kv_arms(self, c: Config, kv: str) -> None:
        self.arm_steps(c, kv, "E1", [("prefill-default", "prefill", []),
                                     ("decode-default-stress", "decode", [*STRESS, *self.speculation(c, c.stress)])])

    def long_arms(self, c: Config) -> None:
        self.arm_steps(c, "bf16", "E2", [("prefill-default", "prefill", []),
                                         ("decode-default-one", "decode", list(CHAINED))], set_name="long")
        if c.key == "q27gw":
            # Every KV format at 32K against the lossless route: engine and design error together, held to the
            # format's class ceiling.
            for kv in KV_FORMATS:
                out = self.arm_steps(c, kv, "E2", [("prefill-default", "prefill", [])], set_name="long",
                                     decide=False)[0]
                self.disclosures.append({"label": f"{c.key} long KV {kv} (engine and design error)",
                                         "arm": [str(p) for p in out], "kv": kv, "scored": str(self.scored(c, "long"))})

    def functional(self) -> None:
        ctest = [ctest_exe(), "--test-dir", str(self.build), "-C", "Release", "--output-on-failure"]
        tests = self.build / "tests" / "Release"
        keys = {c.artifact for c in self.configs}
        suites = []
        if "q27" in keys:
            suites.append(("functional-qwen3_5-q27", ctest + ["-L", "real", "-R", "^infernix_qwen3_5_",
                                                               "-E", "lossless|moe_real|dflash_real|dflash_block_greedy"],
                           {"INFERNIX_TEST_ARTIFACT": self.local["q27"]["path"]}, 240))
        if "q35" in keys:
            env = {"INFERNIX_TEST_ARTIFACT": self.local["q35"]["path"]}
            suites.append(("functional-qwen3_5-q35", ctest + ["-L", "real", "-R", "^infernix_qwen3_5_", "-E",
                                                               "lossless|dflash2|tree|hybrid_prefix|speculation_invariance|"
                                                               "constraint|prefix_real|dflash_prefill"], env, 180))
            suites.append(("functional-qwen3_5-q35-dflash-prefill",
                           [str(tests / "infernix_qwen3_5_dflash_prefill_real_test.exe"), "dflash"], env, 30))
        if "rc" in keys:
            env = {"INFERNIX_QWEN4_ARTIFACT": self.local["rc"]["path"],
                   "INFERNIX_QWEN4_NGRAM": self.local["rc"].get("ngram_volume", "")}
            draft = next((v.get("draft_model") for v in self.local.values() if v.get("draft_model")), None)
            if draft:
                env["INFERNIX_QWEN4_DFLASH2"] = draft
            suites.append(("functional-qwen4_exp-rc", ctest + ["-L", "real", "-R", "^infernix_qwen4_exp_",
                                                                "-E", "lossless"], env, 240))
        for name, argv, env, minutes in suites:
            if earlier := self.reused(name):
                self.checks.append({"name": f"F {name} (from {self.reuse.name})", "result": str(earlier)})
                continue
            result = self.run / "checks" / f"{name}.json"
            self.add(Step("F", name, argv, [str(result)], minutes, env=env, check=str(result)))
            self.checks.append({"name": f"F {name}", "result": str(result)})

    def reused(self, name: str) -> Path | None:
        """The earlier run's result of a check this build leaves unchanged."""
        found = self.reuse / "checks" / f"{name}.json" if self.reuse else None
        return found if found and found.is_file() else None

    def sensitivity(self) -> None:
        """The identity comparison can see a difference: some representative's default route showed one."""
        identities = [self.run / "checks" / f"e2e-{c.key}-identity.json" for c in self.configs if c.representative]
        if not identities:
            return
        if earlier := self.reused("e2e-sensitivity"):
            self.checks.append({"name": f"F identity comparison sensitivity (negative controls, from {self.reuse.name})",
                                "result": str(earlier)})
            return
        result = self.run / "checks" / "e2e-sensitivity.json"
        argv = [sys.executable, "-m", "tools.quality.iqs_e2e", "sensitivity", "--out", str(result), "--results",
                *[str(p) for p in identities]]
        self.add(Step("F", "e2e-sensitivity", argv, [str(result)], 5))
        self.checks.append({"name": "F identity comparison sensitivity (negative controls)", "result": str(result)})

    def end_to_end(self, c: Config) -> None:
        server = ["--kv-dtype", "bf16", "--max-context", "16384", *self.artifact_args(c)[1:]]
        base = [sys.executable, "-m", "tools.quality.iqs_e2e"]
        common = ["--serve", self.serve, "--artifact", self.local[c.artifact]["path"]]
        common += [f"--server-arg={a}" for a in server]
        golden = self.store / c.key / "golden.json"
        control = ["--negative-control"] if c.representative else []
        jobs = [("identity", ["--golden", str(golden), *control], 60)]
        if c.specdist:
            jobs.append(("specdist", [f"--spec-arg={a}" for a in self.speculation(c, c.stress)], 60))
        if c.representative:
            jobs += [("longgen", [], 60), ("tasks", [], 90)]
        for check, args, minutes in jobs:
            if earlier := self.reused(f"e2e-{c.key}-{check}"):
                self.checks.append({"name": f"F {c.key} {check} (from {self.reuse.name})", "result": str(earlier)})
                continue
            result = self.run / "checks" / f"e2e-{c.key}-{check}.json"
            self.add(Step("F", f"e2e-{c.key}-{check}", [*base, check, *common, "--out", str(result), *args],
                          [str(result)], minutes))
            self.checks.append({"name": f"F {c.key} {check}", "result": str(result)})


def complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            return False
    return True


def provenance(build: Path, mode: str) -> dict:
    def run(argv: list[str], cwd: Path | None = None) -> str:
        try:
            return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"standard": "IQS-2", "mode": mode, "planned": dt.datetime.now().isoformat(timespec="seconds"),
            "commit": run(["git", "rev-parse", "HEAD"], ROOT), "build": str(build),
            "gpu": run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit",
                        "--format=csv,noheader"]),
            "corpus": str(CORPUS)}


def plan_command(args) -> int:
    local = json.loads(Path(args.artifacts).read_text(encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    planner = Planner(local, Path(args.build).resolve(), Path(args.store).resolve(), Path(args.out).resolve(),
                      args.mode, only, args.kv_cell, Path(args.reuse).resolve() if args.reuse else None)
    planner.plan()
    run = planner.run
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "checks").mkdir(parents=True, exist_ok=True)
    (run / "arms").mkdir(parents=True, exist_ok=True)
    (run / "steps.json").write_text(json.dumps([asdict(s) for s in planner.steps], indent=1) + "\n", encoding="utf-8")
    prov = provenance(planner.build, args.mode)
    prov["left_out"] = planner.left_out
    plan = {"provenance": prov, "decisions": planner.decisions, "disclosures": planner.disclosures,
            "checks": planner.checks}
    (run / "plan.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8")
    blocks: dict[str, int] = {}
    for s in planner.steps:
        blocks[s.block] = blocks.get(s.block, 0) + 1
    print(f"{len(planner.steps)} steps ({', '.join(f'{b} {n}' for b, n in blocks.items())}); "
          f"{len(planner.decisions)} decisions, {len(planner.disclosures)} disclosures, {len(planner.checks)} checks; "
          f"left out: {', '.join(planner.left_out) or 'none'}")
    return 0


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.kill()


def run_step(step: Step, run: Path) -> tuple[int, float]:
    log = run / "logs" / f"{step.name}.log"
    env = dict(os.environ, **step.env)
    for output in step.outputs:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("ab") as out:
        out.write(f"\n=== {dt.datetime.now().isoformat(timespec='seconds')} {subprocess.list2cmdline(step.argv)}\n"
                  .encode())
        out.flush()
        proc = subprocess.Popen(step.argv, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=step.minutes * 60)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            proc.wait()
            code = -9
            out.write(f"\n=== TIMEOUT after {step.minutes} min\n".encode())
    return code, time.time() - started


def run_command(args) -> int:
    run = Path(args.run).resolve()
    steps = [Step(**s) for s in json.loads((run / "steps.json").read_text(encoding="utf-8"))]
    status = run / "status.jsonl"
    blocks = set(args.blocks.split(",")) if args.blocks else None
    for step in steps:
        if all(complete(Path(o)) for o in step.outputs):
            continue
        if blocks is not None and step.block not in blocks:
            continue
        if args.dry_run:
            print(f"[{step.block}] {step.name}: {subprocess.list2cmdline(step.argv)}")
            continue
        print(f"[{step.block}] {step.name} ...", flush=True)
        code, seconds = run_step(step, run)
        if step.check:
            tail = (run / "logs" / f"{step.name}.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            Path(step.check).write_text(json.dumps({"passed": code == 0, "exit": code,
                                                    "detail": f"exit {code}; " + tail.strip().splitlines()[-1]
                                                    if tail.strip() else f"exit {code}"}, indent=1) + "\n",
                                        encoding="utf-8")
        with status.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"step": step.name, "block": step.block, "exit": code, "seconds": round(seconds, 1),
                                  "finished": dt.datetime.now().isoformat(timespec="seconds")}) + "\n")
        print(f"[{step.block}] {step.name} exit {code} in {seconds / 60:.1f} min", flush=True)
    if args.dry_run:
        return 0
    return certificate_command(args)


def certificate_command(args) -> int:
    from tools.quality import iqs
    run = Path(args.run).resolve()
    return iqs.run_certificate(run / "plan.json", run / "certificate", iqs.POLICY)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--artifacts", required=True)
    plan.add_argument("--build", required=True)
    plan.add_argument("--store", required=True)
    plan.add_argument("--out", required=True)
    plan.add_argument("--mode", choices=("full", "release"), default="full")
    plan.add_argument("--only", help="comma-separated configuration keys")
    plan.add_argument("--reuse", metavar="RUN", help="an earlier run on a build with the same serving, default and "
                      "reproducible routes: reuse its end-to-end, functional and conformance results and the earlier "
                      "windows of its single-request default arms")
    plan.add_argument("--kv-cell", action="append", default=[], metavar="KEY:FORMAT",
                      help="a KV format measured on a non-representative configuration (the release's rotation cell)")
    run = commands.add_parser("run")
    run.add_argument("run")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--blocks", help="comma-separated blocks to run now (the rest wait for a later run)")
    cert = commands.add_parser("certificate")
    cert.add_argument("run")
    args = parser.parse_args(argv)
    return {"plan": plan_command, "run": run_command, "certificate": certificate_command}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
