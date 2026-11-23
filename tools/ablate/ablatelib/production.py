"""The user's production setup: how they start infernix-serve, read once and kept in the run directory.

Accepted the three ways a setup exists:
  * a text file holding the production command line (as typed; line continuations and # comments allowed);
  * Ian's --config TOML (docs/serving.md "Configuration file"), optionally with command-line options on top;
  * on Linux, the running server's own command line (/proc/<pid>/cmdline).
Relative paths in a setup resolve against the directory production was launched from (infernix-serve's own rule),
so the tool resolves them once, here, and stores absolute paths.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import serve_flags as SF

# Options whose value is a path (resolved against the launch directory and checked to exist).
PATH_OPTIONS = {"--ngram-volume", "--expert-stripe", "--chat-template", "--expert-state", "--draft-model",
                "--prefix-cache-file", "--lora", "--ablate", "--context-cost-presets", "--request-log-jsonl",
                "--log-file", "--route-trace", "--ssd-context-dir"}


@dataclass
class Production:
    artifact: Path
    options: list[tuple[str, str | None]] = field(default_factory=list)   # in order; None = a switch
    serve_binary: Path | None = None
    source: str = ""
    launch_dir: Path = Path(".")

    def value(self, flag: str) -> str | None:
        found = [v for f, v in self.options if f == flag]
        return found[-1] if found else None

    def has(self, flag: str) -> bool:
        return any(f == flag for f, _ in self.options)

    def by_kind(self, kind: str) -> list[tuple[str, str | None]]:
        return [(f, v) for f, v in self.options if SF.rule(f).kind == kind]

    def refused(self) -> list[str]:
        return [f"{f}: {SF.rule(f).why}" for f, _ in self.options if SF.rule(f).kind == SF.REFUSE]

    def to_json(self) -> dict:
        return {"artifact": str(self.artifact), "options": [[f, v] for f, v in self.options],
                "serve_binary": str(self.serve_binary) if self.serve_binary else None,
                "source": self.source, "launch_dir": str(self.launch_dir)}

    @classmethod
    def from_json(cls, d: dict) -> Production:
        return cls(artifact=Path(d["artifact"]), options=[(f, v) for f, v in d["options"]],
                   serve_binary=Path(d["serve_binary"]) if d.get("serve_binary") else None,
                   source=d.get("source", ""), launch_dir=Path(d.get("launch_dir", ".")))


def _split_command_line(text: str) -> list[str]:
    """Shell-style split of a command line saved in a file: bash/zsh backslash continuations, cmd's ^ and
    PowerShell's backtick continuations, # comments and cmd's whole-line REM and :: comments. Windows paths keep
    their backslashes."""
    text = re.sub(r"(?im)^[ \t]*(?:rem(?:[ \t].*)?|::.*)\r?$", "", text)
    text = re.sub(r"[\\^`]\r?\n", " ", text)
    windows = os.name == "nt" or re.search(r"(^|\s|\")[A-Za-z]:\\", text) is not None
    lexer = shlex.shlex(text, posix=not windows, punctuation_chars=False)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    if windows:
        tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in tokens]
    return tokens


def _resolve(value: str, launch_dir: Path) -> str:
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (launch_dir / p).resolve())


def named_serve_binary(tokens: list[str], launch_dir: Path) -> Path | None:
    """The infernix-serve a command line starts with, when its first word names one."""
    if tokens and Path(tokens[0]).name.lower().startswith("infernix-serve"):
        return Path(_resolve(tokens[0], launch_dir))
    return None


def parse_tokens(tokens: list[str], arity: dict[str, str], launch_dir: Path, source: str) -> Production:
    """Tokens of an infernix-serve command line -> Production. A --config FILE among them is read here, with the
    command line overriding the file (infernix-serve's rule) and repeatable values accumulating."""
    if not tokens:
        raise ValueError(f"{source}: the production command line is empty")
    serve_binary = named_serve_binary(tokens, launch_dir)
    if serve_binary is not None:
        tokens = tokens[1:]
    artifact: str | None = None
    cli: list[tuple[str, str | None]] = []
    config_file: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            if artifact is not None or cli:
                raise ValueError(f"{source}: unexpected argument {tok!r} (the artifact path comes first)")
            artifact = tok
            i += 1
            continue
        name, eq, inline = tok.partition("=")
        if name == "--config":
            if config_file is not None:
                raise ValueError(f"{source}: --config is given more than once")
            config_file = inline if eq else (tokens[i + 1] if i + 1 < len(tokens) else None)
            if config_file is None:
                raise ValueError(f"{source}: --config needs a file")
            i += 1 if eq else 2
            continue
        if name not in arity:
            raise ValueError(f"{source}: {name} is not an option of this infernix-serve "
                             "(does the command line name the binary production runs?)")
        if arity[name] == "switch":
            cli.append((name, None))
            i += 1
            continue
        if eq:
            cli.append((name, inline))
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise ValueError(f"{source}: {name} needs a value")
        cli.append((name, tokens[i + 1]))
        i += 2
    options = cli
    if config_file is not None:
        cfg_path = Path(_resolve(config_file, launch_dir))
        file_model, file_options = read_config(cfg_path, arity)
        artifact = artifact or file_model
        given = {f for f, _ in cli}
        merged = [(f, v) for f, v in file_options if f not in given or arity.get(f) == "repeatable"]
        options = merged + cli
        source += f" + {cfg_path}"
    if artifact is None:
        raise ValueError(f"{source}: no model artifact (neither a first argument nor a `model` key)")
    resolved = [(f, _resolve(v, launch_dir) if (f in PATH_OPTIONS and v not in (None, "off", "auto")) else v)
                for f, v in options]
    prod = Production(artifact=Path(_resolve(artifact, launch_dir)), options=resolved, serve_binary=serve_binary,
                      source=source, launch_dir=launch_dir)
    for f, _ in prod.options:
        SF.rule(f)      # unknown options are refused here, by name
    return prod


def read_config(path: Path, arity: dict[str, str]) -> tuple[str | None, list[tuple[str, str | None]]]:
    """Ian's --config TOML: top-level keys and the [serve] table (which overrides the top level); [cli] skipped;
    `model` names the artifact; true/false for switches, an array for a repeatable option."""
    import tomllib
    data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    merged = {k: v for k, v in data.items() if not isinstance(v, dict)}
    for name, table in data.items():
        if isinstance(table, dict) and name not in ("serve", "cli"):
            raise ValueError(f"{path}: table [{name}] is not [serve] or [cli]")
    merged.update(data.get("serve") or {})
    model = merged.pop("model", None)
    options: list[tuple[str, str | None]] = []
    for key, value in merged.items():
        flag = "--" + key
        if flag not in arity:
            raise ValueError(f"{path}: key {key!r} is not an infernix-serve option")
        if arity[flag] == "switch":
            if value is True:
                options.append((flag, None))
            elif value is not False:
                raise ValueError(f"{path}: {key} takes true or false")
        elif isinstance(value, list):
            if arity[flag] != "repeatable":
                raise ValueError(f"{path}: {key} does not take an array")
            options += [(flag, str(v)) for v in value]
        else:
            options.append((flag, str(value).lower() if isinstance(value, bool) else str(value)))
    return model, options


def read_command_line(path: Path) -> list[str]:
    """The words of a command line saved in a file (continuations and comments as _split_command_line takes them)."""
    return _split_command_line(Path(path).read_text(encoding="utf-8-sig"))


def read_argv(path: Path) -> list[str]:
    """The words of a command line saved as a JSON list of strings (no quoting or comment rules to get wrong)."""
    words = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
        raise SystemExit(f"{path}: expected a JSON list of strings (the infernix-serve command line, word by word)")
    return words


def from_text_file(path: Path, arity: dict[str, str], launch_dir: Path | None = None) -> Production:
    return parse_tokens(read_command_line(path), arity, (launch_dir or Path.cwd()).resolve(), source=str(path))


def from_config_file(path: Path, arity: dict[str, str], launch_dir: Path | None = None,
                     extra: list[str] | None = None) -> Production:
    """Ian's --config TOML, with optional command-line options on top (an artifact path may come first)."""
    extra = list(extra or [])
    head = [extra.pop(0)] if extra and not extra[0].startswith("--") else []
    return parse_tokens([*head, "--config", str(path), *extra], arity, (launch_dir or Path.cwd()).resolve(),
                        source="--config")


def running_command_line() -> tuple[list[str], Path, str]:
    """(argv, launch directory, source) of the running infernix-serve (Linux only: /proc). On other systems, save the
    production command line to a file and pass it with --production instead."""
    if not sys.platform.startswith("linux"):
        raise ValueError("--from-running reads /proc, which only Linux has; save your production command line to "
                         "a text file and pass --production FILE")
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            if argv and Path(argv[0].decode()).name == "infernix-serve":
                found.append((int(proc.name), [a.decode() for a in argv if a], Path(os.readlink(proc / "cwd"))))
        except (OSError, UnicodeDecodeError):
            continue
    if len(found) != 1:
        raise ValueError(f"--from-running found {len(found)} running infernix-serve processes; it needs exactly one "
                         "(or save the production command line to a file and pass --production FILE)")
    pid, argv, cwd = found[0]
    return argv, cwd, f"running infernix-serve pid {pid}"
