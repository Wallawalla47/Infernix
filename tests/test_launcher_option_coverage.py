"""The launcher's Configure form against infernix-serve's option table.

Every option infernix-serve accepts has a control in the form
(apps/launcher/web/src/pages/configure/options.ts) or is one the launcher sets itself
(LAUNCHER_FLAGS there); the form names no option the server lacks; and a control is a switch
exactly where the server's option takes no value, a list exactly where it may be repeated. An
option added to the server without a control fails here, so the launcher keeps offering
everything the server can do.

Run with pytest, or directly: python tests/test_launcher_option_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVE_OPTIONS = ROOT / "src" / "serve" / "serve_options.cpp"
FORM_OPTIONS = ROOT / "apps" / "launcher" / "web" / "src" / "pages" / "configure" / "options.ts"
FORM_ARGS = ROOT / "apps" / "launcher" / "web" / "src" / "lib" / "args.ts"


def serve_options() -> dict[str, str]:
    """Flag -> "switch", "value" or "repeatable", from the server's ServeOptionArity table
    ({flag}, {flag, takes a value} or {flag, takes a value, repeatable})."""
    text = SERVE_OPTIONS.read_text(encoding="utf-8")
    table = {}
    for match in re.finditer(r'ServeOptionArity\{"(--[a-z0-9-]+)"(?:,\s*(true|false))?(?:,\s*(true|false))?\}', text):
        flag, takes_value, repeatable = match.groups()
        table[flag] = "switch" if takes_value == "false" else "repeatable" if repeatable == "true" else "value"
    assert len(table) > 100, "serve_options.cpp: the ServeOptionArity table was not found"
    return table


def form_controls() -> dict[str, str]:
    """Flag -> control kind, for the form's controls and the page's own fields."""
    text = FORM_OPTIONS.read_text(encoding="utf-8")
    controls = {
        match.group(2): match.group(1)
        for match in re.finditer(r'kind:\s*"(\w+)",\s*flag:\s*"(--[a-z0-9-]+)"', text)
    }
    for fields in re.finditer(r"fields:\s*\[([^\]]*)\]", text):
        for flag in re.findall(r'"(--[a-z0-9-]+)"', fields.group(1)):
            controls.setdefault(flag, "field")
    assert len(controls) > 80, "options.ts: the form's controls were not found"
    return controls


def string_list(path: Path, name: str) -> set[str]:
    text = path.read_text(encoding="utf-8")
    block = re.search(re.escape(name) + r"[^=]*=\s*(?:new Set\()?\[([^\]]*)\]", text)
    assert block, f"{path.name}: {name} was not found"
    return set(re.findall(r'"(--[a-z0-9-]+)"', block.group(1)))


def test_every_server_option_is_in_the_form_or_set_by_the_launcher() -> None:
    serve = serve_options()
    form = form_controls()
    launcher = string_list(FORM_OPTIONS, "LAUNCHER_FLAGS")
    missing = sorted(set(serve) - set(form) - launcher)
    assert not missing, (
        "infernix-serve options without a control in the launcher's Configure form "
        f"(add one to options.ts, or list it in LAUNCHER_FLAGS if the launcher sets it): {missing}"
    )
    overlap = sorted(set(form) & launcher)
    assert not overlap, f"both a control and a launcher-set flag: {overlap}"


def test_the_form_names_only_server_options() -> None:
    serve = serve_options()
    stale = sorted(set(form_controls()) - set(serve))
    assert not stale, f"controls for options infernix-serve does not have: {stale}"
    # --help and --config are read before the option table; the rest must be in it.
    unknown = sorted(string_list(FORM_OPTIONS, "LAUNCHER_FLAGS") - set(serve) - {"--help", "--config"})
    assert not unknown, f"LAUNCHER_FLAGS names options infernix-serve does not have: {unknown}"


def test_switches_match_the_server() -> None:
    serve = serve_options()
    known_switches = string_list(FORM_ARGS, "KNOWN_SWITCHES")
    wrong = []
    for flag, kind in form_controls().items():
        if kind == "field" or flag not in serve:
            continue
        # A switch control for a switch, a list control for a repeatable option, any other
        # control for an option with one value.
        expected = {"switch": "switch", "list": "repeatable"}.get(kind, "value")
        if serve[flag] != expected:
            wrong.append(f"{flag}: a {kind} control, but the server's option is a {serve[flag]} one")
        # Until the server's --help has been read, the form tells switches from valued options
        # by this list.
        if kind == "switch" and flag not in known_switches:
            wrong.append(f"{flag}: a switch missing from KNOWN_SWITCHES in args.ts")
    assert not wrong, "; ".join(wrong)


if __name__ == "__main__":
    failures = 0
    for name, check in list(globals().items()):
        if name.startswith("test_") and callable(check):
            try:
                check()
                print(f"ok    {name}")
            except AssertionError as error:
                failures += 1
                print(f"FAIL  {name}: {error}")
    sys.exit(1 if failures else 0)
