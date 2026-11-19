"""Tools that qualify or measure must not pass on nothing: tools/flash_next/verify_artifact.py needs the n-gram
volume or an explicit --no-ngram, and tools/bench/idle_gap_ttft.py's summary needs a gap-0 baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "bench"))

import idle_gap_ttft  # noqa: E402


def write_rows(path: Path, gaps: list[float]) -> None:
    rows = [{"repeat": r, "gap": gap, "ttft1": 1.0, "ttft2": 0.1 + gap / 1000.0, "total2": 2.0, "pstate": "P2"}
            for r in range(3) for gap in gaps]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_idle_gap_summary_measures_excess_against_gap_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "arm.jsonl"
    write_rows(path, [0.0, 12.0])
    assert idle_gap_ttft.summary([str(path)]) == 0
    line = [l for l in capsys.readouterr().out.splitlines() if l.split()[1:2] == ["12"]][0]
    assert line.split()[5] == "12.0"  # 112 ms against 100 ms at gap 0


def test_idle_gap_summary_without_gap_zero_has_no_baseline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "arm.jsonl"
    write_rows(path, [3.0, 12.0])  # every gap-0 pair failed or was not run
    assert idle_gap_ttft.summary([str(path)]) == 1
    out = capsys.readouterr().out
    assert "no measured gap-0 pair" in out
    # Measured against the 3 s gap, the 3 s row would read 0.0 ms of excess.
    assert " 0.0 " not in out


def test_verify_artifact_requires_the_volume_or_an_explicit_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    sys.path.insert(0, str(ROOT))
    from tools.flash_next import verify_artifact

    monkeypatch.setattr(sys, "argv", ["verify_artifact", "--model", "m", "--artifact", "a.infernix"])
    with pytest.raises(SystemExit) as raised:
        verify_artifact.main()
    assert raised.value.code == 2
    monkeypatch.setattr(sys, "argv", ["verify_artifact", "--model", "m", "--artifact", "a", "--ngram", "v", "--no-ngram"])
    with pytest.raises(SystemExit):
        verify_artifact.main()
