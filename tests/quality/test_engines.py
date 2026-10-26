"""Engine adapters (tools/quality/engines): the base header they read, and Strata's row dump and arguments."""

from __future__ import annotations

import numpy as np
import pytest

from tools.quality.engines import base_file, basalt, strata


def test_the_base_header_gives_each_chunk(tmp_path):
    tokens = np.arange(16, dtype="<i4")
    path = tmp_path / "b.kld"
    path.write_bytes(b"_logits_" + np.array([8, 5, 2], "<i4").tobytes() + tokens.tobytes() + b"\0" * 64)
    header = base_file.read_header(path)
    assert (header.context, header.vocab, header.chunks, header.first_row, header.rows_per_chunk) == (8, 5, 2, 4, 3)
    assert header.chunk(1).tolist() == list(range(8, 16))


def _record(position, logits):
    return np.array([position, len(logits)], "<i8").tobytes() + np.asarray(logits, "<f4").tobytes()


def test_strata_rows_are_read_by_position(tmp_path):
    path = tmp_path / "rows.bin"
    path.write_bytes(_record(5, [1, 2, 3]) + _record(4, [4, 5, 6]))
    rows = strata.read_rows(path, 3)
    assert sorted(rows) == [4, 5] and rows[4].tolist() == [4, 5, 6]
    path.write_bytes(_record(4, [1, 2, 3]) + _record(4, [1, 2, 3]))
    with pytest.raises(ValueError):
        strata.read_rows(path, 3)
    path.write_bytes(_record(4, [1, 2]))
    with pytest.raises(ValueError):
        strata.read_rows(path, 3)
    assert strata.read_rows(tmp_path / "absent.bin", 3) == {}


def test_basalt_runs_its_quality_protocol_in_wsl():
    assert basalt.wsl_path(r"E:\Infernix\x y\b.kld") == "/mnt/e/Infernix/x y/b.kld"
    assert basalt.wsl_path("/already/linux") == "/already/linux"
    command = basalt.engine_command("/opt/basalt/bin/basalt", "/mnt/c/p.basalt", "/mnt/d/b.kld", "/mnt/d/w/rows.tsv",
                                    ["--kv", "int8"], True, "/mnt/d/w/rows.bin")
    assert command[:4] == ["wsl.exe", "-e", "env", "BASALT_PPL_ROWS=/mnt/d/w/rows.bin"]
    tail = command[4:]
    assert tail[:2] == ["/opt/basalt/bin/basalt", "--serve"]
    assert tail[tail.index("--ppl-base") + 1] == "/mnt/d/b.kld" and tail[-2:] == ["--kv", "int8"]
    assert tail[tail.index("--spec") + 1] == "5" and tail[tail.index("--max-context") + 1] == "4096"


def test_strata_keeps_the_configuration_and_reads_every_token_through_decode():
    config = {"args": ["--pack", "p", "--serve", "--kv", "int8", "--max-context", "262144", "--spec", "4", "--vision"]}
    args = strata.engine_args(config, "patched.exe", 512)
    assert args[:2] == ["patched.exe", "--serve"] and args.count("--serve") == 1 and "--vision" not in args
    assert args[args.index("--kv") + 1] == "int8" and args[args.index("--spec") + 1] == "4"
    assert args[args.index("--max-context") + 1] == "1024" and args[args.index("--short-read") + 1] == "576"
