"""The serving request log is one contract with readers written in two languages.

Serve stamps every record with `kRequestLogSchemaVersion` and each reader gates on its own copy of
that number, so a bump that reaches only the header leaves a reader rejecting every record it exists
to analyse. The readers' own tests build fixtures from their own constant, so nothing else connects
them back to the writer.
"""

from __future__ import annotations

import re
from pathlib import Path

from tools.bench import run_serve_corpus as corpus
from tools.bench.ttft import diagnostics

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER = REPO_ROOT / "src" / "serve" / "request_log.h"


def _writer_schema_version() -> int:
    match = re.search(r"kRequestLogSchemaVersion\s*=\s*(\d+)", HEADER.read_text(encoding="utf-8"))
    assert match is not None, f"no request-log schema version found in {HEADER}"
    return int(match.group(1))


def test_every_request_log_reader_accepts_the_schema_serve_writes():
    version = _writer_schema_version()
    assert corpus.SERVER_LOG_SCHEMA_VERSION == version, (
        "tools/bench/run_serve_corpus.py expects serving log schema "
        f"{corpus.SERVER_LOG_SCHEMA_VERSION} while Serve writes {version}; every corpus and "
        "concurrency campaign then stops at the first server_start line"
    )
    assert diagnostics.REQUEST_LOG_SCHEMA_VERSION == version, (
        "tools/bench/ttft/diagnostics.py expects request-log schema "
        f"{diagnostics.REQUEST_LOG_SCHEMA_VERSION} while Serve writes {version}; its reports then "
        "degrade to unavailable for every record"
    )
