"""A CUDA_CHECK failure leaves its record in --log-file and on the console before the process aborts.

    python tests/test_cuda_failure_log.py PRETTY_LOGGING_TEST_EXE

The test binary's --abort-with-log-file mode builds a LoggingRuntime with that file and serve's
console queue, then fails a CUDA_CHECK. The bare stderr line is not enough: the bug is that
std::abort does not flush stdio, so the file must already hold the record when the process dies, and
the console writer thread must have written the record's console line before the abort.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="infernix_cuda_log_") as scratch:
        log = Path(scratch) / "cuda.log"
        result = subprocess.run([sys.argv[1], "--abort-with-log-file", str(log)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
        aborted = result.returncode < 0 if os.name != "nt" else result.returncode != 0
        if not aborted:
            print(f"FAIL: expected an abort, exit status {result.returncode}", file=sys.stderr)
            return 1
        text = log.read_text(encoding="utf-8", errors="replace")
        expected = ("[error] engine | src/x.cu:42: CUDA_CHECK(cudaInvalid()) failed: "
                    "cudaErrorInvalidValue:")
        if expected not in text:
            print(f"FAIL: --log-file does not hold the CUDA_CHECK record:\n{text}", file=sys.stderr)
            return 1
        console = result.stderr.decode("utf-8", errors="replace")
        expected_console = ("ERROR engine | src/x.cu:42: CUDA_CHECK(cudaInvalid()) failed: "
                            "cudaErrorInvalidValue:")
        if expected_console not in console:
            print(f"FAIL: the console does not hold the CUDA_CHECK record:\n{console}",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
