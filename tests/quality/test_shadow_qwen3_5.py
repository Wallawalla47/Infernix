"""The Qwen3.5 shadow on transformers 5.5.0 (its authority): shadow_qwen3_5_tiny.py in a subprocess with 5.5
first on PYTHONPATH. IQS_TRANSFORMERS_5_5 names the directories that provide it (os.pathsep-separated, e.g. a
`uv pip install --target` of transformers 5.5.0 and its pinned dependencies); without it the test skips."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_tiny_qwen3_5_through_the_shadow():
    paths = os.environ.get("IQS_TRANSFORMERS_5_5")
    if not paths:
        pytest.skip("IQS_TRANSFORMERS_5_5 is not set")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([paths, str(ROOT)]))
    done = subprocess.run([sys.executable, str(Path(__file__).with_name("shadow_qwen3_5_tiny.py"))], env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip().endswith("OK")
