"""Write a Flash-Next expert-bank object with the Python writer; the C++ reader
must parse its format and layout and accept its size and alignment."""

from pathlib import Path
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.codecs.nvfp4_expert import encode_nvfp4_expert_bank
from tools.artifact.schema import TensorSpec
from tools.artifact.writer import ArtifactWriter


def main() -> int:
    executable = sys.argv[1]
    g = torch.Generator().manual_seed(1)
    e, h, i = 3, 64, 32

    def codes(rows, k):
        return torch.randint(0, 256, (e, rows, k // 2), generator=g, dtype=torch.uint8)

    def scales(rows, k):
        return torch.randint(0, 0x7F, (e, rows, k // 16), generator=g, dtype=torch.uint8)

    bank = encode_nvfp4_expert_bank(
        gate_codes=codes(i, h), gate_scales=scales(i, h), up_codes=codes(i, h), up_scales=scales(i, h),
        down_codes=codes(h, i), down_scales=scales(h, i),
        multipliers=torch.rand((e, 3), generator=g) * 1e-3 + 1e-5, shape=(e, h, i))
    with tempfile.TemporaryDirectory(prefix="infernix-flash-next-interop-") as temporary:
        path = Path(temporary) / "flash_next.ninfer"
        with ArtifactWriter(
            path,
            [
                TensorSpec("bias", (130,), "bf16", "contiguous_le_v1"),  # misaligns the next offset
                TensorSpec("experts", (e, h, i), "nvfp4_mul", "nvfp4_expert_rg16_v1"),
            ],
            components={"text": {"config": {}}},
            bindings={"experts": {"object": "experts"}},
        ) as writer:
            writer.write_object("bias", bytes(260))
            writer.write_object("experts", bank)
        return subprocess.run([executable, str(path)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
