"""Write Flash-Next expert-bank and block-FP8 objects with the Python writer; the C++ reader
must parse their formats and layouts and accept their sizes and alignment."""

from pathlib import Path
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact.codecs.fp8_block import encode_fp8_block128
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
    fp8 = encode_fp8_block128(
        torch.randint(0, 0x7E, (2, 200, 130), generator=g, dtype=torch.uint8),
        torch.rand((2, 2, 2), generator=g), (2, 200, 130))
    with tempfile.TemporaryDirectory(prefix="ninfer-flash-next-interop-") as temporary:
        path = Path(temporary) / "flash_next.ninfer"
        with ArtifactWriter(
            path,
            [
                TensorSpec("bias", (130,), "bf16", "contiguous_le_v1"),  # misaligns the next offset
                TensorSpec("experts", (e, h, i), "nvfp4_mul", "nvfp4_expert_rg16_v1"),
                TensorSpec("mtp_experts", (2, 200, 130), "fp8_e4m3fn_block128_f32", "block128_scale_v1"),
            ],
            components={"text": {"config": {}}},
            bindings={"experts": {"object": "experts"}, "mtp_experts": {"object": "mtp_experts"}},
        ) as writer:
            writer.write_object("bias", bytes(260))
            writer.write_object("experts", bank)
            writer.write_object("mtp_experts", fp8)
        return subprocess.run([executable, str(path)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
