"""The 27B quality comparison's vLLM checkpoint: unsloth/Qwen3.8-27B-NVFP4 with only its token embedding replaced by
the embedding NInfer's official artifact holds, so vLLM, NInfer, TensorSharp and Infernix run one model.

NInfer's artifact (neroued/Qwen3.8-27B-nvfp4-NInfer, SHA256 74d2c571...) keeps Unsloth's NVFP4 and FP8 words
unchanged and stores the embedding as FP8 rows (E4M3 codes with one BF16 scale per row, object weight/000242). vLLM
loads only a BF16 embedding; an embedding row's output is BF16 in the model (the hidden states), so the table
bf16(code x scale) (the exact product, at most 12 significant bits, rounded once to nearest even) gives the same
model. The copy keeps every file as it is and overwrites the embedding tensor's bytes in place (the same dtype and
shape), so nothing else can change.

  python bench/engine_quality/make_vllm_27b_copy.py UNSLOTH_DIR NINFER_ARTIFACT OUT_DIR
"""
import json
import shutil
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.artifact.reader import Artifact  # noqa: E402

NAME = "model.language_model.embed_tokens.weight"
EMBEDDING = "weight/000242"


def tensor_span(path: Path, name: str) -> tuple[int, int, str, list[int]]:
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    entry = header[name]
    begin, end = entry["data_offsets"]
    return 8 + size + begin, end - begin, entry["dtype"], entry["shape"]


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    source, ninfer, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    if out.exists():
        raise SystemExit(f"{out} exists")
    with Artifact(ninfer) as artifact:
        obj = next(o for o in artifact.objects if getattr(o, "id", "") == EMBEDDING)
        raw = torch.frombuffer(bytearray(artifact.read_object(obj.id)), dtype=torch.uint8)
    n, k = obj.shape
    codes = raw[: n * k].reshape(n, k).view(torch.float8_e4m3fn)
    scales = raw[n * k: n * k + 2 * n].view(torch.bfloat16)
    shutil.copytree(source, out)
    target = out / "model.safetensors"
    offset, nbytes, dtype, shape = tensor_span(target, NAME)
    if dtype != "BF16" or shape != [n, k] or nbytes != 2 * n * k:
        raise SystemExit(f"{NAME}: {dtype} {shape}, not BF16 [{n}, {k}]")
    with target.open("r+b") as f:
        for first in range(0, n, 8192):
            last = min(n, first + 8192)
            # code x scale exactly in FP32 (4 + 8 significant bits), rounded once to BF16
            rows = (codes[first:last].to(torch.float32) * scales[first:last].to(torch.float32)[:, None]).to(torch.bfloat16)
            f.seek(offset + 2 * first * k)
            f.write(rows.contiguous().view(torch.uint8).numpy().tobytes())
    (out / "ENGINE-COMPARE.txt").write_text(
        f"unsloth/Qwen3.8-27B-NVFP4 with {NAME} replaced by bf16(code x scale) of {ninfer} {EMBEDDING} "
        "(Infernix engine quality comparison, bench/engine_quality/make_vllm_27b_copy.py).\n", encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
