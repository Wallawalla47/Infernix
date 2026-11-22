"""The vision golden fixture (tests/fixtures/inkling2/vision_golden.json) for test_vision_real.cpp: two asymmetric test
images (sides not multiples of 40, rows != columns; one RGBA with an alpha ramp and fully transparent and opaque
regions, one RGB) as lossless PNGs, the image processor's exact BF16 pixel words (a SHA-256 of each cell's words), and
the reference tower's rows.

- Pixels: transformers' Inkling2ImageProcessor (convert_to_rgb and _preprocess transcribed below from the transformers
  PR's modular_inkling2.py, Apache-2.0, with TorchvisionBackend.pad / rescale_and_normalize and torchvision's pad and
  normalize as they compute: pad right and bottom with -1, (x - 127.5) / 127.5 in FP32).
- Rows: SGLang's Inkling2VisionEncoder (sglang/srt/models/inkling2_vision.py, imported unmodified from --sglang) on the
  checkpoint's model.visual.* weights, run in FP32 on CPU from the BF16 pixel values (its forward from
  _preprocess_patches on: SGLang's own processor delivers CLIP-normalized frames that it maps back to [-1, 1]), stored
  as BF16. An independent FP64 statement of the tower agreed with this to 1.3e-6 relative RMS on 2026-11-22.

  py -3.11 generate_vision_golden.py --checkpoint C:/models/Inkling-2-Mini --sglang DIR_OF_inkling2_vision.py OUT.json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def convert_to_rgb(image: torch.Tensor) -> torch.Tensor:  # Inkling2ImageProcessor.convert_to_rgb (RGBA input)
    height, width = image.shape[-2:]
    rows = torch.arange(height)[:, None] // 8
    columns = torch.arange(width)[None, :] // 8
    background = torch.where((rows + columns) % 2 == 0, 204, 153)
    rgba = image.to(torch.int32)
    alpha = rgba[3:4]
    return ((rgba[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)


def preprocess(image: torch.Tensor, side: int = 40) -> tuple[torch.Tensor, list[int]]:  # Inkling2ImageProcessor._preprocess
    channels, height, width = image.shape
    rows, columns = math.ceil(height / side), math.ceil(width / side)
    image = F.pad(image.float(), (0, columns * side - width, 0, rows * side - height), value=-1.0)
    mean = (torch.tensor([0.5, 0.5, 0.5]) * (1.0 / 0.00392156862745098)).view(-1, 1, 1)
    std = (torch.tensor([0.5, 0.5, 0.5]) * (1.0 / 0.00392156862745098)).view(-1, 1, 1)
    image = (image - mean) / std
    image = image.reshape(channels, rows, side, columns, side).permute(1, 3, 2, 4, 0)
    return image.flatten(0, 1), [1, rows, columns]


def load_encoder(checkpoint: Path, sglang: Path):
    sys.path.insert(0, str(sglang))
    import inkling2_vision as sgl  # SGLang's file, unmodified
    from safetensors import safe_open

    vision = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))["vision_config"]
    encoder = sgl.Inkling2VisionEncoder(vision, vision["text_hidden_size"])
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
    state = {}
    for name in sorted({f for k, f in index["weight_map"].items() if k.startswith("model.visual.")}):
        with safe_open(str(checkpoint / name), framework="pt") as handle:
            for key in handle.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual."):]] = handle.get_tensor(key)
    encoder.load_state_dict(state, strict=True)
    return encoder.eval().float()


@torch.no_grad()
def tower(encoder, pixels: torch.Tensor, grid: list[list[int]]) -> torch.Tensor:
    patches = pixels.to(torch.bfloat16).float().permute(0, 3, 1, 2).contiguous()
    hidden, positions, lengths, cu_seqlens = encoder._embed_images(patches, grid)
    for layer in encoder.backbone.layers:
        hidden = layer(hidden, positions, lengths, cu_seqlens)
    hidden = encoder._pool(hidden, grid, lengths)
    return encoder.proj_out(F.gelu(encoder.proj_in(hidden)))


def png(rgba: np.ndarray) -> bytes:
    h, w, c = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    color = {4: 6, 3: 2}[c]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def image(height: int, width: int, transparent: bool) -> np.ndarray:
    """Smooth asymmetric content (gradients with per-channel slopes, a bright off-centre disc, a ripple), and with
    `transparent` an alpha ramp plus fully transparent and opaque corners."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 255 * (x / max(width - 1, 1)) ** 0.7
    g = 255 * (y / max(height - 1, 1)) ** 1.3
    b = 127 + 120 * np.sin(x / 7.0 + y / 13.0)
    rgb = np.stack([r, g, b], -1)
    rgb[(x - 0.7 * width) ** 2 + (y - 0.3 * height) ** 2 < (0.2 * min(width, height)) ** 2] = [250, 40, 30]
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    if not transparent:
        return rgb
    alpha = np.clip(np.rint(255 * x / max(width - 1, 1)), 0, 255).astype(np.uint8)
    alpha[: height // 4, : width // 5] = 0
    alpha[-height // 5:, -width // 4:] = 255
    return np.concatenate([rgb, alpha[..., None]], -1)


CASES = (("rgba_57x83", 57, 83, True), ("rgb_41x163", 41, 163, False))
# --large (a local check, not the committed fixture): 27 x 48 = 1,296 cells, 5,185 tower tokens, past the tower's
# 4,096-token chunks.
LARGE = (("rgba_1079x1915", 1079, 1915, True),)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sglang", type=Path, required=True)
    parser.add_argument("--large", action="store_true", help="the 1,296-cell image instead of the fixture's")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    encoder = load_encoder(args.checkpoint, args.sglang)
    cases = []
    for name, height, width, transparent in (LARGE if args.large else CASES):
        pixels_u8 = image(height, width, transparent)
        chw = torch.from_numpy(pixels_u8).permute(2, 0, 1).contiguous()
        pixels, grid = preprocess(convert_to_rgb(chw) if transparent else chw)
        words = pixels.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
        rows = tower(encoder, pixels, [grid])
        cases.append({
            "name": name, "width": width, "height": height, "rows": grid[1], "cols": grid[2],
            "png": base64.b64encode(png(pixels_u8)).decode(),
            # per cell: SHA-256 of its [40][40][3] BF16 words (little-endian), row-major (y, x, channel)
            "cell_sha256": [hashlib.sha256(words[c].tobytes()).hexdigest() for c in range(words.shape[0])],
            "rows_bf16": base64.b64encode(rows.to(torch.bfloat16).view(torch.int16).numpy().tobytes()).decode(),
            "output_size": rows.shape[1],
        })
        print(name, "grid", grid, "rows", tuple(rows.shape))
    args.out.write_text(json.dumps({"provenance": __doc__.split("\n\n")[0], "cases": cases}, indent=1) + "\n",
                        encoding="utf-8")


if __name__ == "__main__":
    main()
