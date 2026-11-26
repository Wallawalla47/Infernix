"""Golden prompt ids for video messages: the reference processor's expansion of the chat template's video wrapper,
which test_video_golden_real.cpp compares with the Frontend's id for id (candidates VK-03).

The Qwen3.5-family checkpoints name Qwen3VLProcessor (preprocessor_config.json). Up to transformers 5.9 (the Qwen3.8
checkpoints were saved with 5.8.0.dev0, the Qwen3.6 ones with 4.57.1) it replaces the template's whole
<|vision_start|><|video_pad|><|vision_end|> with one `<t seconds><|vision_start|>pads<|vision_end|>` group per temporal
patch; from 5.10 it replaces only <|video_pad|>, which keeps the template's pair around those groups. The golden
follows the checkpoints' versions: the wrapper is given to the processor as the bare <|video_pad|> (the path its
5.8 code takes without a wrapper), so the groups, the grid, the frame sampling and the timestamps are the reference's
own code. The 5.10+ rendering is stored beside it for the record. Run with transformers 5.10 or later (5.16.1 was
used); fixtures avoid where its video processor differs from 5.8's: fewer than two frames (refused), a side under
32 pixels (5.8 refuses, later versions upscale) and sizes near the pixel bounds (5.8 rounds the frame count up there).

Each case is a lossless FFV1 video in Matroska written with the ffmpeg command line (INFERNIX_FFMPEG or ffmpeg on
PATH) from synthetic frames, whose raw frames go to the reference with the container's frame count and rate.

  python -I -B generate_video_golden.py <artifact> <golden.json> <resources dir>
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY))

import numpy as np  # noqa: E402
import transformers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast  # noqa: E402
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor  # noqa: E402
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor  # noqa: E402
from transformers.video_utils import VideoMetadata  # noqa: E402

from tools.artifact.reader import Artifact  # noqa: E402

RESOURCES = ("resource/text/tokenizer.json", "resource/text/tokenizer_config.json",
             "resource/text/generation_config.json", "resource/text/chat_template.jinja",
             "resource/vision/preprocessor_config.json", "resource/vision/video_preprocessor_config.json")
WRAPPER = "<|vision_start|><|video_pad|><|vision_end|>"
TEXT = "Describe the video."

# (name, frames in the container, its rate, width, height): two frames (the fewest a video holds), three (an odd
# count, padded to a temporal pair), four sampled from eight at 4 fps (indices 0, 2, 5, 7), nine at 2 fps (odd),
# and a non-square size.
CASES = (
    ("two frames at 2 fps, 64x64", 2, 2.0, 64, 64),
    ("three frames at 2 fps, 64x64", 3, 2.0, 64, 64),
    ("eight frames at 4 fps sampled to four, 96x64", 8, 4.0, 96, 64),
    ("nine frames at 2 fps, 64x96", 9, 2.0, 64, 96),
)


def frames(count: int, width: int, height: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    out = np.empty((count, height, width, 3), dtype=np.uint8)
    for t in range(count):
        out[t, ..., 0] = (x * 3 + t * 29) % 256
        out[t, ..., 1] = (y * 5 + t * 53) % 256
        out[t, ..., 2] = (x + y + t * 17) % 256
    return out


def encode(video: np.ndarray, fps: float) -> bytes:
    ffmpeg = os.environ.get("INFERNIX_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found: set INFERNIX_FFMPEG")
    count, height, width, _ = video.shape
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "video.mkv"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                        "-r", f"{fps:g}", "-i", "-", "-c:v", "ffv1", "-pix_fmt", "bgr0", "-fflags", "+bitexact",
                        "-flags:v", "+bitexact", str(target)], input=video.tobytes(), check=True)
        return target.read_bytes()


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    artifact_path, golden_path, resources_dir = sys.argv[1:]
    target = Path(resources_dir)
    target.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {}
    with Artifact.open(artifact_path) as artifact:
        for name in RESOURCES:
            payloads[name.rsplit("/", 1)[1]] = artifact.read_object(name)
    for name, payload in payloads.items():
        (target / name).write_bytes(payload)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(target)
    image_config = json.loads(payloads["preprocessor_config.json"])
    video_config = json.loads(payloads["video_preprocessor_config.json"])
    for config in (image_config, video_config):
        for key in ("processor_class", "image_processor_type", "video_processor_type"):
            config.pop(key, None)
    processor = Qwen3VLProcessor(image_processor=Qwen2VLImageProcessorFast(**image_config), tokenizer=tokenizer,
                                 video_processor=Qwen3VLVideoProcessor(**video_config),
                                 chat_template=payloads["chat_template.jinja"].decode("utf-8"))
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": TEXT}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if rendered.count(WRAPPER) != 1:
        raise SystemExit(f"the template did not render one video wrapper: {rendered!r}")

    cases = []
    for name, count, fps, width, height in CASES:
        video = frames(count, width, height)
        encoded = encode(video, fps)

        def run(text: str) -> tuple[list[int], dict]:
            metadata = VideoMetadata(total_num_frames=count, fps=fps, frames_indices=list(range(count)))
            out = processor(text=[text], videos=[video], video_metadata=[metadata], do_sample_frames=True,
                            return_metadata=True)
            sampled = out["video_metadata"][0] if "video_metadata" in out else None
            return [int(i) for i in out["input_ids"][0]], {
                "grid": [int(v) for v in out["video_grid_thw"][0]],
                "indices": [int(i) for i in sampled.frames_indices] if sampled is not None else None,
            }

        ids, info = run(rendered.replace(WRAPPER, "<|video_pad|>"))
        ids_token_only, _ = run(rendered)
        cases.append({"name": name, "frames": count, "fps": fps, "width": width, "height": height,
                      "video_mkv_base64": base64.b64encode(encoded).decode("ascii"), "grid": info["grid"],
                      "indices": info["indices"], "ids": ids, "ids_token_only": ids_token_only})
        print(f"{name}: grid {info['grid']}, indices {info['indices']}, {len(ids)} ids "
              f"(token-only rendering {len(ids_token_only)})")

    golden = {
        "transformers_version": transformers.__version__,
        "rule": "the template's whole video wrapper replaced (transformers up to 5.9; the checkpoints' versions)",
        "tokenizer_json_bytes": len(payloads["tokenizer.json"]),
        "tokenizer_json_sha256": hashlib.sha256(payloads["tokenizer.json"]).hexdigest(),
        "text": TEXT,
        "cases": cases,
    }
    Path(golden_path).write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
