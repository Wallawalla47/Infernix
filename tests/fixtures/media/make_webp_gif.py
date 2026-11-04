"""Writes tests/fixtures/media/webp_gif.h: small WebP (lossless, lossy, animated, with alpha) and GIF (static and
animated) images made with Pillow and their first frames decoded by Pillow as RGB8, as transformers' load_image
returns them (candidates MJ-01).

python -I tests/fixtures/media/make_webp_gif.py tests/fixtures/media/webp_gif.h    (Pillow with WebP; 12.3 used)
"""
import base64
import io
import sys

import PIL
from PIL import Image

W, H = 21, 14  # odd width: the chroma subsampling's edge column


def pattern(shift, w=W, h=H):
    img = Image.new("RGB", (w, h))
    img.putdata([(((x * 13 + y * 7 + shift) % 256), ((x * 5 + y * 17 + 2 * shift) % 256), ((x * x + y * 3 + shift) % 256))
                 for y in range(h) for x in range(w)])
    return img


def save(images, **options):
    out = io.BytesIO()
    first, rest = images[0], images[1:]
    if rest:
        first.save(out, save_all=True, append_images=rest, **options)
    else:
        first.save(out, **options)
    return out.getvalue()


def smooth():
    img = Image.new("RGB", (W, H))
    img.putdata([(40 + 8 * x, 60 + 10 * y, 200 - 4 * x - 3 * y) for y in range(H) for x in range(W)])
    return img


def first_frame_rgb(data):
    # As transformers' load_image (5.5): Image.open, then convert("RGB"), which drops an alpha channel.
    img = Image.open(io.BytesIO(data))
    img.seek(0)
    return img.convert("RGB").tobytes()


alpha = pattern(0).convert("RGBA")
alpha.putalpha(Image.linear_gradient("L").resize((W, H)))
cases = [
    ("webp lossless", save([pattern(0)], format="WEBP", lossless=True)),
    ("webp lossy q80", save([pattern(0)], format="WEBP", quality=80)),
    ("webp lossy q80 smooth", save([smooth()], format="WEBP", quality=80)),
    ("webp lossy q100 smooth", save([smooth()], format="WEBP", quality=100)),
    ("webp lossy q80 20x15", save([pattern(5, 20, 15)], format="WEBP", quality=80)),
    ("webp lossy alpha q80", save([alpha], format="WEBP", quality=80)),
    ("webp animated lossy", save([pattern(0), pattern(91)], format="WEBP", quality=80, duration=100, loop=0)),
    ("webp animated lossless", save([pattern(0), pattern(91)], format="WEBP", lossless=True, duration=100, loop=0)),
    ("webp lossless alpha", save([alpha], format="WEBP", lossless=True)),
    ("gif static", save([pattern(0).quantize(256)], format="GIF")),
    ("gif animated", save([pattern(0).quantize(256), pattern(91).quantize(256)], format="GIF", duration=100, loop=0)),
]
lines = ["#pragma once", "", "#include <array>", "#include <string_view>", "", "namespace webp_gif_fixtures {", "",
         f"// WebP and GIF inputs made with Pillow {PIL.__version__} (small patterns and gradients; the animated ones have",
         "// two frames) and Pillow's RGB8 decode of each first frame (Image.open, seek(0), convert(\"RGB\"), as",
         "// transformers' load_image: an alpha channel is dropped): candidates MJ-01, written by make_webp_gif.py.",
         "struct Case {", "    std::string_view name;", "    int width;", "    int height;",
         "    std::string_view encoded;", "    std::string_view rgb;", "};", "",
         f"inline constexpr std::array<Case, {len(cases)}> cases{{{{"]
for name, data in cases:
    rgb = first_frame_rgb(data)
    w, h = Image.open(io.BytesIO(data)).size
    assert len(rgb) == w * h * 3
    lines.append(f"    Case{{\"{name}\", {w}, {h},")
    lines.append(f"         \"{base64.b64encode(data).decode()}\",")
    lines.append(f"         \"{base64.b64encode(rgb).decode()}\"}},")
lines += ["}};", "", "} // namespace webp_gif_fixtures", ""]
open(sys.argv[1], "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("wrote", sys.argv[1], [(n, len(d)) for n, d in cases])
