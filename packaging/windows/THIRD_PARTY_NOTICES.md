# Third-party notices: Infernix Windows release

Infernix is licensed under the Apache License 2.0 (`LICENSE`). The release package also contains or
is built from the components below. Each one's licence text is in the `licenses` folder of the
package, under the file name given here.

## Compiled into the executables

| Component | Origin | Licence | File in `licenses` |
|---|---|---|---|
| NInfer | [Neroued/ninfer](https://github.com/Neroued/ninfer), the engine Infernix is based on | Apache-2.0 | `Infernix-LICENSE.txt` |
| XGrammar (adapted) | [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) | Apache-2.0 | `xgrammar-LICENSE.txt`, `xgrammar-NOTICE.txt` |
| DLPack headers (via XGrammar) | [dmlc/dlpack](https://github.com/dmlc/dlpack) | Apache-2.0 | `dlpack-LICENSE.txt` |
| PicoJSON (via XGrammar) | [kazuho/picojson](https://github.com/kazuho/picojson) | BSD-2-Clause | `picojson-LICENSE.txt` |
| Jinja template engine (adapted) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT; Unicode data under the Unicode licence | `llama-jinja-LICENSE.txt`, `llama-jinja-UNICODE-LICENSE.txt` |
| cpp-httplib | [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) | MIT | `cpp-httplib-LICENSE.txt` |
| JSON for Modern C++ | [nlohmann/json](https://github.com/nlohmann/json) | MIT | `nlohmann-json-LICENSE.txt` |
| spdlog, with its bundled {fmt} | [gabime/spdlog](https://github.com/gabime/spdlog), [fmtlib/fmt](https://github.com/fmtlib/fmt) | MIT | `spdlog-LICENSE.txt`, `fmt-LICENSE.txt` |
| utf8proc | [JuliaStrings/utf8proc](https://github.com/JuliaStrings/utf8proc) | MIT; Unicode data under the Unicode licence | `utf8proc-LICENSE.txt` |
| CUDA runtime (`cudart_static`) | NVIDIA CUDA Toolkit 13 | NVIDIA CUDA Toolkit EULA (the runtime is a redistributable component) | — |

## Shipped as separate DLLs

| DLLs | Component | Licence | File in `licenses` |
|---|---|---|---|
| `avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, `swresample-*.dll`, `swscale-*.dll` | [FFmpeg](https://ffmpeg.org), built by vcpkg without GPL or non-free components | LGPL-2.1-or-later | `FFmpeg-LICENSE.txt` |
| `libcurl.dll` | [curl](https://curl.se) (TLS through Windows Schannel) | curl licence (MIT-style) | `curl-LICENSE.txt` |
| `z.dll` | [zlib](https://zlib.net) | zlib licence | `zlib-LICENSE.txt` |

FFmpeg is used only through its DLLs, unmodified by Infernix, so it can be replaced by another build
of the same FFmpeg major versions. Its corresponding source is published next to the Infernix
release as `ffmpeg-<version>-source.zip`: the FFmpeg source release and the vcpkg port (build
script and patches) that built these DLLs. The FFmpeg source is also available from
[ffmpeg.org](https://ffmpeg.org/download.html).

## Not included

- The NVIDIA driver supplies `nvcuda.dll` and `nvml.dll`.
- The Microsoft Visual C++ runtime comes from the Visual C++ Redistributable for Visual Studio
  2015-2026, which Windows usually already has.
- Models are downloaded separately and carry their own licences (see each model card).
