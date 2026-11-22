// Inkling 2's vision path against tests/fixtures/inkling2/vision_golden.json (generate_vision_golden.py: transformers'
// image processor and SGLang's tower, run as they are):
// 1. Processor (CPU): each fixture PNG through the shared decoder (alpha kept) and frontend/image.h gives, cell by
//    cell, the reference processor's BF16 pixel words exactly (SHA-256 of each cell's [40][40][3] words).
// 2. Tower (GPU, the real artifact): the rows of each image against the reference tower's (FP32, stored BF16):
//    relative RMS <= 4e-2 and cosine >= 0.999 an image. Criterion stated before the first run: the tower keeps BF16
//    between its Ops as the reference's own BF16 run does (that run is 1.2-3.0 % from FP64), while the smallest
//    semantic slip measured on the FP64 oracle (the position table's axes swapped) moves the rows 8 %.
// 3. An image's rows are the same bits encoded first, after another image, and again.
//
//   INFERNIX_INKLING2_VISION_ARTIFACT  an artifact converted with the vision component (part 1 runs without it)
//   INFERNIX_INKLING2_VISION_GOLDEN    another golden (generate_vision_golden.py --large), in place of the fixture

#include <algorithm>

#include "core/arena.h"
#include "core/device.h"
#include "artifact/reader.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/execution/vision.h"
#include "models/inkling2/frontend/image.h"
#include "models/inkling2/load.h"
#include "models/qwen3_5/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace m = infernix::models::inkling2;
using infernix::DeviceContext;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

std::vector<std::uint8_t> base64(const std::string& text) {
    static const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<std::uint8_t> out;
    std::uint32_t buffer = 0;
    int bits             = 0;
    for (const char c : text) {
        if (c == '=') { break; }
        const auto at = alphabet.find(c);
        if (at == std::string::npos) { continue; }
        buffer = (buffer << 6) | static_cast<std::uint32_t>(at);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

float bf16(std::uint16_t w) {
    const std::uint32_t u = std::uint32_t{w} << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

m::VisionConfig inkling2_mini_vision() {
    m::VisionConfig c;
    c.hidden_size = 1024, c.layers = 23, c.heads = 16, c.intermediate = 4096, c.patch_size = 20, c.merge = 2;
    c.position_grid = 32, c.projection_hidden = 4096, c.output_size = 2304, c.layer_norm_eps = 1e-5F;
    c.rope_theta = 10000.0F, c.image_token = "<|unused_200054|>", c.image_start_token = "<|content_image|>";
    return c;
}

// The cell-major [cells][40][40][3] words of an image's merge-major patches.
std::vector<std::uint16_t> cell_words(const m::ProcessedImage& image, const m::VisionConfig& c, std::int32_t cell) {
    const int p = static_cast<int>(c.patch_size), f = static_cast<int>(c.cell()), mm = static_cast<int>(c.merge);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(f) * f * 3);
    for (int y = 0; y < f; ++y) {
        for (int x = 0; x < f; ++x) {
            const int patch = cell * mm * mm + (y / p) * mm + x / p;
            const std::size_t at =
                (static_cast<std::size_t>(patch) * p * p + static_cast<std::size_t>(y % p) * p + x % p) * 3;
            for (int ch = 0; ch < 3; ++ch) { out[(static_cast<std::size_t>(y) * f + x) * 3 + ch] = (*image.patches)[at + ch]; }
        }
    }
    return out;
}

} // namespace

int main() {
    try {
        // INFERNIX_INKLING2_VISION_GOLDEN: another golden of generate_vision_golden.py's (--large: 1,296 cells, past the
        // tower's 4,096-token chunks).
        const char* golden_override = std::getenv("INFERNIX_INKLING2_VISION_GOLDEN");
        const std::filesystem::path fixture =
            golden_override != nullptr ? std::filesystem::path(golden_override)
                                       : std::filesystem::path(INFERNIX_SOURCE_DIR) / "tests/fixtures/inkling2/vision_golden.json";
        std::ifstream file(fixture);
        const nlohmann::json golden = nlohmann::json::parse(file);
        const m::VisionConfig config = inkling2_mini_vision();

        std::vector<m::ProcessedImage> images;
        for (const auto& c : golden.at("cases")) {
            infernix::OwnedMedia media;
            media.bytes = base64(c.at("png").get<std::string>());
            const m::ProcessedImage image = m::process_image(media, config, 32768);
            const std::string name        = c.at("name").get<std::string>();
            check(image.rows == c.at("rows").get<int>() && image.cols == c.at("cols").get<int>(), name + ": cell grid");
            const auto& digests = c.at("cell_sha256");
            for (std::int32_t cell = 0; cell < image.cells() && cell < static_cast<std::int32_t>(digests.size()); ++cell) {
                const std::vector<std::uint16_t> words = cell_words(image, config, cell);
                const auto digest = infernix::models::qwen3_5::frontend::sha256(std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t*>(words.data()), words.size() * 2));
                check(infernix::models::qwen3_5::frontend::sha256_hex(digest) == digests[cell].get<std::string>(),
                      name + ": cell " + std::to_string(cell) + "'s pixel words differ from the reference processor's");
            }
            images.push_back(image);
        }
        std::cout << "processor: " << images.size() << " images compared\n";

        const char* artifact_path = std::getenv("INFERNIX_INKLING2_VISION_ARTIFACT");
        if (artifact_path == nullptr) {
            std::cout << (g_failures == 0 ? "PASS (processor only: no INFERNIX_INKLING2_VISION_ARTIFACT)\n" : "FAILED\n");
            return g_failures == 0 ? 0 : 1;
        }
        DeviceContext device(0);
        infernix::artifact::Reader reader(artifact_path);
        infernix::models::LoadOptions load;
        load.vision = true;
        auto model = m::materialize_model(m::plan_load(reader, load), device);
        auto parameters = std::make_unique<m::Parameters>(*model);
        std::int32_t max_cells = 1;
        for (const auto& image : images) { max_cells = std::max(max_cells, image.cells()); }
        auto tower = std::make_unique<m::VisionTower>(*model->config().vision, *parameters->vision,
                                                      static_cast<std::uint32_t>(max_cells));
        const auto rows_of = [&](const m::ProcessedImage& image) {
            const std::uint16_t* rows = tower->encode(
                {image.rows, image.cols, std::span<const std::uint16_t>(*image.patches)}, device.stream);
            std::vector<std::uint16_t> out(static_cast<std::size_t>(image.cells()) * config.output_size);
            CUDA_CHECK(cudaMemcpyAsync(out.data(), rows, out.size() * 2, cudaMemcpyDeviceToHost, device.stream));
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            return out;
        };
        std::vector<std::vector<std::uint16_t>> first;
        std::size_t index = 0;
        for (const auto& c : golden.at("cases")) {
            const std::string name     = c.at("name").get<std::string>();
            const std::vector<std::uint8_t> raw = base64(c.at("rows_bf16").get<std::string>());
            const std::vector<std::uint16_t> got = rows_of(images[index]);
            check(raw.size() == got.size() * 2, name + ": the reference holds another row count");
            double err = 0, ref2 = 0, dot = 0, got2 = 0;
            for (std::size_t i = 0; i < got.size() && 2 * i + 1 < raw.size(); ++i) {
                const double r = bf16(static_cast<std::uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8)));
                const double g = bf16(got[i]);
                err += (g - r) * (g - r), ref2 += r * r, dot += g * r, got2 += g * g;
            }
            if (const char* dump = std::getenv("INFERNIX_INKLING2_VISION_DUMP")) { // diagnostics: the rows, raw BF16
                std::ofstream out(std::string(dump) + "." + name + ".bf16", std::ios::binary);
                out.write(reinterpret_cast<const char*>(got.data()), static_cast<std::streamsize>(got.size() * 2));
            }
            const double rel = std::sqrt(err / ref2), cosine = dot / std::sqrt(ref2 * got2);
            std::cout << name << ": tower vs reference rel RMS " << rel << ", cosine " << cosine << '\n';
            check(rel <= 4e-2 && cosine >= 0.999, name + ": the tower's rows are outside the criterion");
            first.push_back(got);
            ++index;
        }
        // The same bits encoded again, and after the other image.
        for (std::size_t i = images.size(); i-- > 0;) {
            check(rows_of(images[i]) == first[i], "an image's rows depend on what the tower encoded before it");
        }
        // Each image's encode time (resident), reported.
        const auto encode_ms = [&](m::VisionTower& t, const m::ProcessedImage& image) {
            cudaEvent_t a, b;
            CUDA_CHECK(cudaEventCreate(&a));
            CUDA_CHECK(cudaEventCreate(&b));
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                CUDA_CHECK(cudaEventRecord(a, device.stream));
                (void)t.encode({image.rows, image.cols, std::span<const std::uint16_t>(*image.patches)}, device.stream);
                CUDA_CHECK(cudaEventRecord(b, device.stream));
                CUDA_CHECK(cudaEventSynchronize(b));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, static_cast<double>(ms));
            }
            CUDA_CHECK(cudaEventDestroy(a));
            CUDA_CHECK(cudaEventDestroy(b));
            return best;
        };
        std::vector<double> resident_ms;
        for (const auto& image : images) { resident_ms.push_back(encode_ms(*tower, image)); }
        std::size_t free_resident = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_resident, &total));

        // 4. --vision-offload on (the tower's weights in pinned host memory, streamed through two device slots a
        //    layer): every image's rows the resident tower's bit for bit, in order, then again in reverse; the device
        //    memory it frees (the model and tower loaded each way) and each encode's time, reported.
        tower.reset();
        parameters.reset();
        model.reset();
        load.vision_offload = true;
        model      = m::materialize_model(m::plan_load(reader, load), device);
        parameters = std::make_unique<m::Parameters>(*model);
        if (!model->vision_host()) { throw std::runtime_error("--vision-offload on: the model holds no pinned tower"); }
        tower = std::make_unique<m::VisionTower>(*model->config().vision, *parameters->vision,
                                                 static_cast<std::uint32_t>(max_cells), &*model->vision_host(),
                                                 device.transfer_stream);
        std::size_t free_offloaded = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_offloaded, &total));
        for (std::size_t i = 0; i < images.size(); ++i) {
            check(rows_of(images[i]) == first[i], "an offloaded tower's rows differ from the resident tower's");
        }
        for (std::size_t i = images.size(); i-- > 0;) {
            check(rows_of(images[i]) == first[i], "an offloaded tower's rows depend on what it encoded before");
        }
        std::cout << "offload: rows equal the resident tower's bit for bit (" << images.size() << " images, twice); "
                  << "device memory freed " << (static_cast<double>(free_offloaded) - free_resident) / (1 << 20)
                  << " MiB (reported); streamed " << tower->streamed_bytes() / double(1 << 20) << " MiB an encode\n";
        for (std::size_t i = 0; i < images.size(); ++i) {
            std::cout << "  image " << i << " (" << images[i].cells() << " cells): encode " << resident_ms[i]
                      << " ms resident, " << encode_ms(*tower, images[i]) << " ms offloaded (best of 3, reported)\n";
        }
        std::cout << (g_failures == 0 ? "PASS\n" : "FAILED\n");
        return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
