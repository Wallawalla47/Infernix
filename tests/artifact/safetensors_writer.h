#pragma once

// A minimal safetensors writer for tests (https://github.com/huggingface/safetensors): the 8-byte little-endian
// header length, the JSON header (space-padded to 8 bytes) and the row-major data. Values are written as F32, BF16 or
// F16; the BF16 and F16 encodings are exact only for values those formats represent (the tests use multiples of
// 1/64 below 1).

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace infernix::test::safetensors_writer {

struct TensorSpec {
    std::string name, dtype;
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

inline std::uint16_t f32_to_bf16_truncate(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return static_cast<std::uint16_t>(bits >> 16);
}

inline std::uint16_t f32_to_f16_exact(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    const std::uint32_t sign = (bits >> 16) & 0x8000U;
    if ((bits & 0x7FFFFFFFU) == 0) { return static_cast<std::uint16_t>(sign); }
    const int exponent      = static_cast<int>((bits >> 23) & 0xFF) - 127 + 15;
    const std::uint32_t man = (bits >> 13) & 0x3FFU;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | man);
}

// Values k / 64 for k in [-31, 31], exact in F16 and BF16.
inline std::vector<float> pattern(std::size_t count, int seed) {
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = static_cast<float>(static_cast<int>((i * 37 + static_cast<std::size_t>(seed) * 11) % 63) - 31) / 64.0F;
    }
    return out;
}

// `edit` may alter the header JSON before it is written (to build invalid files).
inline void write(const std::filesystem::path& path, const std::vector<TensorSpec>& tensors,
                  const std::function<void(nlohmann::json&)>& edit = {}) {
    nlohmann::json header = nlohmann::json::object();
    std::vector<std::uint8_t> data;
    for (const TensorSpec& t : tensors) {
        const std::uint64_t begin = data.size();
        for (const float v : t.values) {
            if (t.dtype == "F32") {
                std::uint8_t b[4];
                std::memcpy(b, &v, 4);
                data.insert(data.end(), b, b + 4);
            } else {
                const std::uint16_t h = t.dtype == "BF16" ? f32_to_bf16_truncate(v) : f32_to_f16_exact(v);
                data.push_back(static_cast<std::uint8_t>(h & 0xFF));
                data.push_back(static_cast<std::uint8_t>(h >> 8));
            }
        }
        header[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {begin, data.size()}}};
    }
    header["__metadata__"] = {{"format", "pt"}};
    if (edit) { edit(header); }
    std::string text = header.dump();
    while (text.size() % 8 != 0) { text.push_back(' '); }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t length = text.size();
    for (int i = 0; i < 8; ++i) { out.put(static_cast<char>((length >> (8 * i)) & 0xFF)); }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

inline void write_json(const std::filesystem::path& path, const nlohmann::json& json) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << json.dump(2);
}

// A fresh directory under the temp directory, removed on destruction.
struct TempDir {
    std::filesystem::path path;
    explicit TempDir(const std::string& name) {
        static int counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("infernix_" + name + "_" + std::to_string(++counter) + "_" +
                std::to_string(std::filesystem::file_time_type::clock::now().time_since_epoch().count()));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
};

} // namespace infernix::test::safetensors_writer
