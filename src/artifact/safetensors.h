#pragma once

#include "artifact/file_io.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace infernix::artifact {

// A tensor of a safetensors file (https://github.com/huggingface/safetensors): its element type as the header names it
// ("F32", "BF16", "F16", ...), its shape and its byte range in the file. The file is little-endian and row-major.
struct SafetensorsTensor {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t offset = 0; // from the start of the file
    std::uint64_t bytes  = 0;

    [[nodiscard]] std::uint64_t elements() const noexcept;
};

// A safetensors file's header, validated: every tensor's range lies in the data section, matches its dtype and
// shape, and no two ranges overlap. Tensor data is read on request.
class SafetensorsFile {
public:
    explicit SafetensorsFile(const std::filesystem::path& path);
    ~SafetensorsFile();
    SafetensorsFile(const SafetensorsFile&)            = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;
    [[nodiscard]] const std::map<std::string, SafetensorsTensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept { return metadata_; }
    // Throws ArtifactError naming the tensor when it is absent.
    [[nodiscard]] const SafetensorsTensor& tensor(const std::string& name) const;

    void read_into(const SafetensorsTensor& tensor, std::span<std::byte> destination) const;
    // The tensor's values as FP32 (F32, BF16 and F16 tensors; others are refused by name).
    [[nodiscard]] std::vector<float> read_f32(const std::string& name) const;

private:
    std::filesystem::path path_;
    std::unique_ptr<InputFile> file_;
    std::map<std::string, SafetensorsTensor> tensors_;
    std::map<std::string, std::string> metadata_;
};

// Bytes per element of a safetensors dtype name, 0 for a name this reader does not know.
[[nodiscard]] std::uint64_t safetensors_dtype_bytes(const std::string& dtype) noexcept;

} // namespace infernix::artifact
