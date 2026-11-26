#include "artifact/safetensors.h"

#include "artifact/schema.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace infernix::artifact {
namespace {

// The format caps the header at 100 MB; a larger length is a corrupt or foreign file.
constexpr std::uint64_t kMaxHeaderBytes = 100ULL << 20;

std::uint64_t read_u64_le(const std::byte* bytes) {
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) { value = (value << 8) | static_cast<std::uint8_t>(bytes[i]); }
    return value;
}

float bf16_to_f32(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

float f16_to_f32(std::uint16_t bits) {
    const std::uint32_t sign     = (bits & 0x8000U) << 16;
    std::uint32_t exponent       = (bits >> 10) & 0x1FU;
    std::uint32_t mantissa       = bits & 0x3FFU;
    std::uint32_t word           = 0;
    if (exponent == 0x1FU) {
        word = sign | 0x7F800000U | (mantissa << 13);
    } else if (exponent != 0) {
        word = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    } else if (mantissa != 0) { // subnormal: normalize
        exponent = 113;
        while ((mantissa & 0x400U) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        word = sign | (exponent << 23) | ((mantissa & 0x3FFU) << 13);
    } else {
        word = sign;
    }
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

} // namespace

std::uint64_t SafetensorsTensor::elements() const noexcept {
    std::uint64_t count = 1;
    for (const std::uint64_t extent : shape) { count *= extent; }
    return count;
}

std::uint64_t safetensors_dtype_bytes(const std::string& dtype) noexcept {
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") { return 8; }
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") { return 4; }
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") { return 2; }
    if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" || dtype == "F8_E5M2") { return 1; }
    return 0;
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path)
    : path_(path), file_(std::make_unique<InputFile>(path)) {
    const std::uint64_t total = file_->bytes();
    if (total < 8) { throw ArtifactError(path_.string() + ": not a safetensors file (shorter than its header length)"); }
    std::byte length_bytes[8];
    file_->read_exact(0, std::span(length_bytes));
    const std::uint64_t header_bytes = read_u64_le(length_bytes);
    if (header_bytes == 0 || header_bytes > kMaxHeaderBytes || header_bytes > total - 8) {
        throw ArtifactError(path_.string() + ": not a safetensors file (header length " +
                            std::to_string(header_bytes) + " of a " + std::to_string(total) + "-byte file)");
    }
    std::string header(header_bytes, '\0');
    file_->read_exact(8, std::as_writable_bytes(std::span(header.data(), header.size())));
    Json json;
    try {
        json = Json::parse(header);
    } catch (const Json::exception& error) {
        throw ArtifactError(path_.string() + ": the safetensors header is not JSON (" + error.what() + ")");
    }
    if (!json.is_object()) { throw ArtifactError(path_.string() + ": the safetensors header is not a JSON object"); }
    const std::uint64_t data_begin = 8 + header_bytes;
    const std::uint64_t data_bytes = total - data_begin;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& [name, entry] : json.items()) {
        if (name == "__metadata__") {
            if (!entry.is_object()) { throw ArtifactError(path_.string() + ": __metadata__ is not an object"); }
            for (const auto& [key, value] : entry.items()) {
                if (value.is_string()) { metadata_[key] = value.get<std::string>(); }
            }
            continue;
        }
        const auto fail = [&](const std::string& what) {
            throw ArtifactError(path_.string() + ": tensor '" + name + "': " + what);
        };
        if (!entry.is_object() || !entry.contains("dtype") || !entry.contains("shape") ||
            !entry.contains("data_offsets")) {
            fail("needs dtype, shape and data_offsets");
        }
        SafetensorsTensor tensor;
        tensor.dtype = entry.at("dtype").get<std::string>();
        const std::uint64_t element_bytes = safetensors_dtype_bytes(tensor.dtype);
        if (element_bytes == 0) { fail("unknown dtype " + tensor.dtype); }
        // The element count is checked as it is formed: a product that wraps would read as a small (or zero) count
        // and pass the data range check below.
        std::uint64_t count = 1;
        for (const auto& extent : entry.at("shape")) {
            const std::uint64_t value = extent.get<std::uint64_t>();
            if (value != 0 && count > std::numeric_limits<std::uint64_t>::max() / value) {
                fail("its shape's element count exceeds 64 bits");
            }
            count *= value;
            tensor.shape.push_back(value);
        }
        const auto& offsets = entry.at("data_offsets");
        if (!offsets.is_array() || offsets.size() != 2) { fail("data_offsets is not a [begin, end] pair"); }
        const std::uint64_t begin = offsets[0].get<std::uint64_t>(), end = offsets[1].get<std::uint64_t>();
        if (end < begin || end > data_bytes) { fail("data_offsets lie outside the data section"); }
        if (tensor.elements() > std::numeric_limits<std::uint64_t>::max() / element_bytes ||
            tensor.elements() * element_bytes != end - begin) {
            fail("data_offsets do not match its dtype and shape");
        }
        tensor.offset = data_begin + begin;
        tensor.bytes  = end - begin;
        ranges.emplace_back(begin, end);
        tensors_.emplace(name, std::move(tensor));
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            throw ArtifactError(path_.string() + ": two tensors' data ranges overlap");
        }
    }
}

SafetensorsFile::~SafetensorsFile() = default;

std::uint64_t SafetensorsFile::file_bytes() const noexcept { return file_->bytes(); }

const SafetensorsTensor& SafetensorsFile::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) { throw ArtifactError(path_.string() + ": no tensor '" + name + "'"); }
    return found->second;
}

void SafetensorsFile::read_into(const SafetensorsTensor& tensor, std::span<std::byte> destination) const {
    if (destination.size() != tensor.bytes) {
        throw std::invalid_argument("safetensors read: destination size does not match the tensor");
    }
    file_->read_exact(tensor.offset, destination);
}

std::vector<float> SafetensorsFile::read_f32(const std::string& name) const {
    const SafetensorsTensor& t = tensor(name);
    std::vector<float> values(static_cast<std::size_t>(t.elements()));
    if (t.dtype == "F32") {
        read_into(t, std::as_writable_bytes(std::span(values)));
        return values;
    }
    if (t.dtype != "BF16" && t.dtype != "F16") {
        throw ArtifactError(path_.string() + ": tensor '" + name + "' is " + t.dtype + "; F32, BF16 or F16 is read");
    }
    std::vector<std::uint16_t> halves(values.size());
    read_into(t, std::as_writable_bytes(std::span(halves)));
    const bool bf16 = t.dtype == "BF16";
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = bf16 ? bf16_to_f32(halves[i]) : f16_to_f32(halves[i]);
    }
    return values;
}

} // namespace infernix::artifact
