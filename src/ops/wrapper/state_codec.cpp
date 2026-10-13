#include "infernix/ops/state_codec.h"

#include "ops/launcher/state_codec.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

void validate(const Tensor& state, const void* packed, PackedStateFormat format, const char* op) {
    if (state.dtype != DType::FP32 || !state.is_contiguous() || state.data == nullptr ||
        packed == nullptr) {
        throw std::invalid_argument(std::string(op) +
                                    ": the state must be a contiguous non-null FP32 tensor");
    }
    if (state.numel() % 4 != 0 || (reinterpret_cast<std::uintptr_t>(state.data) & 0xfU) != 0 ||
        (reinterpret_cast<std::uintptr_t>(packed) & 0x3U) != 0) {
        throw std::invalid_argument(std::string(op) +
                                    ": needs a multiple of four elements, a 16-byte aligned "
                                    "state and a 4-byte aligned packed buffer");
    }
    const auto* state_begin  = static_cast<const std::byte*>(state.data);
    const auto* packed_begin = static_cast<const std::byte*>(packed);
    const std::size_t packed_size = packed_state_bytes(state.numel(), format);
    if (packed_begin < state_begin + state.bytes() && state_begin < packed_begin + packed_size) {
        throw std::invalid_argument(std::string(op) + ": state and packed buffer must not alias");
    }
}

} // namespace

std::size_t packed_state_bytes(std::int64_t elements, PackedStateFormat format) {
    if (elements < 0) { throw std::invalid_argument("packed_state_bytes: negative element count"); }
    const auto count = static_cast<std::size_t>(elements);
    switch (format) {
    case PackedStateFormat::Fp24: return 3U * count;
    case PackedStateFormat::Bf16: return 2U * count;
    }
    throw std::invalid_argument("packed_state_bytes: unknown format");
}

void pack_fp32_state(const Tensor& source, void* packed, PackedStateFormat format,
                     cudaStream_t stream) {
    validate(source, packed, format, "pack_fp32_state");
    detail::pack_fp32_state_launch(source, packed, format, stream);
}

void unpack_fp32_state(const void* packed, Tensor& destination, PackedStateFormat format,
                       cudaStream_t stream) {
    validate(destination, packed, format, "unpack_fp32_state");
    detail::unpack_fp32_state_launch(packed, destination, format, stream);
}

} // namespace infernix::ops
