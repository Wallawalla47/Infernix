// ninfer::ops - top_logprobs wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/top_logprobs.h"

#include "ops/launcher/top_logprobs.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("top_logprobs: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("top_logprobs: ") + label + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("top_logprobs: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("top_logprobs: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void top_logprobs(const Tensor& logits, std::int32_t valid_rows, Tensor& ids, Tensor& output,
                  cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("top_logprobs: logits must be BF16");
    }
    if (ids.dtype != DType::I32) { throw std::invalid_argument("top_logprobs: ids must be I32"); }
    if (output.dtype != DType::FP32) {
        throw std::invalid_argument("top_logprobs: output must be FP32");
    }

    require_rank_two(logits, "logits");
    require_rank_two(ids, "ids");
    require_rank_two(output, "output");
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("top_logprobs: valid_rows must be in [1, physical_rows]");
    }
    if (ids.ne[1] != logits.ne[1] || output.ne[0] != ids.ne[0] || output.ne[1] != ids.ne[1]) {
        throw std::invalid_argument("top_logprobs: ids and output must have shape [K, columns]");
    }
    if (ids.ne[0] > valid_rows) {
        throw std::invalid_argument("top_logprobs: K must be in [1, valid_rows]");
    }

    (void)logits.bytes();
    (void)ids.bytes();
    (void)output.bytes();
    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(ids, alignof(std::int32_t), "ids");
    require_accessible(output, alignof(float), "output");
    if (overlaps(ids, logits) || overlaps(output, logits) || overlaps(ids, output)) {
        throw std::invalid_argument("top_logprobs: outputs must not overlap logits or each other");
    }

    detail::top_logprobs_launch(logits, valid_rows, ids, output, stream);
}

} // namespace ninfer::ops
