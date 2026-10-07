// infernix::ops - constrain_logits wrapper: public contract validation and launcher dispatch.
#include "infernix/ops/token_constraint.h"

#include "ops/launcher/token_constraint.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

void require(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
             const char* label) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("constrain_logits: ") + label + " has a wrong dtype");
    }
    if (tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("constrain_logits: ") + label +
                                    " has a wrong shape");
    }
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("constrain_logits: ") + label +
                                    " must be contiguous and non-null");
    }
    (void)tensor.bytes();
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void constrain_logits(Tensor& logits, Tensor* argmax, const Tensor& descriptors,
                      const Tensor& choices, const Tensor& choice_counts, std::int32_t valid_rows,
                      Tensor& records, cudaStream_t stream) {
    if (logits.ne[0] <= 0 || logits.ne[1] <= 0) {
        throw std::invalid_argument("constrain_logits: logits must have positive dimensions");
    }
    const std::int32_t columns = logits.ne[1];
    const std::int32_t sets    = choice_counts.ne[0];
    require(logits, DType::BF16, logits.ne[0], columns, "logits");
    require(descriptors, DType::I32, columns, 1, "descriptors");
    if (sets <= 0) { throw std::invalid_argument("constrain_logits: choice table is empty"); }
    require(choice_counts, DType::I32, sets, 1, "choice_counts");
    require(choices, DType::I32, kTokenConstraintChoices, sets, "choices");
    require(records, DType::FP32, kTokenConstraintRecord, columns, "records");
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("constrain_logits: valid_rows must be in [1, physical_rows]");
    }
    const Tensor* outputs[] = {&logits, &records, argmax};
    const Tensor* inputs[]  = {&descriptors, &choices, &choice_counts};
    if (argmax != nullptr) { require(*argmax, DType::I32, columns, 1, "argmax"); }
    for (const Tensor* output : outputs) {
        if (output == nullptr) { continue; }
        for (const Tensor* other : outputs) {
            if (other != nullptr && other != output && overlaps(*output, *other)) {
                throw std::invalid_argument("constrain_logits: outputs must not overlap");
            }
        }
        for (const Tensor* input : inputs) {
            if (overlaps(*output, *input)) {
                throw std::invalid_argument("constrain_logits: outputs must not overlap inputs");
            }
        }
    }
    detail::constrain_logits_launch(logits, argmax, descriptors, choices, choice_counts,
                                    valid_rows, records, stream);
}

} // namespace infernix::ops
