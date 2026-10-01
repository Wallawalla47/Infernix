// ninfer::ops - speculative verification tree shape and tree-row validation.
#include "ninfer/ops/speculative_tree.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {

void validate_speculative_tree_shape(SpeculativeTreeShape shape) {
    if (shape.main_depth < 1 || shape.main_depth > kSpeculativeTreeMaxPathLength - 1) {
        throw std::invalid_argument("speculative tree: main depth must be in [1,15]");
    }
    if (shape.nodes < shape.main_depth + 2 || shape.nodes > kSpeculativeTreeMaxNodes) {
        throw std::invalid_argument(
            "speculative tree: nodes must exceed the main chain and be at most 32");
    }
    if (shape.paths < 2 || shape.paths > kSpeculativeTreeMaxPaths) {
        throw std::invalid_argument("speculative tree: paths must be in [2,8]");
    }
}

void validate_speculative_tree_rows(const Tensor& tree_rows, std::int32_t batch, const char* op) {
    if (tree_rows.data == nullptr || tree_rows.dtype != DType::I32 ||
        tree_rows.ne[0] != kSpeculativeTreeRowWords || tree_rows.ne[1] != batch ||
        tree_rows.ne[2] != 1 || tree_rows.ne[3] != 1 || !tree_rows.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": tree rows must be contiguous I32 [" +
                                    std::to_string(kSpeculativeTreeRowWords) + ",B]");
    }
}

} // namespace ninfer::ops
