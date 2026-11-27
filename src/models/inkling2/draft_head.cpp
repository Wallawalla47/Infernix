#include "models/inkling2/draft_head.h"

#include "core/device.h"
#include "infernix/ops/inkling2_dspark.h"
#include "infernix/ops/q8_g32.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::inkling2 {
namespace {

// The list's length (2026-11-24, RTX 5090, DSpark auto, nine English, code, Chinese and other-language prompts, on the
// default, --q8-weights and --a8-activations routes): 32K, 49K and 65K all beat the full head on every prompt and route;
// 65K had the best worst case (+2.4 % or more on each route) and holds 99.0 % of held-out positions. 16K lost on
// structured text.
constexpr std::int32_t kRows = 65536;

std::size_t align_256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

std::int32_t head_rows(std::size_t vocabulary) {
    return static_cast<std::int32_t>(std::min<std::size_t>(kRows, vocabulary / 32 * 32));
}

Weight bf16_matrix(void* data, std::int32_t n, std::int32_t k) {
    Weight w;
    w.qdata  = data;
    w.qtype  = QType::BF16;
    w.layout = QuantLayout::Contiguous;
    w.ndim   = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

} // namespace

std::vector<std::int32_t> draft_head_ids(std::span<const std::int32_t> rank) {
    std::vector<bool> seen(rank.size(), false);
    for (const std::int32_t id : rank) {
        if (id < 0 || static_cast<std::size_t>(id) >= rank.size() || seen[static_cast<std::size_t>(id)]) {
            throw std::invalid_argument("inkling2 draft head: proposal/token_rank must be a permutation of the vocabulary");
        }
        seen[static_cast<std::size_t>(id)] = true;
    }
    std::vector<std::int32_t> ids(rank.begin(), rank.begin() + head_rows(rank.size()));
    std::sort(ids.begin(), ids.end());
    return ids;
}

std::uint64_t draft_head_bytes(std::int32_t rows, std::int32_t hidden, std::int32_t markov_rank) {
    const std::uint64_t fold = markov_rank == 0 ? 0
                                                : align_256(ops::q8_g32_bytes(rows, markov_rank)) +
                                                      align_256(static_cast<std::size_t>(rows) * 2);
    return align_256(ops::q8_g32_bytes(rows, hidden)) + fold + align_256(static_cast<std::size_t>(rows) * 4);
}

DraftHead build_draft_head(const Weight& head, const Weight* fold, const std::uint16_t* fold_bias,
                           std::span<const std::int32_t> token_ids, DeviceContext& device) {
    const auto is_bf16 = [](const Weight& w) {
        return w.qdata != nullptr && w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous;
    };
    if (!is_bf16(head) || token_ids.empty() ||
        (fold != nullptr && (!is_bf16(*fold) || fold_bias == nullptr || head.n != fold->n))) {
        throw std::invalid_argument("inkling2 draft head: the BF16 target head (and the Fold and fold_bias) of one "
                                    "vocabulary");
    }
    const auto rows = static_cast<std::int32_t>(token_ids.size());
    const std::int32_t H = head.k, rank = fold != nullptr ? fold->k : 0;
    const std::size_t head_bytes = align_256(ops::q8_g32_bytes(rows, H));
    const std::size_t fold_bytes = rank == 0 ? 0 : align_256(ops::q8_g32_bytes(rows, rank));
    const std::size_t bias_bytes = rank == 0 ? 0 : align_256(static_cast<std::size_t>(rows) * 2);

    DraftHead out;
    out.rows    = rows;
    out.storage = DeviceBuffer(draft_head_bytes(rows, H, rank));
    auto* base  = static_cast<std::uint8_t*>(out.storage.p);
    auto* q_head = base, *q_fold = base + head_bytes;
    auto* bias = reinterpret_cast<std::uint16_t*>(q_fold + fold_bytes);
    auto* ids  = reinterpret_cast<std::int32_t*>(q_fold + fold_bytes + bias_bytes);
    CUDA_CHECK(cudaMemcpy(ids, token_ids.data(), token_ids.size_bytes(), cudaMemcpyHostToDevice));

    // The listed rows in BF16, then their 8-bit form.
    const std::size_t head_words = align_256(static_cast<std::size_t>(rows) * H * 2);
    DeviceBuffer scratch(head_words + align_256(static_cast<std::size_t>(rows) * rank * 2));
    auto* bf16_head = static_cast<std::uint16_t*>(scratch.p);
    namespace ik    = ops::inkling2;
    ik::dspark_gather_rows(static_cast<const std::uint16_t*>(head.qdata), H, ids, rows, bf16_head, device.stream);
    const std::size_t head_codes = static_cast<std::size_t>(rows) * ops::q8_g32_padded_columns(H);
    out.head = ops::quantize_bf16_q8_g32(bf16_matrix(bf16_head, rows, H), q_head, q_head + head_codes, device.stream);
    if (fold != nullptr) {
        auto* bf16_fold = reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(scratch.p) + head_words);
        ik::dspark_gather_rows(static_cast<const std::uint16_t*>(fold->qdata), rank, ids, rows, bf16_fold, device.stream);
        ik::dspark_gather_rows(fold_bias, 1, ids, rows, bias, device.stream);
        const std::size_t fold_codes = static_cast<std::size_t>(rows) * ops::q8_g32_padded_columns(rank);
        const Weight q8_fold =
            ops::quantize_bf16_q8_g32(bf16_matrix(bf16_fold, rows, rank), q_fold, q_fold + fold_codes, device.stream);
        out.fold_codes  = static_cast<const std::int8_t*>(q8_fold.qdata);
        out.fold_scales = static_cast<const std::uint16_t*>(q8_fold.scales);
        out.fold_bias   = bias;
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    out.token_ids = ids;
    return out;
}

} // namespace infernix::models::inkling2
