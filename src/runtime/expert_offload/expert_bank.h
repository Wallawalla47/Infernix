#pragma once

// A layer's routed-expert bank as the offloaded MoE reads it (docs/maintainer/qwen3_5-122b-offload-
// design.md §7: the shared expert-offload component that Qwen3.8-Flash-Next's and Qwen3.5's Programs
// instantiate). The planes are views into the Model's materialized weights; the scalars are derived
// once at load.

#include "core/weight_view.h"
#include "ops/offloaded_sparse_moe/cpu/ggml_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/offloaded_sparse_moe/csf_record.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::runtime::expert_offload {

// W4A4 (Qwen3.8-Flash-Next design §16.2): alpha is derived once from the stored words, alpha =
// fl32(weight_scale_2 * input_scale), so every route (GPU frame, GPU staging, CPU) reads the same
// values. W4A16 (§16.2.1): no input scales; alpha holds the stored multipliers themselves.
struct ExpertBank {
    ExpertBankPlanes planes;
    ops::offloaded_moe::ExpertActivation activation = ops::offloaded_moe::ExpertActivation::kA4;
    std::vector<ops::offloaded_moe::ExpertScales> scales; // [experts]
    // The bank's CSF tables when its pinned records were compressed at load (compress_banks; record e's table at
    // csf.host / csf.device + e * table words): every reader then takes them. Empty: the records as stored.
    ops::offloaded_moe::csf::BankTables csf;

    [[nodiscard]] ops::offloaded_moe::ExpertShape shape() const noexcept {
        return {static_cast<int>(planes.hidden), static_cast<int>(planes.intermediate)};
    }
    // Record e as the CPU reads it (its table when compressed).
    [[nodiscard]] ops::offloaded_moe::Rg16Record record(std::uint32_t e) const noexcept {
        const auto* data = reinterpret_cast<const std::uint8_t*>(planes.records) + static_cast<std::uint64_t>(e) * planes.record_stride;
        return {data, csf ? csf.host + static_cast<std::size_t>(e) * ops::offloaded_moe::csf::Geometry::of(shape()).table_words()
                          : nullptr};
    }
};

// The CSF tables of a Model's compressed banks: one mapped pinned allocation, each compressed bank's
// (experts x table words) in turn. Move-only; the Model owns it as long as its banks.
class CsfTables {
public:
    CsfTables() = default;
    CsfTables(std::size_t words, std::uint64_t expanded_bytes, std::uint64_t compressed_bytes);
    ~CsfTables();
    CsfTables(CsfTables&& other) noexcept;
    CsfTables& operator=(CsfTables&& other) noexcept;
    CsfTables(const CsfTables&)            = delete;
    CsfTables& operator=(const CsfTables&) = delete;

    [[nodiscard]] std::uint32_t* host() const noexcept { return host_; }
    [[nodiscard]] const std::uint32_t* device() const noexcept { return device_; }
    [[nodiscard]] bool empty() const noexcept { return host_ == nullptr; }
    // The compressed records' bytes against their expanded bytes (every compressed bank's).
    [[nodiscard]] std::uint64_t expanded_bytes() const noexcept { return expanded_; }
    [[nodiscard]] std::uint64_t compressed_bytes() const noexcept { return compressed_; }
    void set_compressed_bytes(std::uint64_t bytes) noexcept { compressed_ = bytes; }

private:
    std::uint32_t* host_         = nullptr;
    const std::uint32_t* device_ = nullptr;
    std::uint64_t expanded_ = 0, compressed_ = 0;
};

// Compresses the records of every bank of `banks` the host holds in pinned memory (full mode) and stores as
// nvfp4_expert_rg16_v1 at a registered geometry into CSF records (ops/offloaded_sparse_moe/csf_record.h), in place at
// the bank's stride, on every core, and sets those banks' ExpertBank::csf; returns their tables. Every reader of a host
// record takes the tables (the CPU walks, the staging copies, promotions and the prefill ring), so outputs are
// unchanged while each copy and walk reads ~5 % fewer bytes. Other banks (streamed, device, GGUF, block-FP8) are left
// as they are. The records must be the load's own (materialized, not yet read by anything else). The loads call it with
// LoadOptions::expert_csf (--expert-csf).
[[nodiscard]] CsfTables compress_banks(std::span<ExpertBank> banks);

// The SSD tier's CSF slots (HostExpertTier::Options::csf): the geometry when every bank is an NVFP4 bank of one
// registered geometry and record stride, else none. The Programs ask it with --expert-csf.
[[nodiscard]] std::optional<ops::offloaded_moe::ExpertShape> tier_csf_shape(std::span<const ExpertBank* const> banks);

// The startup log's line on the banks' host records: "expert host records: CSF, <share> of their bytes" when any is
// compressed, else empty.
[[nodiscard]] std::string csf_summary(std::span<const ExpertBank* const> banks);

// A GGUF import's bank: its records' ggml formats for the CPU route (invalid for an NVFP4 bank).
[[nodiscard]] inline ops::offloaded_moe::GgmlExpertFormat ggml_format(const ExpertBank& bank) noexcept {
    if (bank.planes.gate_up_format == QType::NVFP4_MUL) { return {}; }
    return {.gate_up        = bank.planes.gate_up_format,
            .down           = bank.planes.down_format,
            .gate_row_bytes = static_cast<std::uint32_t>(bank.planes.gate_row_bytes),
            .down_row_bytes = static_cast<std::uint32_t>(bank.planes.down_row_bytes)};
}

// The scalars of a W4A16 nvfp4_expert_rg16_v1 bank (Qwen3.8-Flash-Next design §16.2.1): no input
// scales; alpha_* are the record's stored multipliers. Throws std::invalid_argument naming `name`
// unless every multiplier is positive finite FP32.
[[nodiscard]] std::vector<ops::offloaded_moe::ExpertScales> a16_expert_scales(const ExpertBankPlanes& planes,
                                                                             const std::string& name);

// A complete W4A16 nvfp4_expert_rg16_v1 bank of `experts` x [hidden, intermediate] in one
// host-readable parent (a pinned routed bank), with its scalars from its scale tail.
[[nodiscard]] ExpertBank a16_nvfp4_bank(const WeightView& view, std::uint32_t experts, std::uint32_t hidden,
                                        std::uint32_t intermediate, const std::string& name);
// The same bank whose records the host cannot read, built from `multipliers`, the caller's copy of the
// bank's scale tail ([experts][3] FP32, kept alive as long as the bank): left in the artifact for the
// SSD tier (artifact::Residency::Streamed; no records), or resident in device memory only (a shared
// expert; device_records).
[[nodiscard]] ExpertBank a16_nvfp4_streamed_bank(const WeightView& view, const float* multipliers,
                                                 std::uint32_t experts, std::uint32_t hidden,
                                                 std::uint32_t intermediate, const std::string& name);
[[nodiscard]] ExpertBank a16_nvfp4_device_bank(const WeightView& view, const float* multipliers,
                                               std::uint32_t experts, std::uint32_t hidden,
                                               std::uint32_t intermediate, const std::string& name);

} // namespace infernix::runtime::expert_offload
