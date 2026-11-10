#include "runtime/expert_offload/expert_bank.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cmath>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace infernix::runtime::expert_offload {

namespace moe = ops::offloaded_moe;
namespace csf = ops::offloaded_moe::csf;

CsfTables::CsfTables(std::size_t words, std::uint64_t expanded_bytes, std::uint64_t compressed_bytes)
    : expanded_(expanded_bytes), compressed_(compressed_bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaHostAlloc(&p, words * sizeof(std::uint32_t), cudaHostAllocMapped | cudaHostAllocPortable));
    host_ = static_cast<std::uint32_t*>(p);
    void* d = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d, p, 0));
    device_ = static_cast<const std::uint32_t*>(d);
}

CsfTables::~CsfTables() {
    if (host_ != nullptr) { (void)cudaFreeHost(host_); }
}

CsfTables::CsfTables(CsfTables&& other) noexcept
    : host_(std::exchange(other.host_, nullptr)), device_(std::exchange(other.device_, nullptr)),
      expanded_(other.expanded_), compressed_(other.compressed_) {}

CsfTables& CsfTables::operator=(CsfTables&& other) noexcept {
    if (this != &other) {
        if (host_ != nullptr) { (void)cudaFreeHost(host_); }
        host_       = std::exchange(other.host_, nullptr);
        device_     = std::exchange(other.device_, nullptr);
        expanded_   = other.expanded_;
        compressed_ = other.compressed_;
    }
    return *this;
}

CsfTables compress_banks(std::span<ExpertBank> banks) {
    // The banks the host holds as NVFP4 records at a registered geometry.
    std::vector<ExpertBank*> chosen;
    std::size_t words = 0;
    std::uint64_t expanded = 0;
    for (ExpertBank& bank : banks) {
        const ExpertBankPlanes& p = bank.planes;
        if (p.records == nullptr || p.gate_up_format != QType::NVFP4_MUL || !moe::registered_shape(bank.shape()) ||
            p.record_stride < bank.shape().record_bytes() || p.experts == 0) {
            continue;
        }
        chosen.push_back(&bank);
        words += static_cast<std::size_t>(p.experts) * csf::Geometry::of(bank.shape()).table_words();
        expanded += static_cast<std::uint64_t>(p.experts) * bank.shape().record_bytes();
    }
    if (chosen.empty()) { return {}; }
    CsfTables tables(words, expanded, 0);
    // Each chosen bank's tables in turn; then every record of every chosen bank on every core.
    std::vector<std::size_t> first;
    std::size_t at = 0;
    for (ExpertBank* bank : chosen) {
        first.push_back(at);
        at += static_cast<std::size_t>(bank->planes.experts) * csf::Geometry::of(bank->shape()).table_words();
    }
    std::vector<std::uint64_t> base_record; // the first record index of each chosen bank
    std::uint64_t records = 0;
    for (ExpertBank* bank : chosen) {
        base_record.push_back(records);
        records += bank->planes.experts;
    }
    std::atomic<std::uint64_t> next{0}, compressed{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto work = [&] {
        std::uint64_t own = 0;
        try {
            for (std::uint64_t r = next++; r < records; r = next++) {
                const std::size_t b = static_cast<std::size_t>(
                    std::upper_bound(base_record.begin(), base_record.end(), r) - base_record.begin() - 1);
                ExpertBank& bank      = *chosen[b];
                const auto e          = static_cast<std::uint32_t>(r - base_record[b]);
                const auto table_words = static_cast<std::size_t>(csf::Geometry::of(bank.shape()).table_words());
                // The load's own materialized records: writable until the Model hands them out.
                auto* record = const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(bank.planes.records)) +
                               static_cast<std::uint64_t>(e) * bank.planes.record_stride;
                own += csf::compress_record(bank.shape(), record, record, tables.host() + first[b] + e * table_words);
            }
        } catch (...) {
            const std::lock_guard lock(failure_mutex);
            if (!failure) { failure = std::current_exception(); }
            next = records;
        }
        compressed += own;
    };
    const unsigned threads = std::max(1U, std::min(32U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < threads; ++i) { pool.emplace_back(work); }
    work();
    for (auto& thread : pool) { thread.join(); }
    if (failure) { std::rethrow_exception(failure); }
    tables.set_compressed_bytes(compressed.load());
    for (std::size_t b = 0; b < chosen.size(); ++b) {
        chosen[b]->csf = {tables.host() + first[b], tables.device() + first[b]};
    }
    return tables;
}

std::optional<moe::ExpertShape> tier_csf_shape(std::span<const ExpertBank* const> banks) {
    if (banks.empty()) { return std::nullopt; }
    const moe::ExpertShape shape = banks.front()->shape();
    const std::uint64_t stride   = banks.front()->planes.record_stride;
    for (const ExpertBank* bank : banks) {
        const ExpertBankPlanes& p = bank->planes;
        if (p.gate_up_format != QType::NVFP4_MUL || p.down_format != QType::NVFP4_MUL || !moe::registered_shape(bank->shape()) ||
            bank->shape().hidden != shape.hidden || bank->shape().intermediate != shape.intermediate ||
            p.record_stride != stride || stride < shape.record_bytes()) {
            return std::nullopt;
        }
    }
    return shape;
}

std::string csf_summary(std::span<const ExpertBank* const> banks) {
    std::uint64_t expanded = 0, compressed = 0, raw = 0, groups = 0;
    for (const ExpertBank* bank : banks) {
        if (bank == nullptr || !bank->csf) { continue; }
        const csf::Geometry g = csf::Geometry::of(bank->shape());
        const auto words      = static_cast<std::size_t>(g.table_words());
        for (std::uint32_t e = 0; e < bank->planes.experts; ++e) {
            const std::uint32_t* table = bank->csf.host + e * words;
            compressed += table[g.rgs()];
            expanded += bank->shape().record_bytes();
            for (int rg = 0; rg < g.rgs(); ++rg) {
                raw += table[rg + 1] - table[rg] == static_cast<std::uint32_t>(g.blocks(rg)) * moe::kUnitBytes ? 1U : 0U;
            }
            groups += static_cast<std::uint64_t>(g.rgs());
        }
    }
    if (expanded == 0) { return {}; }
    char text[192];
    std::snprintf(text, sizeof(text), "expert host records: CSF, %.4f of their bytes (%.2f GiB less to copy or read), "
                  "%.3f %% of row groups raw",
                  static_cast<double>(compressed) / static_cast<double>(expanded),
                  static_cast<double>(expanded - compressed) / static_cast<double>(1ULL << 30),
                  100.0 * static_cast<double>(raw) / static_cast<double>(groups));
    return text;
}

std::vector<moe::ExpertScales> a16_expert_scales(const ExpertBankPlanes& planes, const std::string& name) {
    if (planes.multipliers == nullptr) { throw std::invalid_argument(name + ": an NVFP4 bank needs its multipliers"); }
    std::vector<moe::ExpertScales> out(planes.experts);
    for (std::uint32_t e = 0; e < planes.experts; ++e) {
        const float* m = planes.multipliers + 3 * static_cast<std::size_t>(e);
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(m[i]) || m[i] <= 0) {
                throw std::invalid_argument(name + ": expert multipliers must be positive finite FP32");
            }
        }
        out[e] = {0.0F, 0.0F, 0.0F, m[0], m[1], m[2]};
    }
    return out;
}

namespace {

const WeightParent& a16_nvfp4_parent(const WeightView& view, const std::string& name) {
    if (view.parts.size() != 1 || !is_complete_weight(view)) {
        throw std::invalid_argument(name + ": an expert bank must be one complete parent");
    }
    const WeightParent& parent = *view.parts.front().parent;
    if (parent.geometry.layout != QuantLayout::ExpertRg16 || parent.geometry.format != QType::NVFP4_MUL) {
        throw std::invalid_argument(name + ": the bank must be nvfp4_mul in nvfp4_expert_rg16_v1");
    }
    return parent;
}

ExpertBank a16_nvfp4_bank(ExpertBankPlanes planes, std::uint32_t experts, std::uint32_t hidden,
                          std::uint32_t intermediate, const std::string& name) {
    ExpertBank bank;
    bank.planes = planes;
    if (bank.planes.experts != experts || bank.planes.hidden != hidden || bank.planes.intermediate != intermediate) {
        throw std::invalid_argument(name + ": expert bank geometry differs from the config");
    }
    if (!moe::registered_shape(bank.shape())) {
        throw std::invalid_argument(name + ": no expert routes are registered at this geometry");
    }
    bank.activation = moe::ExpertActivation::kA16;
    bank.scales     = a16_expert_scales(bank.planes, name);
    return bank;
}

} // namespace

ExpertBank a16_nvfp4_bank(const WeightView& view, std::uint32_t experts, std::uint32_t hidden,
                          std::uint32_t intermediate, const std::string& name) {
    return a16_nvfp4_bank(expert_bank_planes(a16_nvfp4_parent(view, name)), experts, hidden, intermediate, name);
}

ExpertBank a16_nvfp4_streamed_bank(const WeightView& view, const float* multipliers, std::uint32_t experts,
                                   std::uint32_t hidden, std::uint32_t intermediate, const std::string& name) {
    const WeightParent& parent = a16_nvfp4_parent(view, name);
    return a16_nvfp4_bank(expert_bank_layout(parent.geometry, multipliers), experts, hidden, intermediate, name);
}

ExpertBank a16_nvfp4_device_bank(const WeightView& view, const float* multipliers, std::uint32_t experts,
                                 std::uint32_t hidden, std::uint32_t intermediate, const std::string& name) {
    const WeightParent& parent = a16_nvfp4_parent(view, name);
    if (parent.device == nullptr) { throw std::invalid_argument(name + ": a device bank needs device records"); }
    ExpertBankPlanes planes = expert_bank_layout(parent.geometry, multipliers);
    planes.device_records   = parent.device;
    return a16_nvfp4_bank(planes, experts, hidden, intermediate, name);
}

} // namespace infernix::runtime::expert_offload
