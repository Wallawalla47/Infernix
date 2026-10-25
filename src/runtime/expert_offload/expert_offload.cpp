#include "runtime/expert_offload/expert_offload.h"

#include "core/cpu_topology.h"
#include "core/host_load.h"
#include "core/link_probe.h"
#include "core/machine_calibration.h"
#include "core/temporary_switches.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <system_error>
#include <thread>

// The mechanisms below are Qwen3.8-Flash-Next's (docs/maintainer/qwen3_8-flash-next-design.md
// §19.2-19.3, measured there on an RTX 5090 at PCIe 5.0 x8 with an i9-13900K); the constants keep
// their measured values and references.

namespace infernix::runtime::expert_offload {
namespace {

namespace moe = ops::offloaded_moe;

// Promotions per layer call while the frames fill: one per decode token measured fastest; a
// promotion moves as many PCIe bytes as serving the expert once zero-copy (§19.2).
// Once the frames are full, decode promotes every kDecodePromotionInterval-th token on the
// reference link (promotion_interval scales it to the measured one; fewer measured faster, §19.2).
constexpr std::uint64_t kDecodePromotionInterval = 4;
// The SSD tier's smallest RAM (§19.3.7 H_min: 1,024 slots).
constexpr std::uint32_t kMinTierSlots = 1024;
// Unpinned records a layer may stream through the bounce, per part (§19.3.7: 25.4 GiB of 63.3
// unpinned ran 2,350 tok/s with 160 and 2,673 with 256; 18.4 of 42.8 GiB 1,144 with 160, 1,082 with
// 256).
constexpr std::uint32_t kStreamBounceMin = 64, kStreamBounceMax = 256;
// A seeded count is capped, so an expert used heavily in an old session yields to new uses; the warm
// start fills half the frames (the rest stay free for the fill phase's landings, S4b).
constexpr std::uint32_t kSeedCountCap = 16;
constexpr double kSeedShare           = 0.5;
constexpr std::chrono::minutes kStateInterval{10};
// The LFRU count an expert gains when every token of a prompt routed to it (§19.3.27: mixed-topic
// decode, credit only, 256 -1.8 % time, replayed hits +0.59 points).
constexpr double kPromptSeedCredit = 256.0;
// The split plans at this share of the measured CPU rates (§19.3.12: fastest at 6 and 3 workers).
constexpr double kSplitRateShare = 0.3;

// The CPU team's rate probe (§19.3.12, §19.3.28): jobs per round, timed rounds per batch, batches
// per width, link copies loading DRAM meanwhile, and the quiet-machine rule.
constexpr int kRateJobs      = 16;
constexpr int kRateRounds    = 8;
constexpr int kRateRepeats   = 3;
constexpr int kLoadCopies    = 150;
constexpr int kNarrowColumns = moe::kMaxCpuColumns;
constexpr double kQuietShare = 0.25;
constexpr auto kRetryPause   = std::chrono::milliseconds(500);
constexpr auto kRetryBudget  = std::chrono::seconds(20);

class SplitPolicy final : public CpuSplitPolicy {
public:
    explicit SplitPolicy(const ExpertOffload& offload) : offload_(offload) {}

    // The thinnest non-resident experts of the layer (n <= 8 columns, in order of n then id), as many
    // as minimise the later of the CPU's predicted time and the link's for the rest.
    std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                         std::span<std::uint8_t> cpu) const override {
        const ExpertResidency& residency = offload_.residency();
        const std::uint32_t E      = static_cast<std::uint32_t>(columns.size());
        const std::int32_t* frames = residency.host_table() + static_cast<std::size_t>(layer) * E;
        const double link          = static_cast<double>(residency.frame_pitch(layer)) / offload_.link_bytes_per_second();
        const double per_expert    = 1.0 / (offload_.cpu_expert_rate() * kSplitRateShare);
        const double per_column    = 1.0 / (offload_.cpu_column_rate() * kSplitRateShare);
        std::array<std::uint32_t, kNarrowColumns + 1> width_count{};
        std::uint32_t absent = 0;
        for (std::uint32_t e = 0; e < E; ++e) {
            if (columns[e] <= 0 || frames[e] >= 0) { continue; }
            ++absent;
            if (columns[e] <= kNarrowColumns) { ++width_count[static_cast<std::size_t>(columns[e])]; }
        }
        double best_time = absent * link, cpu_time = 0.0;
        std::uint32_t best = 0, k = 0;
        for (int n = 1; n <= kNarrowColumns; ++n) {
            for (std::uint32_t i = 0; i < width_count[static_cast<std::size_t>(n)]; ++i) {
                if (k == static_cast<std::uint32_t>(moe::kMaxCpuJobs)) { break; }
                cpu_time += std::max(per_expert, n * per_column);
                ++k;
                const double time = std::max(cpu_time, (absent - k) * link);
                if (time < best_time) {
                    best_time = time;
                    best      = k;
                }
            }
        }
        std::uint32_t marked = 0;
        for (int n = 1; n <= kNarrowColumns && marked < best; ++n) {
            for (std::uint32_t e = 0; e < E && marked < best; ++e) {
                if (columns[e] == n && frames[e] < 0) {
                    cpu[e] = 1;
                    ++marked;
                }
            }
        }
        return marked;
    }

private:
    const ExpertOffload& offload_;
};

} // namespace

// ------------------------------------------------------------------------------------- planning

std::int32_t ExpertOffload::staging_slots(const OffloadSetup& setup, const OffloadOptions& options) noexcept {
    // INFERNIX_TMP_STAGING_SLOTS fixes the count (a temporary A/B of the rule below).
    if (const std::int32_t fixed = temporary_switches::offload().staging_slots; fixed > 0) { return fixed; }
    // Without the SSD tier a verification call's misses (up to top_k per column) stage in one pass,
    // so it keeps the fork route; with the tier the larger staging took RAM and frames from the cache.
    if (options.expert_ram_bytes != 0) { return kStagingSlots; }
    return std::clamp(static_cast<std::int32_t>(setup.layout.top_k) * setup.max_width, kStagingSlots, kMaxStagingSlots);
}

std::uint64_t ExpertOffload::record_stride(const OffloadSetup& setup) {
    std::uint64_t stride = 0;
    for (const ExpertBank* bank : setup.banks) { stride = std::max(stride, bank->planes.record_stride); }
    if (stride == 0 || stride % 16 != 0) { throw std::invalid_argument("expert offload: banks without a record stride"); }
    return stride;
}

// Whether plain decode's one-column calls take the divisor by context: a GGUF import's (or, with
// INFERNIX_TMP_PLAIN_DIVISOR_ALL, every bank's) without speculation or a fixed divisor.
static bool plain_by_context(const OffloadSetup& setup, const OffloadOptions& options) noexcept {
    return (options.plain_divisor_by_context || temporary_switches::offload().plain_divisor_all) && !setup.speculative &&
           options.cpu_pcie_divisor == 0 && options.cpu_expert_workers > 0 && options.cpu_expert_jobs > 0;
}

std::uint64_t ExpertOffload::fixed_bytes(const OffloadSetup& setup, const OffloadOptions& options) {
    return static_cast<std::uint64_t>(staging_slots(setup, options)) * record_stride(setup) +
           ExpertResidency::table_bytes(setup.layout, setup.columns) + (plain_by_context(setup, options) ? 16U : 0U);
}

// ------------------------------------------------------------------------------------- lifetime

ExpertOffload::Overlap::~Overlap() {
    for (auto event : events) {
        if (event != nullptr) { cudaEventDestroy(event); }
    }
    if (stream != nullptr) { cudaStreamDestroy(stream); }
}

ExpertOffload::ExpertOffload(OffloadSetup setup, OffloadOptions options, Diagnostic diagnostic)
    : setup_(std::move(setup)), options_(std::move(options)), diagnostic_(std::move(diagnostic)) {
    const ExpertLayout& layout = setup_.layout;
    if (layout.layers == 0 || layout.experts == 0 || layout.top_k == 0 || setup_.banks.size() != layout.layers ||
        setup_.device_scales.size() != layout.layers) {
        throw std::invalid_argument("expert offload: one bank and one scale array per offloaded layer");
    }
    vram_          = open_vram_budget_source(setup_.device, options_.vram_past_budget);
    record_stride_ = record_stride(setup_);
    std::vector<const std::uint8_t*> banks; // the promotions' copy sources: where the device reads the banks
    for (const ExpertBank* bank : setup_.banks) {
        // NVFP4 banks share one record size; a GGUF import's layers keep their own (their pools).
        if (bank->planes.gate_up_format == QType::NVFP4_MUL && bank->planes.record_stride != record_stride_) {
            throw std::invalid_argument("expert offload: the NVFP4 banks' record strides differ");
        }
        banks.push_back(reinterpret_cast<const std::uint8_t*>(bank->planes.device_records));
        host_strides_.push_back(bank->planes.record_stride);
    }
    staging_slots_ = staging_slots(setup_, options_);
    staging_       = DeviceBuffer(static_cast<std::size_t>(staging_slots_) * record_stride_);
    placement_     = cpu_placement(temporary_switches::offload(), options_.cpu_expert_workers);
    if (plain_by_context(setup_, options_)) {
        divisor_word_ = DeviceBuffer(16);
        divisor_now_  = plain_divisor(temporary_switches::offload(), 0);
        std::memset(divisor_host_.data(), 0, 16);
        *static_cast<std::int32_t*>(divisor_host_.data()) = divisor_now_;
        divisor_word_.copy_from_host(divisor_host_.data(), 16);
    }
    residency_     = std::make_unique<ExpertResidency>(layout, std::move(banks), record_stride_, setup_.columns,
                                                       setup_.device, host_strides_);
    std::memset(error_.data(), 0, 64);
    // TEMPORARY (INFERNIX_TMP_STREAM_PRIORITY, candidates §11 D5, as Flash-Next's staging and shared-expert
    // streams): the fork and staging-overlap stream at the device's greatest priority (graphs keep it per
    // node, core/decode_graph.cpp), so its copies and side work are scheduled ahead of the round's kernels.
    static const bool prioritized = temporary_switches::flag("INFERNIX_TMP_STREAM_PRIORITY");
    int least = 0, greatest = 0;
    if (prioritized) { CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest)); }
    CUDA_CHECK(cudaStreamCreateWithPriority(&overlap_.stream, cudaStreamNonBlocking, prioritized ? greatest : 0));
    for (auto& event : overlap_.events) { CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)); }
}

ExpertOffload::~ExpertOffload() {
    // The monitor's thread first (it calls back into this object), then the CPU service (its workers
    // read the tier's slots), the stream, the residency and the tier.
    monitor_.reset();
    cpu_service_.reset();
    stream_.reset();
    residency_.reset();
    tier_.reset();
}

void ExpertOffload::diagnostic(const std::string& text, DiagnosticLevel level) const noexcept {
    try {
        if (diagnostic_) {
            diagnostic_(text, level);
        } else {
            std::fprintf(stderr, "[engine] %s\n", text.c_str());
        }
    } catch (...) {}
}

void ExpertOffload::start(cudaStream_t compute, const std::function<void(std::uint64_t, std::uint64_t)>& lendable_frames) {
    // What the offload switches resolved to (their values are in the startup log's temporary switches line).
    {
        const OffloadSwitches& switches = temporary_switches::offload();
        std::string line =
            "expert offload (temporary switches resolved): staging slots " + std::to_string(staging_slots_);
        if (divisor_word_.p != nullptr) {
            line += "; plain decode one-column divisor " + std::to_string(plain_divisor(switches, 0)) + " below " +
                    std::to_string(kPlainDivisorContext) + " context tokens, " +
                    std::to_string(plain_divisor(switches, kPlainDivisorContext)) + " from there";
        }
        if (switches.prompt_credit >= 0.0 || switches.prompt_budget > 0) {
            line += "; prompt seed credit " + std::to_string(switches.credit_or(kPromptSeedCredit)) + ", " +
                    std::to_string(switches.prompt_budget) + " promotions per layer";
        }
        diagnostic(line);
    }
    // The prefill chunk the cache can lend a wide workspace for, from the frames it will start with,
    // before the SSD tier (whose tickets are sized by the chunk).
    if (options_.expert_cache && lendable_frames) {
        const std::uint64_t frames = size_expert_frames(vram_->query(), frame_demand(record_stride_)).frames;
        lendable_frames(frames - frames / 4, record_stride_);
    }
    if (setup_.store != nullptr) { create_tier(); }
    // After the tier (whose pinned slots probe the link when the banks stay in the artifact), before
    // the CPU service and the split take the measured rate.
    measure_link();
    if (options_.prefill_stream && options_.expert_cache) { create_stream(); }
    create_cpu_service();
    size_frames(compute, lendable_frames);
    warm_start(compute);
    // The SSD tier has no records in RAM before its fill: the CPU team's rates are measured on its RAM
    // copies now, so the decode divisor follows this record size and CPU (source() reads it).
    if (tier_ && cpu_service_) { measure_cpu_rate(); }
    if (!options_.route_trace.empty()) {
        trace_ = std::make_unique<RouteTrace>(
            options_.route_trace, RouteTraceSetup{.layers             = setup_.layout.layers,
                                                  .experts            = setup_.layout.experts,
                                                  .top_k              = setup_.layout.top_k,
                                                  .frames             = residency_->frames(),
                                                  .max_columns        = static_cast<std::uint32_t>(setup_.columns),
                                                  .lanes              = static_cast<std::uint32_t>(setup_.lanes),
                                                  .max_width          = static_cast<std::uint32_t>(setup_.max_width),
                                                  .mtp_draft_tokens   = setup_.mtp_draft_tokens,
                                                  .ngram_draft_tokens = setup_.ngram_draft_tokens,
                                                  .prefill_chunk      = static_cast<std::uint32_t>(setup_.chunk)});
    }
}

// ------------------------------------------------------------------------------------- SSD tier

void ExpertOffload::create_tier() {
    const ExpertStore& store = *setup_.store;
    for (std::uint32_t l = 0; l < store.layers(); ++l) {
        if (store.record_bytes(l) != host_strides_[l]) {
            throw std::logic_error("expert offload: the expert store's records differ from the banks' strides");
        }
    }
    // A call's CPU jobs hold their ring slots until its answer, a pass's fetched records until the pass
    // has copied them: the ring holds both with a pass to spare.
    const std::uint32_t cpu_cap =
        options_.cpu_expert_workers > 0 && options_.cpu_expert_jobs > 0
            ? std::min<std::uint32_t>(std::max(options_.cpu_expert_jobs, options_.cpu_assist_jobs),
                                      static_cast<std::uint32_t>(moe::kMaxCpuJobs))
            : 0U;
    HostExpertTier::Options tier;
    tier.ring = std::max<std::uint32_t>(128, cpu_cap + 2U * static_cast<std::uint32_t>(staging_slots_));
    const std::uint32_t lists = tier.ring + tier.prefetch + tier.demotion;
    const std::uint64_t keys  = static_cast<std::uint64_t>(store.layers()) * store.experts();
    if (options_.prefill_stream && options_.expert_cache) {
        // A wide chunk streams about a layer's unpinned share of its experts.
        const double share = std::min(1.0, static_cast<double>(options_.expert_unpinned_bytes / record_stride_) /
                                               static_cast<double>(keys));
        const auto per_part = std::clamp<std::uint32_t>(
            (static_cast<std::uint32_t>(std::ceil(share * store.experts())) + 15U) / 16U * 16U, kStreamBounceMin,
            kStreamBounceMax);
        tier.stream_bounce = ExpertStream::kBounceParts * per_part;
    }
    tier.slots = static_cast<std::uint32_t>(std::min<std::uint64_t>(options_.expert_ram_bytes / record_stride_, keys + lists));
    if (options_.prefill_stream && options_.expert_cache) { tier.stream = std::min(kStreamReadSlots, tier.slots / 8); }
    const std::uint32_t reserved =
        options_.expert_unpinned_bytes >= record_stride_ ? lists + tier.bounce + tier.stream_bounce : lists;
    tier.unpinned = static_cast<std::uint32_t>(std::min<std::uint64_t>(
        options_.expert_unpinned_bytes / record_stride_,
        keys - std::min<std::uint64_t>(keys, tier.slots - std::min(tier.slots, reserved))));
    tier.min_slots   = std::max<std::uint32_t>(kMinTierSlots, lists + kMinTierSlots / 2);
    tier.pin_reserve = options_.later_pinned_bytes;
    const std::uint32_t needed = tier.min_slots + (tier.unpinned > 0 ? tier.bounce + tier.stream_bounce : 0U);
    if (tier.slots < needed) {
        throw std::runtime_error("the SSD expert tier needs RAM for at least " + std::to_string(needed) +
                                 " pinned expert slots (" + std::to_string(tier.slots) +
                                 " fit in the RAM ledger's page-locked expert share; on Windows page-locked memory is "
                                 "limited to the GPU's shared memory, so lower --host-context-mib or close programs "
                                 "that use it)");
    }
    fetch_channel_ = std::make_unique<moe::FetchChannel>();
    tier.fetch     = fetch_channel_.get();
    // A layer walk's step calls ceil(layers / n) layers once per chunk of its n-chunk span.
    tier.layer_calls = store.layers() + setup_.walk_max_tokens / static_cast<std::uint32_t>(std::max(setup_.chunk, 1)) + 1U;
    const ExpertStore* records = &store;
    if (!options_.expert_stripe.empty()) {
        striped_store_.emplace(with_stripe(store, read_expert_stripe(options_.expert_stripe), setup_.artifact_id));
        records = &*striped_store_;
        tier.io.demand_depth *= 2;
        tier.io.total_depth = tier.io.demand_depth + tier.io.prefetch_depth;
        diagnostic("SSD tier: expert record tails from " + options_.expert_stripe.string(), DiagnosticLevel::Info);
    }
    tier_ = std::make_unique<HostExpertTier>(*records, std::move(tier));
    residency_->attach_tier(tier_.get());
    if (tier_->refused_slots() > 0) {
        diagnostic("SSD expert tier: Windows refused to page-lock " + std::to_string(tier_->refused_slots()) +
                       " of the planned expert slots (its limit is the GPU's shared memory); they are kept as "
                       "unpinned records the CPU serves",
                   DiagnosticLevel::Warning);
    }
    if (!tier_->unpinned_lock_failure().empty()) {
        diagnostic("SSD expert tier: the OS would not lock the " + std::to_string(tier_->unpinned_records()) +
                       " unpinned expert records in RAM (" + tier_->unpinned_lock_failure() +
                       "); they still work, but under memory pressure the OS may page them out and responses slow down",
                   DiagnosticLevel::Warning);
    }
}

const std::uint8_t* ExpertOffload::host_record(std::uint32_t layer, std::uint32_t expert) const {
    const ExpertBank& bank = *setup_.banks[layer];
    if (bank.planes.records != nullptr) {
        return reinterpret_cast<const std::uint8_t*>(bank.planes.records) +
               static_cast<std::uint64_t>(expert) * bank.planes.record_stride;
    }
    return tier_ ? tier_->record(tier_->key(layer, expert)) : nullptr;
}

// ------------------------------------------------------------------------------------- link and CPU

void ExpertOffload::measure_link() {
    double rate        = options_.link_bytes_per_second;
    const void* source = setup_.banks.front()->planes.records;
    if (source == nullptr && tier_) { source = tier_->slot_bytes(0); }
    std::string calibrated; // INFERNIX_TMP_MACHINE_CALIBRATION's note
    if (rate <= 0.0 && source != nullptr && staging_.p != nullptr && staging_.bytes > 0) {
        rate = measure_h2d_bytes_per_second(source, staging_.p, staging_.bytes, 3);
        // TEMPORARY (INFERNIX_TMP_MACHINE_CALIBRATION, candidates §7.3 D11, as Flash-Next's probe): a reading far
        // below this GPU's saved rate is taken again, and the saved rate kept when that is low too.
        if (const auto calibration = MachineCalibration::from_switch(); calibration && rate > 0.0) {
            char bus[32] = {};
            (void)cudaDeviceGetPCIBusId(bus, sizeof(bus), setup_.device);
            rate = calibration->resolve(
                std::string("link|") + bus + "|" + std::to_string(staging_.bytes), rate,
                [&] { return measure_h2d_bytes_per_second(source, staging_.p, staging_.bytes, 3); }, calibrated);
        }
    }
    if (rate <= 0.0) { return; } // nothing to probe with: keep the reference rate
    if (!calibrated.empty()) { diagnostic("PCIe link calibration: " + calibrated, DiagnosticLevel::Warning); }
    link_bytes_per_second_ = rate;
    char text[256];
    std::snprintf(text, sizeof(text),
                  "host-to-device link: %.1f GB/s%s; decode promotes every %llu tokens; prefill CPU split up to %d columns",
                  rate / 1e9, options_.link_bytes_per_second > 0.0 ? " (set)" : " measured",
                  static_cast<unsigned long long>(promotion_interval()), options_.prefill_cpu_split ? split_columns() : 0);
    diagnostic(text, DiagnosticLevel::Debug);
    std::snprintf(text, sizeof(text), "PCIe link to the GPU: %.1f GB/s%s", rate / 1e9,
                  options_.link_bytes_per_second > 0.0 ? " (fixed)" : " measured");
    diagnostic(text);
}

double ExpertOffload::cpu_expert_rate() const noexcept {
    return cpu_expert_rate_ > 0.0 ? cpu_expert_rate_ : options_.cpu_split_expert_rate;
}
double ExpertOffload::cpu_column_rate() const noexcept {
    return cpu_column_rate_ > 0.0 ? cpu_column_rate_ : options_.cpu_split_column_rate;
}

// The share of a decode call's misses the PCIe stage keeps, misses / d, from C t (C experts per second
// on the CPU, t seconds per record on the link): d = floor(1.65 + C t) (§19.3.12: the measured fastest
// divisors put the switch from 2 to 3 at C t = 1.35).
int ExpertOffload::pcie_divisor() const noexcept {
    if (options_.cpu_pcie_divisor > 0) { return options_.cpu_pcie_divisor; }
    const double record_seconds = static_cast<double>(setup_.banks.front()->planes.record_stride) / link_bytes_per_second_;
    return std::max(1, static_cast<int>(std::floor(1.65 + cpu_expert_rate() * record_seconds)));
}

// INFERNIX_TMP_LINK_SHARE (Strata's L / (L + 1.85 R)): R from the CPU-rate probe and the first bank's
// record, as pcie_divisor's C t.
int ExpertOffload::pcie_share() const noexcept {
    if (options_.cpu_pcie_divisor > 0) { return 0; } // a fixed divisor wins
    return link_share_256(temporary_switches::offload(), link_bytes_per_second_, cpu_expert_rate(),
                          static_cast<double>(setup_.banks.front()->planes.record_stride));
}

int ExpertOffload::one_column_divisor() const noexcept {
    return plain_by_context(setup_, options_) ? plain_divisor(temporary_switches::offload(), 0) : 0;
}

void ExpertOffload::set_plain_position(std::int32_t position, cudaStream_t compute) {
    if (divisor_word_.p == nullptr) { return; }
    const std::int32_t d = plain_divisor(temporary_switches::offload(), position);
    if (d == divisor_now_) { return; }
    // The last upload has run (rounds synchronize), so the pinned source may change.
    *static_cast<std::int32_t*>(divisor_host_.data()) = d;
    upload_pinned(divisor_word_.p, divisor_host_.data(), 16, compute);
    divisor_now_ = d;
}

std::int32_t ExpertOffload::split_columns() const noexcept {
    const double scaled = options_.cpu_split_columns * kReferenceLinkBytesPerSecond / link_bytes_per_second_;
    return static_cast<std::int32_t>(std::min<double>(scaled, moe::kMaxCpuCallColumns));
}

std::uint64_t ExpertOffload::promotion_interval() const noexcept {
    const double scaled = static_cast<double>(kDecodePromotionInterval) * kReferenceLinkBytesPerSecond / link_bytes_per_second_;
    return static_cast<std::uint64_t>(std::clamp(std::lround(scaled), 1L, 8L));
}

// The team's rates for the configured workers, measured while the link copies records (the staging's
// DRAM reads compete with the CPU's): experts per second on one-column jobs and expert columns per
// second on kNarrowColumns-column jobs, the best of three batches each, retried while other processes
// load the CPU (§19.3.28).
void ExpertOffload::measure_cpu_rate() {
    const std::uint32_t L = setup_.layout.layers, E = setup_.layout.experts;
    std::vector<std::vector<std::uint32_t>> held(L); // per layer, the experts with a host copy
    bool measured = L > 0;
    for (std::uint32_t l = 0; l < L && measured; ++l) {
        for (std::uint32_t e = 0; e < E; ++e) {
            if (host_record(l, e) != nullptr) { held[l].push_back(e); }
        }
        measured = !held[l].empty();
    }
    int attempts = 0;
    bool quiet   = true;
    double first_share = 0.0, last_share = 0.0;
    if (measured) {
        const ExpertBank& front = *setup_.banks.front();
        const moe::ExpertShape shape = front.shape();
        // The service's team and placement (its worker 0, the service thread, is this thread here).
        moe::CpuExpertTeam team({.workers    = static_cast<int>(placement_.workers),
                                 .max_jobs   = kRateJobs,
                                 .activation = front.activation,
                                 .ggml       = ggml_format(front).valid(),
                                 .shape      = shape,
                                 .cpus       = placement_.cpus,
                                 .shared_x   = temporary_switches::flag("INFERNIX_TMP_CPU_SHARED_X")});
        const std::uint64_t caller_cpus = set_current_thread_cpus(placement_.cpus.empty() ? 0 : placement_.cpus.front());
        const auto H = static_cast<std::size_t>(shape.hidden);
        std::vector<std::uint16_t> x(H * kNarrowColumns);
        std::vector<std::uint16_t> y(H * kRateJobs * kNarrowColumns);
        for (std::size_t i = 0; i < x.size(); ++i) { x[i] = static_cast<std::uint16_t>(0x3C00U + (i * 7919U) % 640U); }
        std::vector<moe::CpuExpertJob> jobs(kRateJobs);
        const auto fill = [&](int round, int columns) {
            for (int k = 0; k < kRateJobs; ++k) {
                const std::uint32_t layer = static_cast<std::uint32_t>(round * kRateJobs + k) % L;
                const ExpertBank& bank    = *setup_.banks[layer];
                const auto& experts       = held[layer];
                const std::uint32_t expert =
                    experts[(static_cast<std::uint32_t>(round) * 131U + static_cast<std::uint32_t>(k) * 37U) % experts.size()];
                auto& job  = jobs[static_cast<std::size_t>(k)];
                job.record = host_record(layer, expert);
                job.ggml   = ggml_format(bank);
                job.scales = job.ggml.valid() ? moe::ExpertScales{} : bank.scales[expert];
                job.ncols  = columns;
                for (int c = 0; c < columns; ++c) {
                    job.x[c] = x.data() + static_cast<std::size_t>(c) * H;
                    job.y[c] = y.data() + (static_cast<std::size_t>(k) * kNarrowColumns + c) * H;
                }
            }
        };
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const std::uint64_t source_stride = front.planes.record_stride; // at most the staging pitch
        const auto rate = [&](int columns, int first_round) {
            fill(first_round + kRateRepeats * kRateRounds, columns); // warm-up round
            team.run(jobs);
            double best = 0.0;
            for (int repeat = 0; repeat < kRateRepeats; ++repeat) {
                for (int i = 0; i < kLoadCopies; ++i) {
                    const auto& first = held.front();
                    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(staging_.p) + (i % staging_slots_) * record_stride_,
                                               host_record(0, first[static_cast<std::size_t>(i) % first.size()]),
                                               source_stride, cudaMemcpyHostToDevice, stream));
                }
                const auto start = std::chrono::steady_clock::now();
                for (int r = 0; r < kRateRounds; ++r) {
                    fill(first_round + repeat * kRateRounds + r, columns);
                    team.run(jobs);
                }
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                best = std::max(best, kRateRounds * kRateJobs / seconds);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            return best;
        };
        const auto began = std::chrono::steady_clock::now();
        double expert_rate = 0.0, column_rate = 0.0;
        for (;;) {
            const HostCpuTimes before = host_cpu_times();
            const double experts      = rate(1, 0);
            const double columns      = rate(kNarrowColumns, (kRateRepeats + 1) * kRateRounds) * kNarrowColumns;
            const double others       = other_process_cpu_share(before, host_cpu_times());
            if (attempts == 0) { first_share = others; }
            last_share = others;
            ++attempts;
            expert_rate = std::max(expert_rate, experts);
            column_rate = std::max(column_rate, columns);
            quiet       = others < kQuietShare;
            if (quiet || std::chrono::steady_clock::now() - began >= kRetryBudget) { break; }
            std::this_thread::sleep_for(kRetryPause);
        }
        // TEMPORARY (INFERNIX_TMP_MACHINE_CALIBRATION, as Flash-Next's): a rate far below this configuration's
        // saved one is measured once more, and the saved rate kept when that is low too.
        if (const auto calibration = MachineCalibration::from_switch()) {
            const std::string key = "cpu|" + std::to_string(placement_.workers) + "|" + placement_.text + "|" +
                                    std::to_string(static_cast<int>(front.activation)) + "|" +
                                    std::to_string(ggml_format(front).valid() ? 1 : 0) + "|" +
                                    std::to_string(front.planes.record_stride) + (tier_ ? "|tier" : "");
            std::string note;
            expert_rate = calibration->resolve(key + "|experts", expert_rate, [&] { return rate(1, 0); }, note);
            if (!note.empty()) { calibration_note_ = "experts/s " + note; }
            note.clear();
            column_rate = calibration->resolve(
                key + "|columns", column_rate,
                [&] { return rate(kNarrowColumns, (kRateRepeats + 1) * kRateRounds) * kNarrowColumns; }, note);
            if (!note.empty()) { calibration_note_ += (calibration_note_.empty() ? "" : "; ") + ("columns/s " + note); }
        }
        cpu_expert_rate_ = expert_rate;
        cpu_column_rate_ = column_rate;
        (void)set_current_thread_cpus(caller_cpus);
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    std::string how = measured ? (tier_ ? "measured on the SSD tier's RAM copies" : "measured")
                               : "reference: no expert records in RAM yet";
    if (!calibration_note_.empty()) { how += "; calibration: " + calibration_note_; }
    char share[160];
    if (measured && attempts > 1) {
        std::snprintf(share, sizeof(share), " after %d attempts: other processes used %.0f %% of the CPU at first, %.0f %% last",
                      attempts, 100.0 * first_share, 100.0 * last_share);
        how += share;
    } else if (measured) {
        std::snprintf(share, sizeof(share), "; other processes used %.0f %% of the CPU", 100.0 * last_share);
        how += share;
    }
    char kept[160];
    if (const int share = pcie_share(); share > 0) {
        std::snprintf(kept, sizeof(kept), "%d/256 (INFERNIX_TMP_LINK_SHARE; the divisor rule's 1/%d)", share, pcie_divisor());
    } else {
        std::snprintf(kept, sizeof(kept), "1/%d", pcie_divisor());
    }
    char text[640];
    std::snprintf(text, sizeof(text),
                  "CPU expert team: %u workers (%s), %.0f experts/s and %.0f expert columns/s (%s); decode misses kept "
                  "on the link: %s",
                  placement_.workers, moe::cpu_isa_name(moe::best_cpu_isa()), cpu_expert_rate_, cpu_column_rate_,
                  how.c_str(), kept);
    diagnostic(text, DiagnosticLevel::Info);
    if (moe::best_cpu_isa() == moe::CpuIsa::kScalar) {
        diagnostic("the CPU has no AVX2: the CPU expert team computes experts with scalar code, so the GPU serves "
                   "most misses over PCIe",
                   DiagnosticLevel::Warning);
    }
    if (measured && temporary_switches::flag("INFERNIX_TMP_CPU_PROBE_DETAIL")) { cpu_probe_detail(); }
    if (measured && !quiet) {
        diagnostic("the CPU expert team's rates were measured while other processes kept the CPU busy (" +
                       std::to_string(static_cast<int>(100.0 * last_share + 0.5)) +
                       " %); the decode miss split follows them for this session and may leave decode slower; start "
                       "the engine when the machine is quieter",
                   DiagnosticLevel::Warning);
    }
}

// The SSD tier's probe read 4,887-7,062 one-column experts/s against 9,569-11,619 on the registered banks,
// while its 8-column rate fell only 6-14 % (Ornith-122B, 2026-10-22): a latency effect on single-column jobs.
// This times the same 32 records in three memories, interleaved, best of three each, so the cause (the
// memory's allocation and the tier's slot layout, or the process state after the tier's fill) shows.
void ExpertOffload::cpu_probe_detail() {
    constexpr std::size_t kRecords = 32;
    const std::uint32_t L = setup_.layout.layers, E = setup_.layout.experts;
    struct Pick {
        std::uint32_t layer = 0, expert = 0;
        const std::uint8_t* source = nullptr;
        std::uint64_t bytes = 0, offset = 0;
    };
    std::vector<Pick> picks;
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; picks.size() < kRecords && i < L * E; ++i) {
        const std::uint32_t layer  = i % L;
        const std::uint32_t expert = ((i / L) * 37U + layer * 11U) % E;
        if (const std::uint8_t* record = host_record(layer, expert)) {
            picks.push_back({layer, expert, record, host_strides_[layer], total});
            total += host_strides_[layer];
        }
    }
    if (picks.size() < kRecords) {
        diagnostic("CPU probe detail: fewer than 32 expert records in RAM", DiagnosticLevel::Info);
        return;
    }
    RegisteredHostBuffer registered(total);
    registered.pin();
    void* allocated = nullptr;
    if (cudaHostAlloc(&allocated, total, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
        (void)cudaGetLastError();
        diagnostic("CPU probe detail: cudaHostAlloc of the copies failed", DiagnosticLevel::Info);
        return;
    }
    for (const Pick& p : picks) {
        std::memcpy(registered.data() + p.offset, p.source, p.bytes);
        std::memcpy(static_cast<std::byte*>(allocated) + p.offset, p.source, p.bytes);
    }
    const ExpertBank& front      = *setup_.banks.front();
    const moe::ExpertShape shape = front.shape();
    moe::CpuExpertTeam team({.workers    = static_cast<int>(placement_.workers),
                             .max_jobs   = kRateJobs,
                             .activation = front.activation,
                             .ggml       = ggml_format(front).valid(),
                             .shape      = shape,
                             .cpus       = placement_.cpus});
    const std::uint64_t caller_cpus = set_current_thread_cpus(placement_.cpus.empty() ? 0 : placement_.cpus.front());
    const auto H = static_cast<std::size_t>(shape.hidden);
    std::vector<std::uint16_t> x(H * kNarrowColumns);
    std::vector<std::uint16_t> y(H * kRateJobs * kNarrowColumns);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = static_cast<std::uint16_t>(0x3C00U + (i * 7919U) % 640U); }
    std::vector<moe::CpuExpertJob> jobs(kRateJobs);
    // memory 0: where the records live; 1: registered copies; 2: cudaHostAlloc copies.
    const auto record_in = [&](int memory, const Pick& p) -> const std::uint8_t* {
        if (memory == 1) { return reinterpret_cast<const std::uint8_t*>(registered.data() + p.offset); }
        if (memory == 2) { return static_cast<const std::uint8_t*>(allocated) + p.offset; }
        return p.source;
    };
    const auto timed = [&](int memory, int columns) {
        const auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < kRateRounds; ++r) {
            for (int k = 0; k < kRateJobs; ++k) {
                const Pick& p    = picks[static_cast<std::size_t>(r * kRateJobs + k) % picks.size()];
                const ExpertBank& bank = *setup_.banks[p.layer];
                auto& job        = jobs[static_cast<std::size_t>(k)];
                job.record       = record_in(memory, p);
                job.ggml         = ggml_format(bank);
                job.scales       = job.ggml.valid() ? moe::ExpertScales{} : bank.scales[p.expert];
                job.ncols        = columns;
                for (int c = 0; c < columns; ++c) {
                    job.x[c] = x.data() + static_cast<std::size_t>(c) * H;
                    job.y[c] = y.data() + (static_cast<std::size_t>(k) * kNarrowColumns + c) * H;
                }
            }
            team.run(jobs);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return kRateRounds * kRateJobs * columns / seconds;
    };
    std::array<double, 3> one{}, narrow{};
    (void)timed(0, 1); // warm-up
    for (int repeat = 0; repeat < kRateRepeats; ++repeat) {
        for (int memory = 0; memory < 3; ++memory) {
            one[static_cast<std::size_t>(memory)]    = std::max(one[static_cast<std::size_t>(memory)], timed(memory, 1));
            narrow[static_cast<std::size_t>(memory)] =
                std::max(narrow[static_cast<std::size_t>(memory)], timed(memory, kNarrowColumns));
        }
    }
    (void)set_current_thread_cpus(caller_cpus);
    (void)cudaFreeHost(allocated);
    char text[512];
    std::snprintf(text, sizeof(text),
                  "CPU probe detail (32 records, %s): experts/s %.0f in place, %.0f registered copies, %.0f "
                  "cudaHostAlloc copies; expert columns/s at %d columns %.0f, %.0f, %.0f",
                  tier_ ? "in place = the SSD tier's slots" : "in place = the registered banks", one[0], one[1],
                  one[2], kNarrowColumns, narrow[0], narrow[1], narrow[2]);
    diagnostic(text, DiagnosticLevel::Info);
}

// ------------------------------------------------------------------------------------- stream and CPU

void ExpertOffload::create_stream() {
    ExpertStream::RecordOf record_of;
    if (tier_) {
        record_of = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
            return tier->record(tier->key(layer, expert));
        };
    } else {
        std::vector<const std::uint8_t*> banks; // device addresses: the stream batches its copies
        for (const ExpertBank* bank : setup_.banks) {
            banks.push_back(reinterpret_cast<const std::uint8_t*>(bank->planes.device_records));
        }
        record_of = [banks = std::move(banks), strides = host_strides_](std::uint32_t layer, std::uint32_t expert) {
            return banks[layer] + static_cast<std::size_t>(expert) * strides[layer];
        };
    }
    ExpertStream::Hold hold;
    ExpertStream::SsdReads ssd;
    if (tier_) {
        hold     = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) { tier->stream_pin(tier->key(layer, expert)); };
        ssd.read = [tier = tier_.get()](std::uint32_t layer, std::uint32_t expert) {
            return tier->stream_read(tier->key(layer, expert));
        };
        ssd.wait    = [tier = tier_.get()](std::int32_t slot) { return tier->stream_wait(slot); };
        ssd.release = [tier = tier_.get()](std::int32_t slot) { tier->stream_release(slot); };
        ssd.slots   = [tier = tier_.get()] { return tier->stream_slots(); };
    }
    stream_ = std::make_unique<ExpertStream>(setup_.layout.layers, setup_.layout.experts, record_stride_, std::move(record_of),
                                             !tier_, std::move(hold), host_strides_, std::move(ssd));
    if (tier_ && tier_->stream_bounce_slots() > 0) {
        // Unpinned records stream through the tier's bounce slots, still two layers ahead.
        HostExpertTier* tier = tier_.get();
        stream_->set_staging(ExpertStream::Staging{
            .unpinned = [tier](std::uint32_t layer, std::uint32_t expert) { return tier->unpinned_record(tier->key(layer, expert)); },
            .slot     = [tier](std::uint32_t i) { return tier->stream_bounce_slot(i); },
            .copy     = [tier](std::span<const std::pair<std::uint8_t*, const std::uint8_t*>> copies) { tier->copy_records(copies); },
            .per_part = tier->stream_bounce_slots() / ExpertStream::kBounceParts});
    }
}

// CPU-served misses for decode and verification calls and every call of at most kMaxCpuColumns
// columns (short prompts, chunk remainders, forced tokens), whose experts are all thin and so give the
// same bits on either device (design §16.2); wider prefill chunks stay on the GPU, except the
// assisted calls up to kAssistMaxColumns and the streamed chunks' split.
void ExpertOffload::create_cpu_service() {
    const std::uint32_t workers = placement_.workers, jobs = options_.cpu_expert_jobs;
    if (workers == 0 || jobs == 0) { return; }
    if (!placement_.text.empty()) { diagnostic(placement_.text); }
    if (!tier_) { measure_cpu_rate(); } // the tier's rates follow its fill (start)
    const int decode_columns = std::max(moe::kMaxCpuColumns, setup_.lanes * setup_.max_width);
    const bool assist        = options_.cpu_assist_jobs > 0 && decode_columns < kAssistMaxColumns;
    std::vector<moe::CpuMissService::Layer> layers;
    for (const ExpertBank* bank : setup_.banks) {
        layers.push_back({.records       = reinterpret_cast<const std::uint8_t*>(bank->planes.records),
                          .record_stride = bank->planes.record_stride,
                          .scales        = bank->scales.data(),
                          .ggml          = ggml_format(*bank)});
    }
    const ExpertBank& front = *setup_.banks.front();
    cpu_service_ = std::make_unique<moe::CpuMissService>(
        std::move(layers),
        moe::CpuMissService::Options{
            .workers            = static_cast<int>(workers),
            .max_jobs           = static_cast<int>(std::min<std::uint32_t>(jobs, moe::kMaxCpuJobs)),
            .max_columns        = assist ? kAssistMaxColumns : decode_columns,
            .pcie_divisor       = pcie_divisor(),
            .one_column_divisor = one_column_divisor(),
            .pcie_share         = pcie_share(),
            .wide_from          = assist ? decode_columns + 1 : 0,
            .wide_jobs          = assist ? static_cast<int>(std::min<std::uint32_t>(options_.cpu_assist_jobs, moe::kMaxCpuJobs)) : 0,
            .cpus               = placement_.cpus,
            .activation         = front.activation,
            .shape              = front.shape(),
            .records            = tier_.get(), // tier mode: SSD-only jobs read through the tier
            // Reported when it happens; the next round's check_error then stops the Engine with it.
            .on_failure = [this](const std::string& why) {
                diagnostic("the CPU expert service stopped serving misses: " + why, DiagnosticLevel::Error);
            }});
    if (assist && stream_ && options_.prefill_cpu_split) { split_ = std::make_unique<SplitPolicy>(*this); }
}

ops::MoeExpertSource ExpertOffload::source(std::uint32_t layer) const {
    const ExpertBank& bank = *setup_.banks.at(layer);
    ops::MoeExpertSource s;
    s.frame_base    = residency_->frame_base(layer);
    s.frames        = residency_->table(layer);
    s.record_stride = bank.planes.record_stride;
    s.frame_stride  = residency_->frame_pitch(layer);
    if (tier_) {
        s.host_table = residency_->record_table(layer);
        s.fetch      = fetch_channel_->channel(static_cast<int>(layer));
    } else {
        s.host_records = reinterpret_cast<const std::uint8_t*>(bank.planes.device_records);
    }
    s.scales        = setup_.device_scales[layer];
    s.activation    = bank.activation;
    s.shape         = bank.shape();
    s.staging_base  = static_cast<std::uint8_t*>(staging_.p);
    s.staging_slots = staging_slots_;
    s.error         = static_cast<std::uint32_t*>(error_.data());
    if (bank.planes.gate_up_format != QType::NVFP4_MUL) {
        // A GGUF import's or block-FP8 bank: its own record size in frames of its pool's pitch; stages
        // copy the record rounded up to 16 bytes (the stride's zero padding).
        s.ggml = ops::MoeGgmlBank{.gate_up_format = bank.planes.gate_up_format,
                                  .down_format    = bank.planes.down_format,
                                  .gate_row_bytes = static_cast<std::uint32_t>(bank.planes.gate_row_bytes),
                                  .down_row_bytes = static_cast<std::uint32_t>(bank.planes.down_row_bytes),
                                  .record_bytes   = (bank.planes.record_bytes + 15) / 16 * 16};
    }
    if (cpu_service_) {
        s.cpu              = cpu_service_->channel(static_cast<int>(layer));
        s.cpu.pcie_divisor = pcie_divisor();
        s.cpu.pcie_share   = pcie_share();
        if (divisor_word_.p != nullptr) {
            s.cpu.one_column_divisor_device = static_cast<const std::int32_t*>(divisor_word_.p);
        }
    }
    return s;
}

// ------------------------------------------------------------------------------------- frames

VramDemand ExpertOffload::frame_demand(std::uint64_t frame_bytes) const {
    VramDemand demand;
    demand.headroom    = options_.vram_headroom;
    demand.elastic     = residency_->elastic() && options_.vram_monitor;
    demand.graphs      = setup_.graph_bound;
    demand.eager_graphs = setup_.eager_graphs;
    demand.frame_bytes = frame_bytes;
    demand.max_frames  = residency_->frame_limit();
    return demand;
}

// The frames take what the fixed allocations, the graph executables' reserve and the display headroom
// leave (§19.3.7); a chunk the driver places in system memory is given back and the cache stops there.
void ExpertOffload::size_frames(cudaStream_t compute, const std::function<void(std::uint64_t, std::uint64_t)>& lendable_frames) {
    if (!options_.expert_cache) { return; }
    const VramDemand demand = frame_demand(residency_->frame_bytes());
    sizing_                 = size_expert_frames(vram_->query(), demand);
    // Each pool maps whole chunks: one chunk of slack per pool beyond the first.
    const auto slack = static_cast<std::uint32_t>((residency_->pools() - 1) * ExpertResidency::kChunkBytes / residency_->frame_bytes());
    sizing_.frames   = sizing_.frames > slack ? sizing_.frames - slack : 0U;
    const auto grown = residency_->resize(sizing_.frames, compute, *vram_);
    if (grown.spilled != 0) {
        diagnostic(std::to_string(grown.spilled >> 20) + " MiB of the expert frames were placed in system memory and given "
                       "back; the expert cache starts with " + std::to_string(grown.frames) + " of " +
                       std::to_string(sizing_.frames) + " frames",
                   DiagnosticLevel::Warning);
    } else if (grown.refused) {
        diagnostic("the GPU refused expert-cache memory; the cache starts with " + std::to_string(grown.frames) + " of " +
                       std::to_string(sizing_.frames) + " frames",
                   DiagnosticLevel::Warning);
    }
    sizing_.frames = grown.frames;
    {
        const std::uint64_t experts = static_cast<std::uint64_t>(setup_.layout.layers) * setup_.layout.experts;
        char line[192];
        std::snprintf(line, sizeof(line), "expert cache: %u VRAM frames (%.1f GiB) for %llu routed experts (%.0f %%)",
                      sizing_.frames,
                      static_cast<double>(sizing_.frames) * static_cast<double>(residency_->frame_bytes()) / kLedgerGiB,
                      static_cast<unsigned long long>(experts),
                      experts != 0 ? 100.0 * static_cast<double>(sizing_.frames) / static_cast<double>(experts) : 0.0);
        diagnostic(line);
    }
    // Decode serves every miss through the CPU or the link: a small cache is legal but slow (as
    // check_expert_vram warns for Qwen3.8-Flash-Next before its load).
    if (sizing_.frame_bytes < 0) {
        diagnostic("the expert cache has no VRAM: the Program's fixed allocations (" +
                       std::to_string(sizing_.demand.fixed >> 20) + " MiB beyond the free memory's headroom of " +
                       std::to_string(sizing_.headroom >> 20) + " MiB and reserve of " +
                       std::to_string(sizing_.reserve >> 20) +
                       " MiB) take it all; every expert is read through the CPU or the link and decode is very "
                       "slow. Lower --max-context, --max-concurrency or --prefill-chunk, use a smaller --kv-dtype, "
                       "or free VRAM",
                   DiagnosticLevel::Warning);
    } else if (sizing_.frames < kFewFrames) {
        diagnostic("decode will be slow: only " + std::to_string(sizing_.frames) + " expert frames fit in VRAM (" +
                       std::to_string(kFewFrames) +
                       " or more recommended); lower --max-context or --max-concurrency, use a smaller --kv-dtype, or "
                       "free VRAM",
                   DiagnosticLevel::Warning);
    }
    if (options_.vram_monitor) {
        const VramSnapshot now = vram_->query();
        if (residency_->elastic()) {
            control_.emplace(demand, ExpertResidency::kChunkBytes, now.local_usage, residency_->pool_bytes(),
                             options_.vram_grow_delay_seconds);
        }
        frames_now_     = residency_->frames();
        pool_now_       = residency_->pool_bytes();
        startup_frames_ = frames_now_;
        vram_epoch_     = Clock::now();
        monitor_ = std::make_unique<VramMonitor>(setup_.device, *vram_, [this](const VramSnapshot& s) { on_vram_snapshot(s); });
    }
    if (grown.frames > 0 && lendable_frames) {
        lendable_frames(residency_->lend_capacity_bytes() / residency_->lend_pitch(), residency_->lend_pitch());
    }
}

double ExpertOffload::vram_seconds() const { return std::chrono::duration<double>(Clock::now() - vram_epoch_).count(); }

ExpertCacheMemorySummary ExpertOffload::memory_summary() const noexcept {
    ExpertCacheMemorySummary out;
    out.frames         = residency_->frames();
    out.startup_frames = startup_frames_;
    out.routed_experts = static_cast<std::uint64_t>(setup_.layout.layers) * setup_.layout.experts;
    out.frame_bytes    = residency_->frame_bytes();
    out.staging_bytes  = staging_.bytes;
    out.ssd_tier       = tier_ != nullptr;
    if (tier_) {
        out.host_ram_bytes = options_.expert_ram_bytes + options_.expert_unpinned_bytes;
    } else {
        for (const std::uint64_t stride : host_strides_) { out.host_ram_bytes += stride * setup_.layout.experts; }
    }
    return out;
}

// The monitor thread: wakes an idle engine when the cache should resize; without an elastic pool,
// warns once per pressure episode that nothing absorbs it.
void ExpertOffload::on_vram_snapshot(const VramSnapshot& snapshot) {
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    if (maintenance_due_ && waker_ && maintenance_due_()) { waker_(); }
    if (control_) {
        const auto d = control_->decide(snapshot, pool_now_, frames_now_, vram_seconds(), true);
        if ((d.shrink || d.grow) && waker_) { waker_(); }
        return;
    }
    const std::uint64_t headroom = display_headroom(snapshot.display, false, options_.vram_headroom);
    const bool pressure =
        snapshot.device_free < headroom / 2 || (snapshot.has_budget && snapshot.local_usage > snapshot.local_budget);
    if (pressure && !pressure_warned_) {
        const std::uint64_t short_by = headroom > snapshot.device_free ? headroom - snapshot.device_free : 0;
        diagnostic("free VRAM is " + std::to_string(snapshot.device_free >> 20) + " MiB, below the " +
                       std::to_string(headroom >> 20) + " MiB headroom by " + std::to_string(short_by >> 20) +
                       " MiB; the expert cache cannot shrink on this system, so Infernix memory may move to system "
                       "memory and slow down. Raise --vram-headroom-mib to " +
                       std::to_string((headroom + short_by + (256ULL << 20)) >> 20),
                   DiagnosticLevel::Warning);
    }
    pressure_warned_ = pressure;
}

void ExpertOffload::set_maintenance_waker(std::function<void()> waker) {
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    waker_ = std::move(waker);
}

void ExpertOffload::set_maintenance_due(std::function<bool()> due) {
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    maintenance_due_ = std::move(due);
}

void ExpertOffload::apply_vram_target(cudaStream_t compute, bool idle) {
    if (!control_) { return; }
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    const auto d = control_->decide(monitor_->latest(), pool_now_, frames_now_, vram_seconds(), idle);
    if (d.frames == frames_now_) { return; }
    const std::uint32_t before = frames_now_;
    const auto resized         = residency_->resize(d.frames, compute, *vram_);
    frames_now_                = resized.frames;
    pool_now_                  = residency_->pool_bytes();
    const VramSnapshot after   = monitor_->refresh();
    diagnostic(std::string(d.shrink ? "free VRAM fell to " : "free VRAM rose to ") + std::to_string(after.device_free >> 20) +
                   " MiB: expert cache " + std::to_string(before) + " -> " + std::to_string(frames_now_) + " frames" +
                   (resized.spilled != 0 ? " (a chunk placed in system memory was given back)" : ""),
               DiagnosticLevel::Info);
    if (d.shrink && frames_now_ * 4 < startup_frames_ && !low_frames_warned_) {
        diagnostic("another program holds VRAM: the expert cache is down to " + std::to_string(frames_now_) + " of " +
                       std::to_string(startup_frames_) + " frames, so decode is slower until that memory is released",
                   DiagnosticLevel::Warning);
        low_frames_warned_ = true;
    }
    if (frames_now_ * 4 >= startup_frames_) { low_frames_warned_ = false; }
}

std::uint32_t ExpertOffload::spare_frames() const {
    if (!control_) { return 0; }
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    const std::uint32_t frames = residency_->frames();
    return std::max<std::uint32_t>(control_->target(vram_->query(), pool_now_), frames) - frames;
}

bool ExpertOffload::take_frames(std::uint64_t bytes, cudaStream_t compute) {
    const std::uint64_t stride = residency_->frame_bytes();
    const std::uint32_t frames = residency_->frames();
    const std::uint32_t spare  = spare_frames();
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    // Memory the sizing would give the frames anyway comes first; the rest is taken from the frames,
    // one chunk per pool over so the pools unmap at least that much.
    const std::uint64_t spare_bytes = static_cast<std::uint64_t>(spare) * stride;
    std::uint32_t give              = 0;
    if (bytes > spare_bytes) {
        give = static_cast<std::uint32_t>((bytes - spare_bytes + residency_->pools() * ExpertResidency::kChunkBytes + stride - 1) /
                                          stride);
    }
    if (give > frames) { return false; }
    if (give != 0) {
        const auto resized = residency_->resize(frames - give, compute, *vram_);
        if (resized.frames > frames - give) { return false; } // a lease holds the frames
    }
    if (control_) { control_->account_fixed(static_cast<std::int64_t>(bytes)); }
    frames_now_ = residency_->frames();
    pool_now_   = residency_->pool_bytes();
    return true;
}

void ExpertOffload::give_frames(std::uint64_t bytes, cudaStream_t compute) {
    const std::lock_guard<std::mutex> lock(vram_mutex_);
    const std::uint32_t frames = residency_->frames();
    (void)residency_->resize(frames + static_cast<std::uint32_t>(bytes / residency_->frame_bytes()), compute, *vram_);
    if (control_) { control_->account_fixed(-static_cast<std::int64_t>(bytes)); }
    frames_now_ = residency_->frames();
    pool_now_   = residency_->pool_bytes();
}

// ------------------------------------------------------------------------------------- warm start

// The SSD tier's RAM fill order (Flash-Next §19.3.7 §4.9): the saved ranking (the last session's VRAM
// residents, best first), then every other key that session used, most used first, then the keys it
// never used, one expert of every layer in turn, so no layer is left wholly on the SSD.
static std::vector<std::uint32_t> tier_fill_order(std::uint32_t layers, std::uint32_t experts,
                                                  const expert_cache::SavedState* state) {
    const std::uint32_t keys = layers * experts;
    const auto spread        = [&](std::uint32_t key) { return (key % experts) * layers + key / experts; };
    std::vector<std::uint32_t> order;
    order.reserve(keys);
    std::vector<std::uint8_t> listed(keys, 0);
    if (state != nullptr) {
        for (const std::uint32_t key : state->ranked) {
            if (key < keys && listed[key] == 0) {
                order.push_back(key);
                listed[key] = 1;
            }
        }
        if (state->counts.size() == keys) {
            std::vector<std::uint32_t> used;
            for (std::uint32_t key = 0; key < keys; ++key) {
                if (listed[key] == 0 && state->counts[key] != 0) { used.push_back(key); }
            }
            std::sort(used.begin(), used.end(), [&](std::uint32_t a, std::uint32_t b) {
                const std::uint32_t ca = state->counts[a], cb = state->counts[b];
                return ca != cb ? ca > cb : spread(a) < spread(b);
            });
            for (const std::uint32_t key : used) {
                order.push_back(key);
                listed[key] = 1;
            }
        }
    }
    for (std::uint32_t i = 0; i < keys; ++i) {
        const std::uint32_t key = (i % layers) * experts + i / layers; // spread(key) == i
        if (listed[key] == 0) { order.push_back(key); }
    }
    return order;
}

// Fills the frames from the saved state before the first request; without one (a first start, or a
// state of another artifact), from the model's calibrated order when it has one (S4b). With the SSD
// tier, RAM is filled first (the frames load from it), then the seeds' RAM copies are released and
// their slots refilled with the next keys of the fill order, so the two levels hold different experts.
void ExpertOffload::warm_start(cudaStream_t compute) {
    const std::uint32_t keys = setup_.layout.layers * setup_.layout.experts;
    expert_cache::ExpertStateLoad load;
    enum class Seed { None, Saved, Calibrated } seed = Seed::None;
    if (!options_.expert_state.empty() && options_.expert_cache) {
        load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, keys);
        if (load.state) {
            seed = Seed::Saved;
        } else if (setup_.calibrated_state != nullptr) {
            diagnostic("expert cache: no saved state used: " + load.message, DiagnosticLevel::Debug);
            load.state = *setup_.calibrated_state;
            seed       = Seed::Calibrated;
        }
    }
    const expert_cache::SavedState* state = load.state ? &*load.state : nullptr;
    // The tier's first RAM pass.
    std::vector<std::uint32_t> order;
    std::uint32_t filled = 0;
    double tier_seconds  = 0.0;
    if (tier_) {
        if (tier_->keys() != keys) { throw std::logic_error("expert offload: the SSD tier has another key count"); }
        const auto began = Clock::now();
        order            = tier_fill_order(setup_.layout.layers, setup_.layout.experts, state);
        if (state != nullptr && state->counts.size() == keys) { tier_->controller().seed_uses(state->counts, kSeedCountCap); }
        filled       = tier_->prefill(order);
        tier_seconds = std::chrono::duration<double>(Clock::now() - began).count();
    }
    // The frames.
    std::uint32_t seeded = 0;
    if (!options_.expert_state.empty() && options_.expert_cache) {
        if (seed == Seed::None) {
            diagnostic("expert cache: no saved state yet, so it starts empty and fills as requests arrive");
            diagnostic("expert cache starts empty: " + load.message, DiagnosticLevel::Debug);
        } else {
            const auto began    = Clock::now();
            const auto max_keys = static_cast<std::uint32_t>(kSeedShare * residency_->frames());
            seeded              = residency_->warm_start(*state, kSeedCountCap, max_keys, compute);
            CUDA_CHECK(cudaStreamSynchronize(compute));
            const double seconds = std::chrono::duration<double>(Clock::now() - began).count();
            const double gib = static_cast<double>(seeded) * static_cast<double>(residency_->frame_bytes()) / (1ULL << 30);
            char line[256];
            if (seed == Seed::Saved) {
                std::snprintf(line, sizeof(line),
                              "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                              seeded, gib, seconds);
            } else {
                std::snprintf(line, sizeof(line),
                              "expert cache: no saved state yet, so it starts with the %u highest-ranked experts of the "
                              "calibrated order (%.1f GiB in %.2f s)",
                              seeded, gib, seconds);
            }
            diagnostic(line);
            state_routed_ = residency_->stats().routed;
        }
        state_saved_ = Clock::now();
    }
    if (!tier_) { return; }
    // The seeds are in VRAM: their RAM copies (shadows) are released and their slots take the next keys
    // of the fill order; the unpinned records take the next keys neither the slots nor the frames hold.
    const auto began       = Clock::now();
    std::uint32_t released = 0, refilled = 0;
    if (seeded > 0) {
        released = tier_->controller().release_shadows();
        if (released > 0) { refilled = tier_->prefill(std::span<const std::uint32_t>(order).subspan(filled)); }
    }
    const std::int32_t* in_vram = residency_->host_table();
    const std::uint32_t unpinned =
        tier_->fill_unpinned(order, [in_vram](std::uint32_t key) { return in_vram != nullptr && in_vram[key] >= 0; });
    const double seconds     = tier_seconds + std::chrono::duration<double>(Clock::now() - began).count();
    const std::uint32_t in_ram = filled - released + refilled;
    const double gib           = static_cast<double>(tier_->record_bytes()) / (1ULL << 30);
    char line[512];
    std::snprintf(line, sizeof(line),
                  "SSD expert tier: %u RAM slots (%.1f GiB) hold %u experts (%s), %u unpinned records %u more, read in %.1f s; "
                  "%u of %u experts are read from the artifact when used",
                  tier_->pinned_slots(), gib * tier_->pinned_slots(), in_ram,
                  seed == Seed::Saved        ? "the saved ranking, then the last session's use counts"
                  : seed == Seed::Calibrated ? "the calibrated expert order"
                                             : "spread across the layers",
                  tier_->unpinned_records(), unpinned, seconds, keys - in_ram - released - unpinned, keys);
    diagnostic(line);
}

void ExpertOffload::save_state() noexcept {
    if (options_.expert_state.empty() || !residency_ || residency_->frames() == 0) { return; }
    try {
        const std::string error =
            expert_cache::save_expert_state(options_.expert_state, options_.expert_state_identity, residency_->saved_state());
        if (!error.empty()) { diagnostic("expert state not saved: " + error, DiagnosticLevel::Warning); }
        state_saved_  = Clock::now();
        state_routed_ = residency_->stats().routed;
    } catch (const std::exception& error) {
        diagnostic(std::string("expert state not saved: ") + error.what(), DiagnosticLevel::Warning);
    } catch (...) {}
}

void ExpertOffload::maintain(cudaStream_t compute) {
    apply_vram_target(compute, true);
    if (options_.expert_state.empty() || !residency_ || residency_->stats().routed == state_routed_) { return; }
    if (Clock::now() - state_saved_ < kStateInterval) { return; }
    save_state();
}

void ExpertOffload::report_layers() noexcept {
    if (!residency_) { return; }
    try {
        const std::span<const std::uint64_t> routed = residency_->routed_by_layer();
        const std::span<const std::uint64_t> hits   = residency_->hits_by_layer();
        if (routed.empty()) { return; }
        std::string text = "expert cache per layer (hit% of routed uses):";
        char part[40];
        std::uint64_t total_routed = 0, total_hits = 0;
        for (std::size_t layer = 0; layer < routed.size(); ++layer) {
            total_routed += routed[layer];
            total_hits += hits[layer];
            std::snprintf(part, sizeof(part), " %zu:%.1f%%(%llu)", layer,
                          routed[layer] ? 100.0 * static_cast<double>(hits[layer]) / static_cast<double>(routed[layer]) : 0.0,
                          static_cast<unsigned long long>(routed[layer]));
            text += part;
        }
        std::snprintf(part, sizeof(part), "; run %.1f%% of %llu",
                      total_routed ? 100.0 * static_cast<double>(total_hits) / static_cast<double>(total_routed) : 0.0,
                      static_cast<unsigned long long>(total_routed));
        text += part;
        diagnostic(text, DiagnosticLevel::Debug);
    } catch (...) {}
}

std::string ExpertOffload::request_report(const ExpertResidency::Stats& at, std::uint64_t cpu_at,
                                          const HostExpertTier::Stats* tier_at) const {
    const auto& s       = residency_->stats();
    const auto routed   = s.routed - at.routed;
    const auto hits     = s.hits - at.hits;
    const auto cpu      = cpu_service_ ? cpu_service_->served_experts() - cpu_at : 0;
    char text[512];
    std::snprintf(text, sizeof(text),
                  "expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, %llu landed, "
                  "%llu misses CPU-served; since start %.1f%%",
                  residency_->frames(), routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                  static_cast<unsigned long long>(routed), static_cast<unsigned long long>(s.promotions - at.promotions),
                  static_cast<unsigned long long>(s.landed - at.landed), static_cast<unsigned long long>(cpu),
                  s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0);
    std::string out = text;
    if (tier_ && tier_at != nullptr) {
        const auto& t    = tier_->stats();
        const auto reads = t.demand_reads - tier_at->demand_reads;
        std::snprintf(text, sizeof(text), "; SSD tier: %llu expert reads (%.2f GiB), %llu failed, %llu for the prefill stream",
                      static_cast<unsigned long long>(reads),
                      static_cast<double>(reads) * static_cast<double>(residency_->frame_bytes()) / (1ULL << 30),
                      static_cast<unsigned long long>(t.demand_failures - tier_at->demand_failures),
                      static_cast<unsigned long long>(t.stream_reads - tier_at->stream_reads));
        out += text;
    }
    return out;
}

void ExpertOffload::report_request(const ExpertResidency::Stats& at, std::uint64_t cpu_at,
                                   const HostExpertTier::Stats* tier_at) const noexcept {
    try {
        diagnostic(request_report(at, cpu_at, tier_at), DiagnosticLevel::Debug);
    } catch (...) {}
}

std::uint64_t ExpertOffload::cpu_served() const noexcept { return cpu_service_ ? cpu_service_->served_experts() : 0; }

// ------------------------------------------------------------------------------------- rounds

void ExpertOffload::before_round(cudaStream_t compute, bool landing, bool walking, std::uint32_t stream_slots) {
    if (cpu_service_) { cpu_service_->wake(); } // its first request comes within this round
    residency_->before_round(compute, landing, walking, stream_slots);
}

void ExpertOffload::check_error(const std::string& model) {
    auto* word           = static_cast<volatile std::uint32_t*>(error_.data());
    std::uint32_t value  = *word;
    if (value == 0 && testing::take_expert_fault()) { value = EIO; } // layer 0
    if (value == 0) { return; }
    *word         = 0;
    failed_round_ = true;
    // A stopped service or read agent fails every later call too: stop the Engine with its cause.
    if (cpu_service_) {
        if (const std::string why = cpu_service_->failure(); !why.empty()) {
            throw std::runtime_error(model + ": the CPU expert service stopped (" + why + ")");
        }
    }
    if (tier_) {
        if (const std::string why = tier_->failure(); !why.empty()) {
            throw std::runtime_error(model + ": the SSD expert tier's read agent stopped (" + why + ")");
        }
    }
    const std::uint32_t code = value & 0xFFFFU;
    const std::string what   = code == moe::kErrorUnservedRecord ? "an SSD-only expert had no path to it"
                               : code == moe::kErrorHostSilent   ? "the CPU expert service stopped answering"
                               : code == moe::kErrorFetchSilent  ? "the SSD expert tier's read agent stopped answering"
                                                                 : "an expert record could not be read (" +
                                                                     std::generic_category().message(static_cast<int>(code)) + ")";
    throw RecoverableExecutionError(model + " layer " + std::to_string(value >> 16U) + ": " + what);
}

std::size_t ExpertOffload::decode_budget(std::uint32_t tokens) {
    // The fill phase: until this session promoted or landed as many experts as there are frames.
    const ExpertResidency::Stats& stats = residency_->stats();
    if (stats.promotions + stats.landed < residency_->frames()) { return kDecodePromotionsPerLayer * tokens; }
    const std::uint64_t interval = promotion_interval();
    budget_tokens_ += tokens;
    const std::uint64_t due = budget_tokens_ / interval;
    budget_tokens_ %= interval;
    return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
}

void ExpertOffload::trace_round(const TracedRound& round, std::uint32_t tokens, std::size_t budget,
                                std::span<const std::uint8_t> live) {
    if (!trace_) { return; }
    trace_->append({.kind     = round.kind,
                    .rows     = static_cast<std::uint32_t>(round.rows),
                    .width    = static_cast<std::uint32_t>(round.width),
                    .tokens   = tokens,
                    .budget   = static_cast<std::uint32_t>(std::min<std::size_t>(budget, 0xFFFFFFFFU)),
                    .position = round.position,
                    .frames   = residency_->frames()},
                   round.lanes, live, residency_->route_host(), residency_->route_stride());
}

// A traced round's lanes, one per row (the residency's route predictor, INFERNIX_TMP_ROUTE_PROTECT).
static std::span<const std::int32_t> round_lanes(const ExpertOffload::TracedRound& round) noexcept {
    return round.lanes.first(std::min(round.lanes.size(), static_cast<std::size_t>(std::max(round.rows, 0))));
}

void ExpertOffload::defer(std::int32_t columns, std::uint32_t tokens, const TracedRound& round,
                          std::span<const std::uint8_t> live) {
    deferred_.columns = columns;
    deferred_.tokens  = tokens;
    deferred_.live.assign(live.begin(), live.end());
    deferred_.round = round;
    deferred_.lanes.assign(round.lanes.begin(), round.lanes.end());
    deferred_.round.lanes = deferred_.lanes;
    deferred_.pending     = true;
}

void ExpertOffload::settle(cudaStream_t compute) {
    if (!deferred_.pending) { return; }
    deferred_.pending        = false;
    const std::size_t budget = decode_budget(deferred_.tokens);
    const std::span<const std::uint8_t> live(deferred_.live.data(), deferred_.live.size());
    trace_round(deferred_.round, deferred_.tokens, budget, live);
    residency_->after_round(compute, deferred_.columns, budget, live, round_lanes(deferred_.round));
    apply_vram_target(compute, false);
}

void ExpertOffload::after_call(cudaStream_t compute, std::int32_t columns, std::uint32_t tokens,
                               std::size_t promotions, const TracedRound& round, std::span<const std::uint8_t> live) {
    trace_round(round, tokens, promotions, live);
    residency_->after_round(compute, columns, promotions, live, round_lanes(round));
    apply_vram_target(compute, false);
}

void ExpertOffload::note_prompt_routes(std::uint32_t lane, std::int32_t columns) {
    if (columns <= 0) { return; }
    const std::uint32_t L = setup_.layout.layers, E = setup_.layout.experts, K = setup_.layout.top_k;
    if (prompt_.size() <= lane) { prompt_.resize(static_cast<std::size_t>(lane) + 1); }
    PromptUses& prompt = prompt_[lane];
    if (prompt.uses.size() != static_cast<std::size_t>(L) * E) {
        prompt.uses.assign(static_cast<std::size_t>(L) * E, 0);
        prompt.counted = 0;
    }
    const std::int32_t* routes = residency_->route_host();
    const std::size_t stride   = residency_->route_stride();
    for (std::uint32_t l = 0; l < L; ++l) {
        const std::int32_t* ids = routes + static_cast<std::size_t>(l) * stride;
        std::uint32_t* uses     = prompt.uses.data() + static_cast<std::size_t>(l) * E;
        for (std::size_t i = 0; i < static_cast<std::size_t>(columns) * K; ++i) {
            const std::int32_t expert = ids[i];
            if (expert >= 0 && static_cast<std::uint32_t>(expert) < E) { ++uses[expert]; }
        }
    }
    prompt.counted += static_cast<std::uint32_t>(columns);
}

void ExpertOffload::seed_prompt(std::uint32_t lane, cudaStream_t compute) {
    if (lane >= prompt_.size() || prompt_[lane].counted == 0) { return; }
    // INFERNIX_TMP_PROMPT_SEED overrides the credit and adds promotions at the prompt's end.
    const OffloadSwitches& switches = temporary_switches::offload();
    residency_->seed_prompt(prompt_[lane].uses, switches.credit_or(kPromptSeedCredit) / prompt_[lane].counted,
                            switches.prompt_budget, compute);
    clear_prompt_uses(lane);
}

void ExpertOffload::clear_prompt_uses(std::uint32_t lane) {
    if (lane >= prompt_.size()) { return; }
    std::fill(prompt_[lane].uses.begin(), prompt_[lane].uses.end(), 0U);
    prompt_[lane].counted = 0;
}

// ------------------------------------------------------------------------------------- leases

bool ExpertOffload::lend_stream(std::int32_t width, std::uint64_t wide_bytes, std::uint64_t walk_bytes,
                                cudaStream_t compute) {
    if (!stream_ || width < kStreamMinColumns || residency_->frames() == 0) { return false; }
    if (leases_.ring.valid()) { return true; }
    // The ring holds two halves of as many records as the layer with the most non-resident experts has
    // (+ one slot for the slot tables), so no streamed layer leaves misses to staging (F2: halves of 384
    // of a 512-expert layer were 9 % slower at pp4096). Each lease is one contiguous run of one pool's
    // frames; the frames a lease takes evict experts of that pool's layers, so each half widens by the
    // most any lease takes per layer.
    const std::int32_t* table = residency_->host_table();
    const std::uint32_t E     = setup_.layout.experts;
    std::uint32_t most        = 0;
    for (std::uint32_t l = 0; l < setup_.layout.layers; ++l) {
        std::uint32_t absent = 0;
        for (std::uint32_t e = 0; e < E; ++e) { absent += table[static_cast<std::size_t>(l) * E + e] < 0 ? 1U : 0U; }
        most = std::max(most, absent);
    }
    const std::uint64_t slot     = record_stride_;
    const cudaStream_t writers[] = {stream_->stream()};
    const auto per_layer         = [&](const ExpertResidency::FrameLease& lease) {
        const std::uint32_t layers = std::max(1U, residency_->lease_layers(lease));
        return (lease.count + layers - 1) / layers;
    };
    ExpertResidency::FrameLease wide, walk;
    if (wide_bytes > 0) {
        if (wide_bytes + residency_->lend_pitch() > residency_->lendable_bytes()) { return false; }
        wide = residency_->lend(wide_bytes, compute, writers);
    }
    if (walk_bytes > 0 && walk_bytes + 2 * (kStreamHalfMin + 1) * slot <= residency_->lendable_bytes()) {
        walk = residency_->lend(walk_bytes, compute, writers);
    }
    const std::uint32_t widened    = std::min(E, most + std::max(per_layer(wide), per_layer(walk)));
    const auto room_slots          = static_cast<std::uint32_t>(residency_->lendable_bytes() / slot);
    const std::uint32_t half       = std::min(widened, room_slots > 1 ? (room_slots - 1) / 2 : 0U);
    if (half < kStreamHalfMin) {
        if (walk.valid()) { residency_->give_back(walk, compute); }
        if (wide.valid()) { residency_->give_back(wide, compute); }
        return false;
    }
    leases_.ring = residency_->lend((2ULL * half + 1) * slot, compute, writers);
    leases_.wide = wide;
    leases_.walk = walk;
    return true;
}

void ExpertOffload::return_stream(cudaStream_t compute) {
    if (!leases_.ring.valid()) { return; }
    residency_->give_back(leases_.ring, compute);
    residency_->give_back(leases_.wide, compute);
    residency_->give_back(leases_.walk, compute);
    leases_ = {};
}

ExpertResidency::FrameLease ExpertOffload::lend(std::uint64_t bytes, cudaStream_t compute,
                                                std::span<const cudaStream_t> writers) {
    if (residency_->frames() == 0 || bytes + residency_->lend_pitch() > residency_->lendable_bytes()) { return {}; }
    return residency_->lend(bytes, compute, writers);
}

namespace testing {

namespace {
std::atomic<std::uint32_t> g_expert_fault{0};
} // namespace

void set_expert_fault(std::uint32_t checks) noexcept { g_expert_fault.store(checks); }

bool take_expert_fault() noexcept {
    std::uint32_t left = g_expert_fault.load(std::memory_order_relaxed);
    while (left != 0) {
        if (g_expert_fault.compare_exchange_weak(left, left - 1)) { return left == 1; }
    }
    return false;
}

} // namespace testing

} // namespace infernix::runtime::expert_offload
