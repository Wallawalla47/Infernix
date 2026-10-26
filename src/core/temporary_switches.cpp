#include "core/temporary_switches.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace infernix::temporary_switches {
namespace {

// Flag: a 0/1 switch, which describe() validates; Value: parsed by its reader.
enum class Kind : std::uint8_t { Flag, Value };

struct Registration {
    std::string_view name;
    Scope scope;
    Kind kind;
};

// Every switch, grouped by the change that added it; the reader named beside each parses it.
constexpr Registration kSwitches[] = {
    // Q5/Q6 prefill MMA (q5/q6_a16_mma, the sparse-MoE qx_down): integer codes into the MMA, each
    // 64-group scale on its FP32 partial (ops/linear/q5/q5_mma_launch.cuh). Bit-changing, default 0.
    {"INFERNIX_TMP_Q56_FP32_SCALE", Scope::Qwen3_5, Kind::Flag},
    // Groupwise Q4/Q5/Q6 sliced-K kernels (ops/linear/common/sliced_k_route.cpp): a forced <rows>x<stages>
    // shape of the one-wave pipelined kernel. Same bits.
    {"INFERNIX_TMP_GW_SHAPE", Scope::Qwen3_5, Kind::Value},
    // batched-a8-subset-decode's scope (a8_subset).
    {"INFERNIX_TMP_A8_SUBSET_MIN_COLUMNS", Scope::Qwen3_5, Kind::Value},
    {"INFERNIX_TMP_A8_SUBSET_CLASSES", Scope::Qwen3_5, Kind::Value},
    {"INFERNIX_TMP_A8_SUBSET_LAYERS", Scope::Qwen3_5, Kind::Value},
    // Qwen3.5 MTP drafting (models/qwen3_5/program/planning/startup.cpp, mtp_draft_switches): fused
    // draft probabilities for resident targets, the chain's device gate, calibrated extents and a
    // per-round draft log path. Draft-only.
    {"INFERNIX_TMP_QWEN35_MTP_PROBS", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_QWEN35_MTP_GATE", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_QWEN35_MTP_CALIBRATED", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_QWEN35_MTP_DRAFT_LOG", Scope::Qwen3_5, Kind::Value},
    // The offloaded-expert MTP drafter's round cost (Qwen3.5-122B-A10B; models/qwen3_5/program/planning/
    // startup.h MtpDraftSwitches; candidates §11 OR-09). Output-neutral: drafts only.
    {"INFERNIX_TMP_QWEN35_MTP_PAD_MASK", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_QWEN35_MTP_STEP_GATE", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_QWEN35_MTP_SKIP_ROUNDS", Scope::Qwen3_5, Kind::Flag},
    // Column-invariant verification attention as rows of one launch after one append (ops/
    // softmax_attention/dense/causal_cache/causal_softmax_attention.cpp, verify_rows_attention): BF16,
    // INT8, FP8, NVFP4 and K8V4 KV. Same bits.
    {"INFERNIX_TMP_VERIFY_ROWS_ATTENTION", Scope::Qwen3_5, Kind::Flag},
    // The 35B small-T MoE (ops/sparse_moe, candidates §10.3 KW-06): D4 as D3's programmatic dependent,
    // and T = 1's L2 hints (S1 warms the shared down codes, D4's tail the next consumer). Same bits.
    {"INFERNIX_TMP_SMALL_T_D4_PDL", Scope::Qwen3_5, Kind::Flag},
    {"INFERNIX_TMP_SMALL_T_L2_HINTS", Scope::Qwen3_5, Kind::Flag},
    // The scheduler's decode share (runtime/engine/engine_core.h, decode_share_from_environment): a
    // prefill unit's share of decode batches and the remaining tokens below which it stops. Same bits.
    {"INFERNIX_TMP_DECODE_SHARE", Scope::All, Kind::Value},
    {"INFERNIX_TMP_DECODE_SHARE_MIN_TOKENS", Scope::All, Kind::Value},
    // Flash-Next's prefill walk step while another lane decodes (models/qwen4_exp/program/prefill_walk.cpp).
    {"INFERNIX_TMP_DECODE_SHARE_WALK_TOKENS", Scope::Qwen4Exp, Kind::Value},
    // Diagnostic: the CPU team's rates on 32 records in three memories, logged at startup
    // (runtime/expert_offload/expert_offload.cpp, cpu_probe_detail; candidates §11.2 OR-08).
    {"INFERNIX_TMP_CPU_PROBE_DETAIL", Scope::All, Kind::Flag},
    // The expert offload stack's placement switches (offload()).
    {"INFERNIX_TMP_LINK_SHARE", Scope::All, Kind::Value},
    {"INFERNIX_TMP_PLAIN_DIVISOR_ALL", Scope::All, Kind::Value},
    {"INFERNIX_TMP_CPU_PLACEMENT", Scope::All, Kind::Value},
    {"INFERNIX_TMP_ENGINE_ISOLATE", Scope::All, Kind::Value},
    {"INFERNIX_TMP_PROMPT_SEED", Scope::All, Kind::Value},
    {"INFERNIX_TMP_STAGING_SLOTS", Scope::All, Kind::Value},
    // GGUF expert tiles (ops::MoeGgmlTiles::from_environment): the stage-decode column interval, the
    // decoded planes' threshold and shared-memory codebooks. Same bits.
    {"INFERNIX_TMP_IQ_STAGE_DECODE_MIN_COLUMNS", Scope::Qwen4Exp, Kind::Value},
    {"INFERNIX_TMP_IQ_STAGE_DECODE_MAX_COLUMNS", Scope::Qwen4Exp, Kind::Value},
    {"INFERNIX_TMP_IQ_PLANE_MIN_COLUMNS", Scope::Qwen4Exp, Kind::Value},
    {"INFERNIX_TMP_GGML_SHARED_CODEBOOKS", Scope::Qwen4Exp, Kind::Flag},
    // QSA prompt kernel's split K/V stage (ops/qsa/qsa_prompt.cu, qsa_prompt_v2). Same bits.
    {"INFERNIX_TMP_QSA_PROMPT_V2", Scope::Qwen4Exp, Kind::Flag},
    // Flash-Next n-gram copy proposals for structured output (models/qwen4_exp/program/program_impl.h):
    // a one-row round keeps a copy only when its first token is the neural drafter's first draft, and
    // the copies' learned acceptance per match-length bucket is shared by the process's requests.
    // Draft-only.
    {"INFERNIX_TMP_NGRAM_AGREE", Scope::Qwen4Exp, Kind::Flag},
    {"INFERNIX_TMP_NGRAM_PROCESS_STATS", Scope::Qwen4Exp, Kind::Flag},
    // Flash-Next n-gram rows of a prompt (candidates §8.3 GX-08): RAM-mode calls of 16,384+ rows gathered
    // on eight threads (models/qwen4_exp/program/ngram_volume.cpp), and the first walk span's later
    // chunks read on a worker while its first chunk runs (prefill_walk.cpp). Same bits.
    {"INFERNIX_TMP_NGRAM_PARALLEL_GATHER", Scope::Qwen4Exp, Kind::Flag},
    {"INFERNIX_TMP_NGRAM_FIRST_SPAN_ASYNC", Scope::Qwen4Exp, Kind::Flag},
    // Debug diagnostic: the prefill projection inputs' FP16 headroom scan, written to this path
    // (models/qwen4_exp/execution/activation_scan.cpp).
    {"INFERNIX_TMP_ACTIVATION_SCAN", Scope::Qwen4Exp, Kind::Value},
    // Hyper-connection mixers with a BF16 W_up (IQ3_S, FP8, FP4-Local, recipe A): wide calls compute
    // the up linear and the collapse in one kernel (ops/hyper_connection/hyper_connection_up_collapse.cu;
    // candidates §7.4 P4). Designed to give the composed route's bits.
    {"INFERNIX_TMP_HC_BF16_UP_COLLAPSE", Scope::Qwen4Exp, Kind::Flag},
    // QSA block selection without the histogram pass: a strided score sample sets a key floor and one
    // filter pass per slice gathers the candidates (ops/qsa/qsa_select.cu; candidates §13 PUB-02).
    // Same ids.
    {"INFERNIX_TMP_QSA_SAMPLED_SELECT", Scope::Qwen4Exp, Kind::Flag},
    // W4A4 and two-term narrow expert kernels: a job of two or more columns runs them two at a time,
    // converting each weight unit once per pair (ops/offloaded_sparse_moe/cuda/moe_layer.cu, "column
    // pairs"; candidates §0j-1). Same bits.
    {"INFERNIX_TMP_NARROW_PAIRS", Scope::All, Kind::Flag},
    // 35B (Q4 routed experts) small-T D3: two Q4 quads' loads in flight per warp at T >= 2
    // (ops/sparse_moe/decode/sparse_moe_decode_kernels.cu; candidates §10 GX-02). Same bits.
    {"INFERNIX_TMP_D3_TWO_QUADS", Scope::Qwen3_5, Kind::Flag},
    // Flash-Next's staging and shared-expert streams and the 122B's fork / staging-overlap stream
    // (runtime/expert_offload) at the device's greatest priority, and every CUDA Graph instantiated with node
    // priorities (core/decode_graph.cpp; candidates §11 D5). Same bits.
    {"INFERNIX_TMP_STREAM_PRIORITY", Scope::All, Kind::Flag},
    // Flash-Next MTP: a verification round's drafter catch-up, then the next round's drafts (at the
    // length its policy will choose, taken when that round is these rows again), on a stream of
    // greater priority beside the target's commit kernels (models/qwen4_exp/program/program_impl.h
    // commit_verified, mtp_pre_draft; candidates §13 SR-01). Same bits.
    {"INFERNIX_TMP_DRAFTER_STREAM", Scope::Qwen4Exp, Kind::Flag},
    // INT8 KV attention: the 16/2 geometry's (35B) rows of two to four columns on the pipelined
    // kernel (ops/softmax_attention/dense/causal_cache/int8/launch.cu; candidates KW-10 (b)). Same bits.
    {"INFERNIX_TMP_INT8_PIPELINED16", Scope::Qwen3_5, Kind::Flag},
    // Flash-Next: each block's attention-half inject folded into the MLP half's fused mixer reads
    // (ops::hyper_connection_inject_mix; models/qwen4_exp/execution/forward.cpp; candidates §7.3 D6).
    // Same bits.
    {"INFERNIX_TMP_HC_INJECT_FOLD", Scope::Qwen4Exp, Kind::Flag},
    // Offloaded experts: =<n> (1-64) runs a routing-sequence predictor beside the expert cache whose next
    // n experts per layer and lane each eviction spares while another resident is left
    // (runtime/expert_offload/route_predictor.h; candidates §13 PUB-22; A/B only after its replay
    // gains 3 points). Same bits (placement only).
    {"INFERNIX_TMP_ROUTE_PROTECT", Scope::All, Kind::Value},
    // Qwen3.5 DFlash2 with --draft-tree-nodes auto: each automatic round proposes first, then chooses
    // its width (the chain or a tree) from the round's own draft probabilities, calibrated by measured
    // gains, and the round's measured costs (models/qwen3_5/program/decode.cpp tree block rounds;
    // candidates §13 PUB-03). Exact: a width depends on the main chain's draws only.
    {"INFERNIX_TMP_TREE_ROUND_WIDTH", Scope::Qwen3_5, Kind::Flag},
    // =<rounds>: the replay bench (candidates §8.5 CF-03). The expert residency freezes after that many
    // rounds (its loads landed, then no admission, eviction, landing or prompt seed); Flash-Next's MTP
    // width cost keeps its prior, so greedy rounds repeat across runs; its release log reports the
    // decode rounds' device time and the CPU waits inside it (core/round_clock.h). Same bits.
    {"INFERNIX_TMP_REPLAY_BENCH", Scope::All, Kind::Value},
    // Resident drafter experts (Flash-Next's and the 122B's MTP MoE): gate/up and down as programmatic
    // dependents in captured graphs, the down kernel prefetching its expert rows into L2 before it waits
    // (ops/resident_moe/resident_moe.cu; candidates §2.3 W5). Same bits.
    {"INFERNIX_TMP_RESIDENT_MOE_PDL", Scope::All, Kind::Flag},
    // Flash-Next hyper-connection mix, calls of 2-8 columns: the down partials on a grid of one CTA per
    // (row group, stream, column) instead of one per (row group, stream) looping over the columns
    // (ops/hyper_connection/hyper_connection_mix_fused.cu; candidates §2.3 ST-03). Same bits.
    {"INFERNIX_TMP_HC_DOWN_COLUMNS", Scope::Qwen4Exp, Kind::Flag},
    // =<file>: the offloaded-expert models' startup link and CPU expert team rates (Flash-Next's, and the
    // Qwen3.5-122B-A10B's through runtime/expert_offload) are checked against the last good ones saved in
    // that file; a reading below 0.85 of the saved rate is measured again, and the saved rate kept when
    // that is low too (core/machine_calibration.h; candidates §7.3 D11). Same bits.
    {"INFERNIX_TMP_MACHINE_CALIBRATION", Scope::All, Kind::Value},
    // Flash-Next elastic KV: the idle shrink also runs while requests decode (after kKvIdleShrink without
    // an admission or release), releasing idle conversations' host-backed device blocks so the free top
    // returns to the expert frames (models/qwen4_exp/program/program_impl.h apply_vram_target;
    // candidates §8.3 CF-06). Same bits.
    {"INFERNIX_TMP_KV_SHRINK_BUSY", Scope::Qwen4Exp, Kind::Flag},
    // Windows: the linear TMA kernels read their descriptors from the kernel parameters (an 8-byte-
    // aligned word array, used in place once a probe launch finds it on a 64-byte boundary) instead of a
    // staging kernel's device copy before every eager launch (core/tma_descriptor_staging.cuh
    // TmaDescriptorParameter: kernel instantiations of their own, so the default ones are unchanged;
    // candidates §2.3 GX-03). Same bits.
    {"INFERNIX_TMP_TMA_PARAM_WORDS", Scope::All, Kind::Flag},
    // Flash-Next QSA layers: the output gate applied in the attention's split merge instead of a separate
    // sigmoid_mul launch (ops/qsa/qsa.cu merge_kernel; candidates §7.3 D7). Same bits.
    {"INFERNIX_TMP_QSA_GATE_MERGE", Scope::Qwen4Exp, Kind::Flag},
    // Flash-Next eager prefill chunks: the shared expert on its side stream beside the routed experts,
    // as in decode (models/qwen4_exp/execution/forward.cpp; candidates W18). Same bits.
    {"INFERNIX_TMP_SHARED_SIDE_PREFILL", Scope::Qwen4Exp, Kind::Flag},
    // Flash-Next MTP: the argmax drafts' second-ranked head rows, and at each lane's release how often a
    // greedy row's first rejected draft was that second choice (models/qwen4_exp/program/program_impl.h;
    // candidates GX-09: draft trees need above about 0.20). Same bits.
    {"INFERNIX_TMP_MTP_RUNNER_UP", Scope::Qwen4Exp, Kind::Flag},
    // Serve: before READY, max-concurrency warm-up requests of staggered lengths, so the decode,
    // verification and draft graphs of every batch size are captured before the first request
    // (serve/generation_service.cpp warmup; candidates §7.6 D14). Same bits.
    {"INFERNIX_TMP_WARMUP_GRAPHS", Scope::All, Kind::Flag},
    // Offloaded-expert caches: =<fraction> (0, 1]; a prompt whose reused prefix covers at least that much
    // of it neither scores its prefill routes nor seeds its experts (Flash-Next: models/qwen4_exp/program/
    // program_impl.h apply_prompt_routes; the 122B: models/qwen3_5/program/offload.cpp end_pass; candidates
    // CF-01). Same bits (placement only).
    {"INFERNIX_TMP_CONTINUING_PROMPT", Scope::All, Kind::Value},
    // VQ2 / K4V2 KV append: calls of at least one row per SM on the parity-sorted split encoder, fewer
    // on the block kernel staging the FP32 codebook; _SPLIT=1|2|4|8 forces the split encoder at that
    // many pattern ranges (ops/kv_cache/append/vq2_launch.cu; candidates §2.3 VQ append). Same bits.
    {"INFERNIX_TMP_VQ_APPEND_V2", Scope::All, Kind::Flag},
    {"INFERNIX_TMP_VQ_APPEND_SPLIT", Scope::All, Kind::Value},
    // The 122B's offloaded-expert prefill: the wide arena lent from the expert cache at the pass width on the
    // planner's ladder, not the configured chunk (models/qwen3_5/program/offload.cpp wide_bytes; candidates
    // BX-01, Flash-Next's default since its A/B). Same bits (the arena holds intermediates only).
    {"INFERNIX_TMP_PROMPT_SIZED_LOAN", Scope::All, Kind::Flag},
    // Flash-Next n-gram volume: a large call's 8 reader threads take its blocks in a scattered (stride
    // permutation) order instead of contiguous ascending ranges (models/qwen4_exp/program/ngram_volume.cpp;
    // candidates BX-04, Basalt edbdde8). Same rows and cache contents.
    {"INFERNIX_TMP_NGRAM_SCATTER", Scope::Qwen4Exp, Kind::Flag},
    // Flash-Next MTP: =<base threshold> in (0, 1]: a one-row chain keeps draft j while the product of its
    // drafts' binned acceptance rates stays at or above the request's adaptive threshold, instead of the
    // --spec-min-p cut (models/qwen4_exp/program/program_impl.h MtpCumulative; candidates §7 D8, Basalt
    // draft_cum.hpp; Basalt's base is 0.35). Draft-only.
    {"INFERNIX_TMP_MTP_CUMULATIVE", Scope::Qwen4Exp, Kind::Value},
    // Flash-Next QSA layers: the [13952,2560] group projection (Q8 Dense8, MXFP8 FP4-Local) writes its query,
    // gate, key, value and index parts directly, without the [13952,T] result and its split_rows copy
    // (ops::linear_qsa_group; models/qwen4_exp/execution/forward.cpp; candidates §7 D7). Same bits.
    {"INFERNIX_TMP_QSA_GROUP_DIRECT", Scope::Qwen4Exp, Kind::Flag},
    // f16-accumulate-prefill on Q8 (Dense8): past 8,192 tokens only the problems measured faster take the route
    // (ops/linear/q8/q8_f16acc.cu q8_f16acc_admits).
    {"INFERNIX_TMP_Q8_F16ACC_ADMIT", Scope::Qwen4Exp, Kind::Flag},
    // CPU expert team (A16 and two-term A4 rounds): each distinct x column encoded once per round, not once per
    // job that reads it (ops/offloaded_sparse_moe/cpu/expert_team.cpp x_owner_). Same bits.
    {"INFERNIX_TMP_CPU_SHARED_X", Scope::All, Kind::Flag},
    // Flash-Next dflash2-pairs: DFlash2 rounds of up to <value> rows (2..8; 2: today's pairs only)
    // (models/qwen4_exp/program/program_impl.h dflash2_rows_). Bits as dflash2-pairs'.
    {"INFERNIX_TMP_DFLASH2_ROWS", Scope::Qwen4Exp, Kind::Value},
    // Flash-Next MTP and DFlash2: a one-row round's confident drafts are followed by the n-gram copy that
    // continues them (as long a prefix as its learned acceptance pays for), instead of a copy only ever
    // replacing the drafts (models/qwen4_exp/program/program_impl.h chain_copy; candidates §13 lookup chain).
    // Draft-only.
    {"INFERNIX_TMP_NGRAM_CHAIN", Scope::Qwen4Exp, Kind::Flag},
    // Drafting: Flash-Next's MTP acceptance per position and DFlash2's per probability decile, and the
    // Qwen3.5 adaptive-dflash calibration, are learned across requests, and a new request starts from them
    // instead of the fixed priors (models/qwen4_exp/program/program_impl.h draft_process_stats_,
    // models/qwen3_5/program/speculative/dflash_width_policy.cpp learn; candidates VS-2). Draft-only.
    {"INFERNIX_TMP_DRAFT_PROCESS_STATS", Scope::All, Kind::Flag},
    // Offloaded experts' wide prefill GEMMs (W4A4 and the A4-residual fast paths: Dense8, FP4-Local, recipe C,
    // Qwen3.5-122B): 64-token warp tiles, each decoded weight fragment feeding four MMA tiles instead of two
    // (ops/offloaded_sparse_moe/cuda/wide_expert.cuh WarpTile64; candidates P5, Basalt ebf6928), in calls of at
    // least <value> columns per expert on average (1: every call). Same bits.
    {"INFERNIX_TMP_WIDE_EXPERT_T64", Scope::All, Kind::Value},
};
constexpr std::size_t kCount = std::size(kSwitches);

const char* environment(const char* name) {
#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4996) // std::getenv: read once, before any thread could change it
#endif
    return std::getenv(name);
#ifdef _MSC_VER
#    pragma warning(pop)
#endif
}

// Every registered switch's raw value, read together once.
const std::array<std::optional<std::string>, kCount>& snapshot() {
    static const auto values = [] {
        std::array<std::optional<std::string>, kCount> out;
        for (std::size_t i = 0; i < kCount; ++i) {
            if (const char* v = environment(std::string(kSwitches[i].name).c_str()); v != nullptr && *v != '\0') {
                out[i] = std::string(v);
            }
        }
        return out;
    }();
    return values;
}

std::size_t index_of(std::string_view name) {
    for (std::size_t i = 0; i < kCount; ++i) {
        if (kSwitches[i].name == name) { return i; }
    }
    throw std::logic_error("temporary switch " + std::string(name) + " is not registered (core/temporary_switches.cpp)");
}

[[noreturn]] void malformed(std::string_view name, const std::string& text, const char* expected) {
    throw std::invalid_argument(std::string(name) + "=" + text + ": expected " + expected);
}

bool parse_long(const std::string& text, long lo, long hi, long& out) {
    if (text.empty()) { return false; }
    char* end = nullptr;
    errno     = 0;
    out       = std::strtol(text.c_str(), &end, 10);
    return errno == 0 && end != nullptr && *end == '\0' && out >= lo && out <= hi;
}

bool parse_double(const std::string& text, double& out) {
    if (text.empty()) { return false; }
    char* end = nullptr;
    errno     = 0;
    out       = std::strtod(text.c_str(), &end);
    return errno == 0 && end != nullptr && *end == '\0' && std::isfinite(out);
}

// "a,b" -> a and b (no comma: b empty).
std::pair<std::string, std::string> split_comma(const std::string& text) {
    const auto comma = text.find(',');
    if (comma == std::string::npos) { return {text, {}}; }
    return {text.substr(0, comma), text.substr(comma + 1)};
}

// Splits `text` at commas, skipping empty items.
template <class Item>
void for_each_item(std::string_view text, Item&& item) {
    while (!text.empty()) {
        const std::size_t comma     = text.find(',');
        const std::string_view head = text.substr(0, comma);
        if (!head.empty()) { item(head); }
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
}

constexpr std::string_view kClassNames[kA8SubsetClasses] = {"input", "gate_up", "output", "down", "head"};

std::int32_t layer_index(std::string_view text) {
    long out = 0;
    if (!parse_long(std::string(text), 0, 1'000'000, out)) {
        malformed("INFERNIX_TMP_A8_SUBSET_LAYERS", std::string(text), "nonnegative layer indices");
    }
    return static_cast<std::int32_t>(out);
}

A8Subset read_a8_subset() {
    A8Subset out;
    out.min_columns =
        static_cast<std::int32_t>(integer("INFERNIX_TMP_A8_SUBSET_MIN_COLUMNS", out.min_columns, 2, 1'000'000));
    if (const auto v = value("INFERNIX_TMP_A8_SUBSET_CLASSES")) {
        out.classes.fill(false);
        for_each_item(*v, [&](std::string_view item) {
            for (std::size_t kind = 0; kind < kA8SubsetClasses; ++kind) {
                if (item == kClassNames[kind]) {
                    out.classes[kind] = true;
                    return;
                }
            }
            malformed("INFERNIX_TMP_A8_SUBSET_CLASSES", *v, "classes among input, gate_up, output, down, head");
        });
    }
    if (const auto v = value("INFERNIX_TMP_A8_SUBSET_LAYERS"); v && *v != "all") {
        for_each_item(*v, [&](std::string_view item) {
            const std::size_t dash   = item.find('-');
            const std::int32_t first = layer_index(item.substr(0, dash));
            const std::int32_t last  = dash == std::string_view::npos ? first : layer_index(item.substr(dash + 1));
            if (last < first) { malformed("INFERNIX_TMP_A8_SUBSET_LAYERS", *v, "nonempty ranges"); }
            out.layers.emplace_back(first, last);
        });
        if (out.layers.empty()) { malformed("INFERNIX_TMP_A8_SUBSET_LAYERS", *v, "at least one layer range"); }
    }
    return out;
}

GwShape read_gw_shape() {
    const auto v = value("INFERNIX_TMP_GW_SHAPE");
    if (!v) { return {}; }
    const auto bad = [&] { malformed("INFERNIX_TMP_GW_SHAPE", *v, "<rows>x<stages>, rows 16, 32 or 64 and stages 1-4 (for example 32x3)"); };
    const std::size_t x = v->find('x');
    long rows = 0, stages = 0;
    if (x == std::string::npos || !parse_long(v->substr(0, x), 16, 64, rows) || !parse_long(v->substr(x + 1), 1, 4, stages) ||
        (rows != 16 && rows != 32 && rows != 64)) {
        bad();
    }
    return {static_cast<int>(rows), static_cast<int>(stages)};
}

OffloadSwitches read_offload() {
    OffloadSwitches s;
    const auto reject = [&](const char* name, const std::string& v) { s.errors.push_back(std::string(name) + "=" + v); };
    if (const auto v = value("INFERNIX_TMP_LINK_SHARE")) {
        double share = 0.0;
        if (*v == "auto") {
            s.link_share = OffloadSwitches::LinkShare::kAuto;
        } else if (parse_double(*v, share) && share > 0.0 && share <= 1.0) {
            s.link_share       = OffloadSwitches::LinkShare::kFixed;
            s.link_share_fixed = share;
        } else {
            reject("INFERNIX_TMP_LINK_SHARE", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_PLAIN_DIVISOR_ALL")) {
        long a = 0, b = 0;
        const auto [first, second] = split_comma(*v);
        if (*v == "on" || *v == "1") {
            s.plain_divisor_all = true;
        } else if (parse_long(first, 1, 64, a) && parse_long(second, 1, 64, b)) {
            s.plain_divisor_all = true;
            s.plain_short       = static_cast<std::int32_t>(a);
            s.plain_long        = static_cast<std::int32_t>(b);
        } else if (*v != "off" && *v != "0") {
            reject("INFERNIX_TMP_PLAIN_DIVISOR_ALL", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_CPU_PLACEMENT")) {
        const auto [mode, count] = split_comma(*v);
        long workers             = 0;
        const bool counted       = count.empty() || parse_long(count, 1, 40, workers);
        if (counted && (mode == "default" || mode == "physical" || mode == "physical-set")) {
            s.placement = mode == "physical"       ? OffloadSwitches::Placement::kPhysical
                          : mode == "physical-set" ? OffloadSwitches::Placement::kPhysicalSet
                                                   : OffloadSwitches::Placement::kDefault;
            s.workers   = static_cast<std::uint32_t>(workers);
        } else {
            reject("INFERNIX_TMP_CPU_PLACEMENT", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_ENGINE_ISOLATE")) {
        if (*v == "1" || *v == "on") {
            s.engine_isolate = true;
        } else if (*v != "0" && *v != "off") {
            reject("INFERNIX_TMP_ENGINE_ISOLATE", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_PROMPT_SEED")) {
        const auto [credit, budget] = split_comma(*v);
        double c                    = 0.0;
        long b                      = 0;
        if (parse_double(credit, c) && c >= 0.0 && (budget.empty() || parse_long(budget, 0, 64, b))) {
            s.prompt_credit = c;
            s.prompt_budget = static_cast<std::uint32_t>(b);
        } else {
            reject("INFERNIX_TMP_PROMPT_SEED", *v);
        }
    }
    if (const auto v = value("INFERNIX_TMP_STAGING_SLOTS")) {
        long slots = 0;
        if (parse_long(*v, 64, 512, slots)) {
            s.staging_slots = static_cast<std::int32_t>(slots);
        } else {
            reject("INFERNIX_TMP_STAGING_SLOTS", *v);
        }
    }
    return s;
}

} // namespace

std::optional<std::string> value(std::string_view name) { return snapshot()[index_of(name)]; }

bool flag(std::string_view name) {
    const auto v = value(name);
    if (!v || *v == "0") { return false; }
    if (*v == "1") { return true; }
    malformed(name, *v, "0 or 1");
}

std::int64_t integer(std::string_view name, std::int64_t fallback, std::int64_t lo, std::int64_t hi) {
    const auto v = value(name);
    if (!v) { return fallback; }
    char* end = nullptr;
    errno     = 0;
    const long long out = std::strtoll(v->c_str(), &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || out < lo || out > hi) {
        malformed(name, *v, ("an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]").c_str());
    }
    return out;
}

double number(std::string_view name, double fallback, double lo, double hi) {
    const auto v = value(name);
    if (!v) { return fallback; }
    double out = 0.0;
    if (!parse_double(*v, out) || !(out >= lo && out <= hi)) {
        malformed(name, *v, ("a number in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]").c_str());
    }
    return out;
}

bool A8Subset::covers(A8SubsetClass kind, std::int32_t layer) const noexcept {
    if (!classes[static_cast<std::size_t>(kind)]) { return false; }
    if (kind == A8SubsetClass::Head || layers.empty()) { return true; }
    for (const auto& [first, last] : layers) {
        if (layer >= first && layer <= last) { return true; }
    }
    return false;
}

bool q56_fp32_scale() {
    static const bool on = flag("INFERNIX_TMP_Q56_FP32_SCALE");
    return on;
}

GwShape gw_shape() {
    static const GwShape shape = read_gw_shape();
    return shape;
}

const A8Subset& a8_subset() {
    static const A8Subset subset = read_a8_subset();
    return subset;
}

const OffloadSwitches& offload() {
    static const OffloadSwitches switches = read_offload();
    return switches;
}

std::string describe(Scope scope) {
    const auto& values = snapshot();
    // A malformed arm fails here, at startup, rather than at its first use.
    if ((static_cast<unsigned>(scope) & static_cast<unsigned>(Scope::Qwen3_5)) != 0) {
        (void)a8_subset();
        (void)gw_shape();
    }
    std::string out;
    for (std::size_t i = 0; i < kCount; ++i) {
        if ((static_cast<unsigned>(kSwitches[i].scope) & static_cast<unsigned>(scope)) == 0) { continue; }
        if (kSwitches[i].kind == Kind::Flag) { (void)flag(kSwitches[i].name); }
        if (!out.empty()) { out += " | "; }
        out += std::string(kSwitches[i].name) + "=" + (values[i] ? *values[i] : std::string("unset"));
    }
    if (const OffloadSwitches& offloaded = offload(); !offloaded.errors.empty()) {
        out += "; ignored (invalid):";
        for (const auto& error : offloaded.errors) { out += " " + error; }
    }
    return out;
}

} // namespace infernix::temporary_switches
