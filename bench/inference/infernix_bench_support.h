#pragma once

// Host-only support for the product throughput benchmark. The benchmark itself drives only the
// public infernix::Engine API; this file owns its CLI, matrix, statistics, and report schema.

#include "infernix/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::bench {

inline constexpr int kSchemaVersion                   = 25;
inline constexpr std::string_view kArtifactType       = "infernix_bench_report";
inline constexpr std::string_view kDefaultCorpusPath  = "bench/fixtures/bench_corpus.ids";
inline constexpr int kDecodeSeedTokens                = 1;
inline constexpr int kDefaultNPrompt                  = 512;
inline constexpr int kDefaultNGen                     = 128;
inline constexpr int kDefaultRepetitions              = 5;
inline constexpr int kDefaultWarmup                   = 1;
inline constexpr std::uint32_t kDefaultPrefillChunk   = 0; // the model's default
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

enum class TestKind { Prefill, Decode, PrefillDecode };

// A pp test requests one output token because Engine::generate() deliberately performs no model
// work for a zero-token request. A tg or pp+tg test requests G+1 tokens: the first token belongs to
// Engine's begin/prefill round and the following G tokens are timed by GenerationTimings::decode.
struct BenchTest {
    TestKind kind = TestKind::Prefill;
    int n_prompt  = 0;
    int n_gen     = 0;
    std::string label;

    [[nodiscard]] bool has_prefill() const noexcept {
        return kind == TestKind::Prefill || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] bool has_decode() const noexcept {
        return kind == TestKind::Decode || kind == TestKind::PrefillDecode;
    }

    [[nodiscard]] std::uint32_t requested_output_tokens() const;
    [[nodiscard]] std::uint32_t required_context(const SpeculativeOptions& speculative) const;
};

enum class OutputFormat { Table, Json, Csv };

struct BenchOptions {
    std::string artifact_path;
    std::string ngram_volume_path;
    infernix::NgramResidency ngram_residency = infernix::NgramResidency::Auto;
    std::string expert_stripe_path;
    std::uint64_t ram_headroom_bytes = infernix::kDefaultRamHeadroomBytes;
    std::optional<std::uint64_t> expert_ram_bytes; // --expert-ram-mib; empty: the RAM ledger's share
    std::optional<std::uint64_t> expert_pin_bytes; // --expert-pin-mib; empty: what the OS lets it pin
    std::optional<std::size_t> vram_headroom_bytes; // empty: the model's automatic headroom
    bool vram_past_budget = false;
    std::uint32_t expert_tail_skip = 0; // EngineOptions::expert_tail_skip
    bool expert_a4_activations     = false; // EngineOptions::expert_a4_activations
    bool embedding_host            = false; // EngineOptions::embedding_host
    std::string corpus_path{kDefaultCorpusPath};
    // Prompts start this many tokens into the corpus (wrapping), so A/B blocks can vary them.
    std::size_t corpus_offset = 0;
    // --prompt-ids: one exact prompt for every test (comma- or space-separated ids, such as a chat-templated
    // fixture); each -n N then prefills it and decodes N tokens. Empty: corpus slices.
    std::string prompt_ids_path;
    // Sampling: the bench is greedy unless --temperature is given; measured repetition r draws with seed + r
    // (warm-ups with seed + 1000 + w). Unset fields keep the model's defaults.
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<std::int32_t> top_k;
    std::optional<float> min_p;
    std::uint64_t seed = 1;
    // --stop-at-eos: the model's stop tokens end a request, which may then finish before its limit.
    bool stop_at_eos = false;
    // --route-trace: EngineOptions::route_trace_path (Flash-Next / offloaded experts).
    std::string route_trace_path;
    std::string constraint_file;
    std::optional<OutputConstraint> constraint;
    std::uint32_t concurrency = 1;
    bool mixed_constraints    = false;
    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<std::pair<int, int>> prompt_gen;
    int repetitions = kDefaultRepetitions;
    int warmup      = kDefaultWarmup;
    float rope_yarn_factor = 1.0F;
    std::optional<std::uint32_t> max_context;
    bool original_int8_prefill_kernel = false;
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    bool original_nvfp4_prefill_kernel = false;
    bool fast_paths                    = true;
    std::vector<FastPath> fast_path_candidates;
    std::uint32_t prefill_chunk       = kDefaultPrefillChunk;
    KvCacheStorage kv_cache     = KvCacheStorage::BFloat16;
    LogitsDtype logits_dtype    = LogitsDtype::FP32;
    SpeculativeOptions speculative;
    int device            = 0;
    bool use_cuda_graph   = true;
    bool profile_measured = false;
    OutputFormat output   = OutputFormat::Table;
    std::string output_file;
    bool help_requested = false;
};

struct RepTiming {
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::uint32_t generated_output_tokens = 0;
    std::optional<ConstraintObservation> constraint;
    double constraint_draft_wait_exposed_seconds = 0.0;
};

struct TestResult {
    BenchTest test;
    // Request samples are repetition-major. Wall times cover all requests in each repetition.
    std::uint32_t concurrency = 1;
    std::vector<RepTiming> reps;
    std::vector<double> repetition_wall_seconds;
    std::size_t workspace_peak_bytes           = 0;
    std::size_t workspace_allocator_peak_bytes = 0;
    // The measured repetitions' Engine execution units (RuntimeStats::execution over them: counts
    // and totals as differences, maxima restarted after the warm-ups).
    RuntimeExecutionLatency execution;
};

// RuntimeStats::execution from `before` to `after`, the maxima `after`'s.
[[nodiscard]] RuntimeExecutionLatency execution_delta(const RuntimeExecutionLatency& before,
                                                      const RuntimeExecutionLatency& after);

struct Stats {
    double mean   = 0.0;
    double stddev = 0.0;
    int count     = 0;
};

// The machine around the GPU, so that reports from different machines compare: the PCIe link read
// right after the measured runs (an idle GPU lowers it), the pinned host-to-device rate measured
// once the model has loaded (none under --profile-measured, which keeps the profiled launches the
// run's own), and a RAM read rate measured before it loads. 0: unknown or not measured.
struct BenchMachine {
    std::uint32_t pcie_generation     = 0;
    std::uint32_t pcie_width          = 0;
    std::uint32_t pcie_max_generation = 0;
    std::uint32_t pcie_max_width      = 0;
    double h2d_bytes_per_second       = 0.0; // kH2dProbeBytes pinned copies (core/link_probe)
    double ram_read_bytes_per_second =
        0.0; // every hardware thread reading its share of kRamProbeBytes
    std::uint32_t ram_read_threads = 0;
};

inline constexpr std::size_t kH2dProbeBytes = 64ULL << 20;
inline constexpr std::size_t kRamProbeBytes = 512ULL << 20;

// The best of three passes in which `threads` threads each sum their share of a `bytes` pageable
// buffer (written first); 0 when either is 0.
[[nodiscard]] double measure_ram_read_bytes_per_second(std::size_t bytes, std::uint32_t threads);

struct BenchEnvironment {
    std::string gpu_name;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
    int device_id = 0;
    BenchMachine machine;

    std::string artifact_path;
    std::uint64_t artifact_file_size_bytes = 0;
    LoadSummary load;
    MemorySummary memory;

    float rope_yarn_factor      = 1.0F;
    std::uint32_t max_context   = 0;
    std::uint32_t prefill_chunk       = kDefaultPrefillChunk;
    bool original_int8_prefill_kernel = false;
    std::uint32_t prefill_split_workspace_mib = kDefaultPrefillSplitWorkspaceMiB;
    bool original_nvfp4_prefill_kernel = false;
    bool fast_paths                    = true;
    std::uint32_t expert_tail_skip     = 0; // approximate: changes the output when set
    bool expert_a4_activations         = false; // FP4-Local W4A4 experts (changes the output)
    // Qwen3.8-Flash-Next placement (they move its throughput): empty paths and budgets are the defaults.
    std::string ngram_volume_path;
    NgramResidency ngram_residency = NgramResidency::Auto;
    std::string expert_stripe_path;
    std::optional<std::uint64_t> expert_ram_bytes;
    std::optional<std::uint64_t> expert_pin_bytes;
    KvCacheStorage kv_cache           = KvCacheStorage::BFloat16;
    LogitsDtype logits_dtype          = LogitsDtype::FP32;
    SpeculativeOptions speculative;
    bool use_cuda_graph                            = true;
    bool decode_graph_primed                       = false;
    std::uint32_t decode_graph_prime_output_tokens = 0;
    int repetitions                                = 0;
    int warmup                                     = 0;
    std::string corpus_path;
    std::size_t corpus_tokens = 0;
    std::size_t corpus_offset = 0;
    std::string prompt_ids_path; // the exact prompt (BenchOptions::prompt_ids_path)
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<std::int32_t> top_k;
    std::optional<float> min_p;
    std::uint64_t seed       = 1;
    bool stop_at_eos         = false;
    std::string route_trace_path;
    std::uint32_t concurrency = 1;
    std::string constraint_file;
    std::optional<OutputConstraint> constraint;
    bool mixed_constraints = false;
};

BenchOptions parse_args(int argc, char** argv);
std::string usage_text(std::string_view program);

std::vector<BenchTest> expand_tests(const BenchOptions& options);
std::uint32_t resolve_max_context(const std::vector<BenchTest>& tests,
                                  std::optional<std::uint32_t> override_max_context,
                                  const SpeculativeOptions& speculative, bool use_cuda_graph);
void validate_prompt_lengths(const std::vector<BenchTest>& tests, std::size_t corpus_tokens);

std::vector<TokenId> load_corpus_ids(const std::string& path);
// --prompt-ids: comma- or whitespace-separated ids.
std::vector<TokenId> load_prompt_ids(const std::string& path);
std::vector<TokenId> prompt_slice(const std::vector<TokenId>& corpus, int n_prompt);
std::string decode_path_name(bool use_cuda_graph, const SpeculativeOptions& speculative);
std::uint32_t decode_graph_prime_output_tokens(const SpeculativeOptions& speculative);
std::uint32_t decode_graph_prime_required_context(const SpeculativeOptions& speculative);

Stats compute_stats(const std::vector<double>& values);
std::vector<double> prefill_tok_s_series(const TestResult& result);
std::vector<double> decode_output_tok_s_series(const TestResult& result);
std::vector<double> decode_engine_tok_s_series(const TestResult& result);
std::vector<double> output_tok_s_series(const TestResult& result);
std::vector<double> prepare_time_series(const TestResult& result);
std::vector<double> prefill_time_series(const TestResult& result);
std::vector<double> decode_time_series(const TestResult& result);
std::vector<double> total_time_series(const TestResult& result);

std::string format_table(const BenchEnvironment& env, const std::vector<TestResult>& results);
std::string format_json(const BenchEnvironment& env, const std::string& command,
                        const std::vector<TestResult>& results);
std::string format_csv(const BenchEnvironment& env, const std::vector<TestResult>& results);

std::string json_escape(std::string_view value);
std::string kv_cache_name(KvCacheStorage storage);
std::string proposal_head_name(ProposalHead head);
std::uint64_t file_size_or_zero(const std::string& path);

} // namespace infernix::bench
