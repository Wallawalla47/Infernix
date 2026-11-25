// Inkling 2's launch options: every option Inkling 2 has nothing to apply to is refused by its flag
// (runtime::normalize_inkling2_options before anything loads, serve::refuse_unsupported_options once the family is
// known) instead of being ignored, and the configurations its callers start with still start: infernix-serve's and the
// launcher's defaults, tools/ablate's engines, infernix_bench's and the real tests' Engine options. A scoring start
// with generation options reports what it leaves unused, as the other models do. CPU only: nothing reads an artifact.

#include "runtime/engine/inkling2_instance.h"
#include "serve/generation_service.h"
#include "serve/serve_options.h"

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix;

constexpr const char* kInkling = "Inkling2LoopForCausalLM";

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

serve::ServeOptions parse(std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), {"infernix-serve", "model.infernix"});
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

std::string joined(const std::vector<std::string>& arguments) {
    std::string out;
    for (const std::string& argument : arguments) { out += (out.empty() ? "" : " ") + argument; }
    return out.empty() ? "(no options)" : out;
}

struct Start {
    std::string error;                // the first refusal, or empty
    std::vector<std::string> reports; // the Engine's Info diagnostics
};

// An infernix-serve start's option checks, in its order: the parser, the Engine's (twice: the options it resolves
// start the same start), and the server's once the model is known.
Start start(const std::vector<std::string>& arguments, bool bundled_drafter = true) {
    Start out;
    try {
        const serve::ServeOptions options = parse(arguments);
        EngineOptions engine              = serve::make_engine_options(options, {}, {});
        engine.diagnostic_observer.callback = [&out](const Diagnostic& diagnostic) {
            out.reports.emplace_back(diagnostic.message);
        };
        const EngineOptions resolved = runtime::normalize_inkling2_options(std::move(engine), bundled_drafter);
        (void)runtime::normalize_inkling2_options(resolved, bundled_drafter);
        serve::refuse_unsupported_options(options, kInkling);
    } catch (const std::exception& error) { out.error = error.what(); }
    return out;
}

void refused(const std::vector<std::string>& arguments, const std::string& flag, bool bundled_drafter = true) {
    const Start result = start(arguments, bundled_drafter);
    check(result.error.find(flag) != std::string::npos && result.error.starts_with("Inkling 2: "),
          joined(arguments) + ": expected a refusal naming " + flag + ", got \"" + result.error + "\"");
}

void accepted(const std::vector<std::string>& arguments, bool bundled_drafter = true) {
    const Start result = start(arguments, bundled_drafter);
    check(result.error.empty(), joined(arguments) + ": refused: " + result.error);
}

std::string engine_error(EngineOptions options) {
    try {
        (void)runtime::normalize_inkling2_options(std::move(options), true);
    } catch (const std::exception& error) { return error.what(); }
    return {};
}

void test_refused_options() {
    // Read by the Qwen families only.
    refused({"--kv-dtype", "int8", "--use-original-int8-prefill-kernel"}, "--use-original-int8-prefill-kernel");
    refused({"--kv-dtype", "nvfp4", "--use-original-nvfp4-prefill-kernel"}, "--use-original-nvfp4-prefill-kernel");
    refused({"--prefill-split-workspace-mib", "512"}, "--prefill-split-workspace-mib");
    refused({"--fast-path", "f16-accumulate-prefill"}, "--fast-path");
    refused({"--no-fast-path", "f16-accumulate-prefill"}, "--no-fast-path");
    refused({"--context-cost-presets", "presets.json"}, "--context-cost-presets");
    refused({"--media-cache-mib", "512"}, "--media-cache-mib");
    refused({"--media-live-mib", "512"}, "--media-live-mib");
    refused({"--media-preprocess-threads", "4"}, "--media-preprocess-threads");
    refused({"--vision-max-merged", "4096"}, "--vision-max-merged");
    // Offloaded experts.
    refused({"--expert-tail-skip", "7"}, "--expert-tail-skip");
    refused({"--expert-ram-mib", "4096"}, "--expert-ram-mib");
    refused({"--expert-ram-layout", "full"}, "--expert-ram-layout");
    refused({"--expert-csf"}, "--expert-csf");
    refused({"--expert-pin-mib", "4096"}, "--expert-pin-mib");
    refused({"--expert-state", "off"}, "--expert-state");
    refused({"--expert-state", "experts.state"}, "--expert-state");
    refused({"--expert-stripe", "experts.stripe"}, "--expert-stripe");
    refused({"--route-trace", "routes.trace"}, "--route-trace");
    refused({"--ram-resize", "off"}, "--ram-resize");
    refused({"--ram-resize", "auto"}, "--ram-resize");
    // Speculation: DSpark drafts a chain from the artifact's own component.
    refused({"--draft-model", "companion.infernix"}, "--draft-model");
    refused({"--draft-tree-paths", "4"}, "--draft-tree-paths");
    refused({"--ngram-residency", "ram"}, "--ngram-residency");
    refused({"--ngram-min-match", "8"}, "--ngram-min-match");
    refused({"--ngram-session-mib", "64"}, "--ngram-session-mib");
    refused({"--lm-head-draft"}, "--lm-head-draft", false); // --spec auto finds no bundled drafter
    refused({"--lossless", "--lm-head-draft"}, "--lm-head-draft");
    // The lossless route keeps its sliding windows BF16.
    refused({"--lossless", "--kv-window-dtype", "int8"}, "--kv-window-dtype");
    refused({"--lossless", "--kv-window-dtype", "bf16p"}, "--kv-window-dtype");
    // What each start sizes.
    refused({"--no-prefix-reuse", "--vram-headroom-mib", "512"}, "--vram-headroom-mib");
    refused({"--kv-capacity", "65536", "--vram-headroom-mib", "512"}, "--vram-headroom-mib");
    refused({"--no-prefix-reuse", "--vram-past-budget"}, "--vram-past-budget");
    refused({"--no-prefix-reuse", "--ram-free-mib", "8192"}, "--ram-free-mib");
    refused({"--host-context-mib", "4096", "--ram-free-mib", "8192"}, "--ram-free-mib");
    // The server's thinking and tool options.
    refused({"--reasoning-loop", "stop"}, "--reasoning-loop");
    refused({"--preserve-thinking"}, "--preserve-thinking");
    refused({"--tolerant-tool-calls"}, "--tolerant-tool-calls");
    refused({"--tool-parameter-names", "declared"}, "--tool-parameter-names");

    // Engine fields no serve flag sets alone (the API, infernix-certify).
    EngineOptions api;
    // The windows hold bf16, bf16p or int8 only (the serve and CLI parsers refuse another name before this).
    api.kv_window_cache = KvCacheStorage::Fp8E4M3Row256;
    check(engine_error(api).find("--kv-window-dtype") != std::string::npos, "an fp8 window format is refused");
    api = {};
    api.speculative.draft_min_p = 0.1F;
    check(engine_error(api).find("--spec-min-p") != std::string::npos, "the API's draft_min_p is refused");
    api = {};
    api.speculative.ngram_tree_nodes = 8;
    check(engine_error(api).find("--ngram-tree-nodes") != std::string::npos, "the API's ngram_tree_nodes is refused");
    api = {};
    api.speculative.ngram_archive_bytes = 64ULL << 20;
    check(engine_error(api).find("--ngram-archive-mib") != std::string::npos, "the API's archive is refused");
    api = {};
    api.reference  = true;
    api.lossless   = true;
    api.fast_paths = false; // the lossless route runs none
    check(engine_error(api).find("--reference") != std::string::npos, "the design reference outside scoring is refused");
    api = {};
    api.lossless_noise_seed = 7;
    check(engine_error(api).find("noise twin") != std::string::npos, "the noise twin outside lossless scoring is refused");
    // ... and on lossless scoring too: Inkling 2's own lossless kernels have no twin.
    api.lossless   = true;
    api.fast_paths = false;
    api.purpose    = EnginePurpose::CausalScoring;
    check(engine_error(api).find("not Inkling 2's") != std::string::npos,
          "the noise twin on Inkling 2's lossless scoring is refused");
    // The prefix cache's options Inkling 2 does not have are refused on a scoring start too (it would drop them).
    api         = {};
    api.purpose = EnginePurpose::CausalScoring;
    api.context_cache.hybrid.device_snapshot_slots = 2U;
    check(engine_error(api).find("Device snapshot slots") != std::string::npos,
          "--device-snapshot-slots on a scoring start is refused");
}

void test_accepted_starts() {
    accepted({});
    // The launcher's defaults for Inkling 2 (launcher/model_defaults.cpp), with and without a bundled drafter.
    const std::vector<std::string> launcher = {"--max-context", "auto",   "--max-concurrency", "2",
                                               "--kv-dtype",    "int8",   "--spec",            "auto",
                                               "--draft-tokens", "auto",  "--pending-timeout-ms", "900000"};
    accepted(launcher);
    accepted(launcher, false);
    // tools/ablate's engines (ablatelib/engine.py: no --expert-state on Inkling 2), generation and scoring.
    std::vector<std::string> ablate = {"--host",        "127.0.0.1", "--port",            "18080",
                                       "--model-id",    "ablate",    "--max-context",     "16384",
                                       "--max-concurrency", "1",     "--spec",            "off"};
    accepted(ablate);
    ablate.emplace_back("--causal-scoring");
    accepted(ablate);
    for (const char* kv : {"bf16", "bf16p", "int8", "fp8", "nvfp4", "k8v4", "vq2", "k4v2"}) {
        accepted({"--kv-dtype", kv});
        // Every pool takes every window format.
        for (const char* window : {"auto", "bf16", "bf16p", "int8"}) {
            accepted({"--kv-dtype", kv, "--kv-window-dtype", window});
        }
    }
    accepted({"--no-prefix-reuse"});
    accepted({"--no-prefix-reuse", "--max-context", "auto", "--vram-headroom-mib", "512", "--vram-past-budget"});
    accepted({"--vram-headroom-mib", "512", "--vram-past-budget"}); // the cache's --kv-capacity auto
    accepted({"--ram-free-mib", "8192"});                           // the automatic Host tier
    accepted({"--max-context", "8192", "--max-concurrency", "2", "--kv-capacity", "32768"});
    accepted({"--max-context", "8192", "--max-concurrency", "2", "--kv-capacity", "16384"});
    // The lanes share the pool, which needs only one request's context (a resident pauses when they outgrow it), with
    // or without the prefix cache; --max-context auto stays within an explicit capacity.
    accepted({"--max-context", "8192", "--max-concurrency", "4", "--kv-capacity", "8192"});
    accepted({"--no-prefix-reuse", "--max-context", "8192", "--kv-capacity", "16384"});
    accepted({"--max-context", "auto", "--kv-capacity", "65536"});
    accepted({"--no-queue-holds"});
    {
        EngineOptions small;
        small.max_context = 8192;
        small.kv_capacity = KvCapacityPolicy::explicit_capacity(4096);
        check(engine_error(small).find("--kv-capacity") != std::string::npos,
              "a KV capacity below one request's context is refused: " + engine_error(small));
    }
    accepted({"--lm-head-draft"});
    accepted({"--vision", "--vision-max-merged", "4096"});
    accepted({"--vision-max-merged", "32768"}); // the default, given
    accepted({"--end-in-thinking", "allow"});
    accepted({"--no-thinking"});
    // The thinking controls apply to every Inkling 2 request (serve::settle_thinking), --no-thinking's included.
    accepted({"--default-thinking-budget", "1024", "--thinking-budget-message", "Time to answer."});
    accepted({"--no-thinking-answer-room"});
    accepted({"--no-thinking", "--end-in-thinking", "allow", "--default-thinking-budget", "2048"});
    accepted({"--reasoning-loop", "off", "--draft-tree-paths", "8", "--ngram-min-match", "12"}); // the defaults, given

    // A scoring start with a generation configuration: what it leaves unused is reported, not refused (the launcher's
    // and tools/ablate's scoring starts carry their generation options).
    const Start scoring = start({"--causal-scoring", "--max-context", "4096", "--max-concurrency", "2",
                                 "--ram-free-mib", "8192", "--vram-headroom-mib", "512", "--lm-head-draft"});
    bool reported = false;
    for (const std::string& line : scoring.reports) {
        reported = reported || (line.find("--ram-free-mib") != std::string::npos &&
                                line.find("--vram-headroom-mib") != std::string::npos &&
                                line.find("--lm-head-draft") != std::string::npos && line.find("unused") != std::string::npos);
    }
    check(scoring.error.empty() && reported, "a scoring start reports its unused generation options: " + scoring.error);

    // The Engine options the in-tree callers build (no serve parser).
    EngineOptions real;                                    // tests/models/inkling2/test_engine_real.cpp
    real.max_context     = 4096;
    real.max_concurrency = 4;
    real.kv_capacity     = KvCapacityPolicy::explicit_capacity(4 * 4096);
    check(engine_error(real).empty(), "the real tests' options: " + engine_error(real));
    EngineOptions bench;                                   // bench/inference/infernix_bench.cpp
    bench.max_context                       = 4100;
    bench.max_concurrency                   = 2;
    bench.kv_capacity                       = KvCapacityPolicy::explicit_capacity((2 * 4100 + 63) / 64 * 64);
    bench.context_cache.enabled             = false;
    bench.context_cache.host_capacity_bytes = 0;
    check(engine_error(bench).empty(), "infernix_bench's options: " + engine_error(bench));
    EngineOptions perplexity;                              // apps/perplexity/main.cpp
    perplexity.purpose     = EnginePurpose::CausalScoring;
    perplexity.max_context = 4096;
    check(engine_error(perplexity).empty(), "infernix-perplexity's options: " + engine_error(perplexity));
    // A Qwen model is not this check's to refuse.
    serve::refuse_unsupported_options(parse({"--preserve-thinking", "--expert-state", "off"}), "Qwen3_5ForCausalLM");
}

} // namespace

int main() {
    try {
        test_refused_options();
        test_accepted_starts();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
