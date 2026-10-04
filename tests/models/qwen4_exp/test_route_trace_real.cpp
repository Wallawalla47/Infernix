// Engine-level route trace on the real artifact (memory track step R0, design §19.3.7): runs a
// prompt list greedily through the public Engine with the internal route trace on, requires the
// token ids to equal a run with the trace off (the hook is bit-neutral), and checks the trace
// against the requests. The trace feeds tools/expert_cache_replay/host_tier.py; its graph records
// give the device memory of each CUDA graph executable (measurement RM0d). Skips (77) unless
// NINFER_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   NINFER_QWEN4_ARTIFACT=out.ninfer [NINFER_QWEN4_NGRAM=out.ninfer.ngram]
//   ninfer_qwen4_exp_route_trace_real_test PROMPTS.json TRACE.bin [--max-context N] [--mtp K]
//       [--ngram N] [--concurrency C] [--single-pass] [--ids-out FILE] [--ids-expect FILE]
//
// PROMPTS.json is a list of {"name", "max_tokens", "thinking", "messages": [{"role", "content"}]}.
// By default the prompts run twice (trace off, then on) and the ids must be identical. With
// --ids-expect they run once with the trace on and must equal the ids in FILE. With --single-pass
// they run once with the trace on and only the trace is checked (graph measurements, captures
// whose reference was already established). --ids-out writes "name: id id ..." lines.
// At C = 1 the prompts run one after another; at C > 1 they are submitted together in prompt
// order (FIFO), so up to C run as one batch; the trace-off/on comparison then also assumes that
// a request's ids do not depend on the rows it shares a round with.

#include "models/qwen4_exp/program/route_trace.h"
#include "ninfer/engine.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct PromptSpec {
    std::string name;
    std::uint32_t max_tokens = 0;
    bool thinking            = false;
    std::vector<std::pair<std::string, std::string>> messages; // (role, content)
};

struct Outcome {
    std::uint32_t prompt_tokens = 0;
    std::vector<ninfer::TokenId> ids;
    double prefill_seconds = 0.0;
    double decode_seconds  = 0.0;
};

struct RunConfig {
    std::uint32_t max_context = 16384, mtp = 0, ngram = 0, concurrency = 1;
};

std::size_t free_vram() {
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { return 0; }
    return free_bytes;
}

std::vector<PromptSpec> read_prompts(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::invalid_argument("cannot read " + path.string()); }
    const auto root = nlohmann::json::parse(in);
    std::vector<PromptSpec> out;
    for (const auto& item : root) {
        PromptSpec spec;
        spec.name       = item.at("name").get<std::string>();
        spec.max_tokens = item.at("max_tokens").get<std::uint32_t>();
        spec.thinking   = item.value("thinking", false);
        for (const auto& message : item.at("messages")) {
            spec.messages.emplace_back(message.at("role").get<std::string>(),
                                       message.at("content").get<std::string>());
        }
        out.push_back(std::move(spec));
    }
    if (out.empty()) { throw std::invalid_argument("no prompts in " + path.string()); }
    return out;
}

ninfer::ChatRole role_of(const std::string& role) {
    if (role == "system") { return ninfer::ChatRole::System; }
    if (role == "assistant") { return ninfer::ChatRole::Assistant; }
    if (role == "user") { return ninfer::ChatRole::User; }
    throw std::invalid_argument("unsupported role " + role);
}

ninfer::EngineOptions engine_options(const char* artifact, const char* ngram_volume, const RunConfig& config,
                                     std::size_t prompts) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram_volume != nullptr) { options.ngram_volume_path = ngram_volume; }
    options.max_context     = config.max_context;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(config.max_context);
    options.prefill_chunk   = 1024;
    options.kv_cache        = ninfer::KvCacheStorage::Int8Group64;
    options.max_concurrency = config.concurrency;
    // Every prompt is queued at once at C > 1: none may be refused or time out while it waits.
    options.max_pending_requests = std::max<std::uint32_t>(options.max_pending_requests,
                                                           static_cast<std::uint32_t>(prompts));
    options.pending_timeout_ms   = 3'600'000;
    if (config.mtp > 0) {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = config.mtp;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    options.speculative.ngram_draft_tokens = config.ngram;
    return options;
}

ninfer::PromptInput prompt_input(const PromptSpec& spec) {
    ninfer::PromptInput input;
    for (const auto& [role, content] : spec.messages) {
        ninfer::ChatMessage message;
        message.role = role_of(role);
        message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = content});
        input.messages.push_back(std::move(message));
    }
    input.options.enable_thinking = spec.thinking;
    return input;
}

ninfer::RequestOptions request_options(const PromptSpec& spec) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = spec.max_tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

Outcome outcome_of(const ninfer::GenerationResult& result) {
    Outcome o;
    o.prompt_tokens   = result.prompt.prompt_tokens;
    o.ids             = result.generated_token_ids;
    o.prefill_seconds = result.timings.prefill_seconds;
    o.decode_seconds  = result.timings.decode_seconds;
    return o;
}

std::vector<Outcome> run_pass(const ninfer::EngineOptions& options, std::uint32_t concurrency,
                              const std::vector<PromptSpec>& prompts, const std::filesystem::path& trace,
                              const char* label) {
    ninfer::models::qwen4_exp::testing::set_route_trace(trace);
    std::vector<Outcome> out;
    {
        ninfer::Engine engine(options);
        ninfer::models::qwen4_exp::testing::set_route_trace({});
        const std::size_t ready = free_vram();
        if (concurrency == 1) {
            for (const auto& spec : prompts) {
                out.push_back(outcome_of(engine.generate(engine.prepare(prompt_input(spec)), request_options(spec))));
            }
        } else {
            std::vector<ninfer::GenerationHandle> handles;
            for (const auto& spec : prompts) {
                handles.push_back(engine.submit(engine.prepare(prompt_input(spec)), request_options(spec)));
            }
            for (auto& handle : handles) { out.push_back(outcome_of(handle.wait())); }
        }
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Outcome& o = out[i];
            std::printf("[%s] %-24s prompt %6u generated %5zu  prefill %6.2f s  decode %6.2f s (%.1f tok/s)\n", label,
                        prompts[i].name.c_str(), o.prompt_tokens, o.ids.size(), o.prefill_seconds, o.decode_seconds,
                        o.decode_seconds > 0 && !o.ids.empty() ? static_cast<double>(o.ids.size() - 1) / o.decode_seconds
                                                               : 0.0);
        }
        const std::size_t done = free_vram();
        std::printf("[%s] VRAM free after construction %zu MiB, after the requests %zu MiB (lazy graphs and "
                    "modules %+lld MiB)\n",
                    label, ready >> 20, done >> 20,
                    (static_cast<long long>(ready) - static_cast<long long>(done)) >> 20);
    }
    std::fflush(stdout);
    return out;
}

std::map<std::string, std::vector<ninfer::TokenId>> read_ids(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) { throw std::invalid_argument("cannot read " + path.string()); }
    std::map<std::string, std::vector<ninfer::TokenId>> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) { continue; }
        std::stringstream stream(line.substr(colon + 1));
        std::vector<ninfer::TokenId> ids;
        ninfer::TokenId id = 0;
        while (stream >> id) { ids.push_back(id); }
        out[line.substr(0, colon)] = std::move(ids);
    }
    return out;
}

const char* family_name(std::uint32_t family) {
    static const char* const names[] = {"decode", "verify", "mtp draft", "mtp catch-up"};
    return family < 4 ? names[family] : "?";
}

// Re-reads the trace and checks it against the requests. A prefill chunk at position 0 starts the
// next request (FIFO order) on its lane; that request's chunks are contiguous over its prompt, and
// its decode, verification and forced-token rows account for its generated tokens. Graph records
// are summarized (RM0d): each graph slot must be instantiated at most once.
int check_trace(const std::filesystem::path& path, const std::vector<PromptSpec>& prompts,
                const std::vector<Outcome>& outcomes, bool speculative) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "trace file missing: " << path << '\n';
        return 1;
    }
    std::uint32_t header[16]{};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || std::memcmp(header, "NRTRACE1", 8) != 0 || header[2] != 1) {
        std::cerr << "trace header is invalid\n";
        return 1;
    }
    const std::uint32_t layers = header[3], experts = header[4], top_k = header[5], lanes = header[8];
    std::printf("trace: %u layers, %u experts, top-%u, %u frames, %u max columns, %u lanes, max width %u, MTP %u, "
                "n-gram %u, prefill chunk %u\n",
                layers, experts, top_k, header[6], header[7], lanes, header[9], header[10], header[11], header[12]);
    if (lanes == 0 || lanes > 8) {
        std::cerr << "trace header lane count is invalid\n";
        return 1;
    }
    struct Request {
        std::uint64_t prefill_tokens = 0, prefill_rounds = 0, decode_rounds = 0, verify_rounds = 0;
        std::uint64_t live_tokens = 0, last_live = 0, forced_tokens = 0;
    };
    std::vector<Request> requests;
    std::array<std::int64_t, 8> active{};
    active.fill(-1);
    struct Graph {
        std::uint32_t family = 0, batch = 0, width = 0;
        long long bytes      = 0;
    };
    std::vector<Graph> graphs;
    std::set<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>> graph_slots;
    std::uint64_t records = 0, bytes = sizeof(header);
    std::vector<std::uint8_t> live;
    std::vector<std::int32_t> routes;
    const auto fail = [&](const std::string& what) {
        std::cerr << "trace record " << records << ": " << what << '\n';
        return 1;
    };
    while (true) {
        std::uint32_t head[8];
        in.read(reinterpret_cast<char*>(head), sizeof(head));
        if (in.gcount() == 0) { break; }
        if (!in) { return fail("truncated"); }
        const std::uint32_t kind = head[0];
        if (kind == 4) {
            const auto before = static_cast<long long>(head[4] | (static_cast<std::uint64_t>(head[5]) << 32));
            const auto after  = static_cast<long long>(head[6] | (static_cast<std::uint64_t>(head[7]) << 32));
            if (head[1] > 3 || head[2] == 0 || head[2] > lanes || head[3] == 0) { return fail("malformed graph"); }
            if (!graph_slots.insert({head[1], head[2], head[3]}).second) {
                return fail("a graph slot was instantiated twice");
            }
            graphs.push_back({head[1], head[2], head[3], before - after});
            ++records;
            bytes += sizeof(head);
            continue;
        }
        const std::uint32_t columns = head[1], rows = head[2], width = head[3], tokens = head[4], position = head[6];
        if (kind > 3 || columns == 0 || rows == 0 || rows > lanes || rows * width != columns) {
            return fail("malformed round");
        }
        std::vector<std::int32_t> row_lanes(rows);
        in.read(reinterpret_cast<char*>(row_lanes.data()), 4LL * rows);
        live.resize((columns + 3) / 4 * 4);
        in.read(reinterpret_cast<char*>(live.data()), static_cast<std::streamsize>(live.size()));
        routes.resize(static_cast<std::size_t>(layers) * top_k * columns);
        in.read(reinterpret_cast<char*>(routes.data()), static_cast<std::streamsize>(routes.size() * 4));
        if (!in) { return fail("truncated"); }
        for (std::size_t i = 0; i < routes.size(); ++i) {
            if (routes[i] < 0 || static_cast<std::uint32_t>(routes[i]) >= experts) {
                return fail("routes expert " + std::to_string(routes[i]));
            }
            const std::size_t first = i - i % top_k;
            for (std::size_t j = first; j < i; ++j) {
                if (routes[j] == routes[i]) { return fail("routes one expert twice in a column"); }
            }
        }
        for (const auto lane : row_lanes) {
            if (lane < 0 || static_cast<std::uint32_t>(lane) >= lanes) { return fail("lane out of range"); }
        }
        if (kind == 0 && position == 0) {
            active[static_cast<std::size_t>(row_lanes[0])] = static_cast<std::int64_t>(requests.size());
            requests.emplace_back();
        }
        std::uint32_t longest = 0;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const std::int64_t index = active[static_cast<std::size_t>(row_lanes[r])];
            if (index < 0) { return fail("round on a lane without a request"); }
            Request& q = requests[static_cast<std::size_t>(index)];
            std::uint64_t row_live = 0;
            for (std::uint32_t c = r * width; c < (r + 1) * width; ++c) { row_live += live[c]; }
            longest = std::max<std::uint32_t>(longest, static_cast<std::uint32_t>(row_live));
            if (kind == 0) {
                if (position != q.prefill_tokens || q.decode_rounds + q.verify_rounds != 0 || row_live != columns) {
                    return fail("prefill chunks are not contiguous");
                }
                q.prefill_tokens += columns;
                ++q.prefill_rounds;
            } else if (kind == 1) {
                q.forced_tokens += tokens;
            } else if (kind == 2) {
                if (width != 1 || row_live != 1 || tokens != 1) { return fail("decode row is not one live column"); }
                ++q.decode_rounds;
                q.live_tokens += 1;
                q.last_live = 1;
            } else {
                if (row_live == 0) { return fail("verification row without a live column"); }
                ++q.verify_rounds;
                q.live_tokens += row_live;
                q.last_live = row_live;
            }
        }
        if (kind == 3 && longest != tokens) { return fail("verification tokens are not the longest row's"); }
        ++records;
        bytes += sizeof(head) + 4ULL * rows + live.size() + routes.size() * 4;
    }
    int failures = 0;
    if (requests.size() != outcomes.size()) {
        std::cerr << "trace holds " << requests.size() << " requests, expected " << outcomes.size() << '\n';
        ++failures;
    }
    for (std::size_t i = 0; i < std::min(requests.size(), outcomes.size()); ++i) {
        const Request& r      = requests[i];
        const std::uint64_t g = outcomes[i].ids.size();
        // The first output token comes from the prefill; every further token from a round or a
        // forced-token call. A verification round may license more tokens than a stopping request
        // commits.
        const std::uint64_t committed = r.live_tokens + r.forced_tokens + 1;
        const bool rounds_ok          = speculative ? (committed >= g && committed - r.last_live < g)
                                                    : (committed == g && r.verify_rounds == 0);
        const bool ok = r.prefill_tokens == outcomes[i].prompt_tokens && (g <= 1 || rounds_ok);
        std::printf("trace %-24s prefill %5llu tokens in %llu chunks; %llu decode + %llu verify rounds, "
                    "%llu live + %llu forced tokens for %llu generated%s\n",
                    prompts[i].name.c_str(), static_cast<unsigned long long>(r.prefill_tokens),
                    static_cast<unsigned long long>(r.prefill_rounds),
                    static_cast<unsigned long long>(r.decode_rounds),
                    static_cast<unsigned long long>(r.verify_rounds),
                    static_cast<unsigned long long>(r.live_tokens), static_cast<unsigned long long>(r.forced_tokens),
                    static_cast<unsigned long long>(g), ok ? "" : "  MISMATCH");
        failures += ok ? 0 : 1;
    }
    std::printf("trace: %llu records, %.1f MiB\n", static_cast<unsigned long long>(records),
                static_cast<double>(bytes) / (1 << 20));

    // RM0d: device memory per graph executable (device free before its capture minus after its
    // instantiation and first launch).
    std::sort(graphs.begin(), graphs.end(), [](const Graph& a, const Graph& b) {
        return std::tie(a.family, a.batch, a.width) < std::tie(b.family, b.batch, b.width);
    });
    long long total = 0, largest = 0;
    for (const auto& g : graphs) {
        std::printf("graph %-12s batch %u width %2u: %+8.2f MiB\n", family_name(g.family), g.batch, g.width,
                    static_cast<double>(g.bytes) / (1 << 20));
        total += g.bytes;
        largest = std::max(largest, g.bytes);
    }
    for (std::uint32_t family = 0; family < 4; ++family) {
        long long sum = 0, count = 0;
        for (const auto& g : graphs) {
            if (g.family == family) {
                sum += g.bytes;
                ++count;
            }
        }
        if (count > 0) {
            std::printf("graphs %-12s %3lld executables, %8.2f MiB, %.2f MiB each\n", family_name(family), count,
                        static_cast<double>(sum) / (1 << 20), static_cast<double>(sum) / (1 << 20) / count);
        }
    }
    std::printf("graphs total: %zu executables, %.2f MiB, %.2f MiB each, largest %.2f MiB\n", graphs.size(),
                static_cast<double>(total) / (1 << 20),
                graphs.empty() ? 0.0 : static_cast<double>(total) / (1 << 20) / static_cast<double>(graphs.size()),
                static_cast<double>(largest) / (1 << 20));
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const char* artifact = std::getenv("NINFER_QWEN4_ARTIFACT");
        if (artifact == nullptr || argc < 3) {
            std::cerr << "skipped: set NINFER_QWEN4_ARTIFACT and pass PROMPTS.json TRACE.bin\n";
            return 77;
        }
        const char* ngram_volume = std::getenv("NINFER_QWEN4_NGRAM");
        const std::filesystem::path prompts_path = argv[1], trace = argv[2];
        RunConfig config;
        bool single_pass = false;
        std::filesystem::path ids_out, ids_expect;
        for (int i = 3; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value      = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--max-context") {
                config.max_context = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--mtp") {
                config.mtp = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--ngram") {
                config.ngram = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--concurrency") {
                config.concurrency = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--single-pass") {
                single_pass = true;
            } else if (arg == "--ids-out") {
                ids_out = value();
            } else if (arg == "--ids-expect") {
                ids_expect = value();
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (config.concurrency == 0 || config.concurrency > 8) {
            throw std::invalid_argument("--concurrency must be in [1,8]");
        }
        if (single_pass && !ids_expect.empty()) {
            throw std::invalid_argument("--single-pass and --ids-expect exclude each other");
        }
        const auto prompts = read_prompts(prompts_path);
        const auto options = engine_options(artifact, ngram_volume, config, prompts.size());
        const auto start   = std::chrono::steady_clock::now();

        std::vector<std::vector<ninfer::TokenId>> reference;
        if (!ids_expect.empty()) {
            const auto expected = read_ids(ids_expect);
            for (const auto& spec : prompts) {
                const auto it = expected.find(spec.name);
                if (it == expected.end()) { throw std::invalid_argument("no expected ids for " + spec.name); }
                reference.push_back(it->second);
            }
        } else if (!single_pass) {
            for (auto& o : run_pass(options, config.concurrency, prompts, {}, "trace off")) {
                reference.push_back(std::move(o.ids));
            }
        }
        const auto traced = run_pass(options, config.concurrency, prompts, trace, "trace on");

        int failures = 0;
        if (!reference.empty()) {
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                if (traced[i].ids != reference[i]) {
                    std::size_t at = 0;
                    while (at < traced[i].ids.size() && at < reference[i].size() &&
                           traced[i].ids[at] == reference[i][at]) {
                        ++at;
                    }
                    std::printf("ids differ: %s at token %zu (%zu vs %zu tokens)\n", prompts[i].name.c_str(), at,
                                traced[i].ids.size(), reference[i].size());
                    ++failures;
                }
            }
            std::printf("ids %s for %zu prompts (%s)\n", failures == 0 ? "identical" : "DIFFER", prompts.size(),
                        ids_expect.empty() ? "trace off vs on" : "against --ids-expect");
        } else {
            std::printf("ids not compared (--single-pass)\n");
        }
        if (!ids_out.empty()) {
            std::ofstream out(ids_out);
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                out << prompts[i].name << ':';
                for (const auto id : traced[i].ids) { out << ' ' << id; }
                out << '\n';
            }
        }
        failures += check_trace(trace, prompts, traced, config.mtp > 0 || config.ngram > 0);
        std::printf("total %.1f s; %s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                    failures == 0 ? "PASS" : "FAIL");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
