// Measures a model's token-frequency ranking through the public Engine, as NInfer measured Qwen's: the model's own most
// probable token at every assistant position of a chat corpus (teacher-forced on the corpus's text), counted per stratum.
//
//   infernix_token_ranking <artifact> --corpus train.jsonl --output PREFIX --vocab V [--context N]
//
// The corpus is tools/freq_corpus/prepare_corpus.py's JSONL. Writes PREFIX.counts.i64 (int64 LE; row 0 = total[V], rows
// 1..5 = strata[V] in the order below) and PREFIX.manifest.json. It saves its counts every five minutes and resumes from
// PREFIX.state.json, so a stopped run loses at most those minutes.

#include "infernix/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using nlohmann::json;
namespace fs = std::filesystem;

constexpr std::array<std::string_view, 5> kStrata{"en_chat", "zh_chat", "other_chat", "code", "math"};

struct Options {
    fs::path artifact, corpus, output, ngram_volume;
    std::int32_t vocab   = 0;
    std::uint32_t context = 4096;
};

struct Progress {
    std::uint64_t lines = 0, used = 0, skipped = 0, truncated = 0, windows = 0, forwarded = 0, emitted = 0, hits = 0;
    std::array<std::uint64_t, kStrata.size()> strata{};
    double seconds = 0;
};

Options parse(int argc, char** argv) {
    if (argc < 2) { throw std::invalid_argument("usage: infernix_token_ranking <artifact> --corpus F --output P --vocab V [--context N]"); }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; i += 2) {
        const std::string_view key = argv[i];
        if (i + 1 >= argc) { throw std::invalid_argument(std::string(key) + " needs a value"); }
        const char* value = argv[i + 1];
        if (key == "--corpus") {
            out.corpus = value;
        } else if (key == "--output") {
            out.output = value;
        } else if (key == "--vocab") {
            out.vocab = std::stoi(value);
        } else if (key == "--context") {
            out.context = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "--ngram-volume") {
            out.ngram_volume = value; // Qwen3.8-Flash-Next
        } else {
            throw std::invalid_argument("unknown option " + std::string(key));
        }
    }
    if (out.corpus.empty() || out.output.empty() || out.vocab <= 0 || out.context < 128) {
        throw std::invalid_argument("--corpus, --output, --vocab and a --context of at least 128 are required");
    }
    return out;
}

fs::path with_suffix(const fs::path& prefix, std::string_view suffix) {
    fs::path out = prefix;
    out += suffix;
    return out;
}

int stratum_of(const std::string& name) {
    const auto it = std::find(kStrata.begin(), kStrata.end(), name);
    if (it == kStrata.end()) { throw std::runtime_error("unknown stratum " + name); }
    return static_cast<int>(it - kStrata.begin());
}

// Some chat templates write the reasoning block only on the last assistant turn (Qwen3.6's). The earlier turns are then
// history in a form the model never writes, and its top token there is mostly the end of the turn, so they are not
// counted. A template that opens every assistant turn the same way keeps them all.
void keep_turns_in_model_form(const infernix::Engine& engine, infernix::ChatTokens& chat) {
    const std::size_t n = chat.ids.size();
    const auto opens_turn = [&](std::size_t i) { return chat.generated[i] != 0 && (i == 0 || chat.generated[i - 1] == 0); };
    const auto reasons    = [&](std::size_t i) { return engine.detokenize(std::span(&chat.ids[i], 1)) == "<think>"; };
    bool any_reasoning    = false;
    for (std::size_t i = 0; i < n && !any_reasoning; ++i) { any_reasoning = opens_turn(i) && reasons(i); }
    if (!any_reasoning) { return; }
    for (std::size_t i = 0; i < n;) {
        if (!opens_turn(i)) {
            ++i;
            continue;
        }
        std::size_t end = i;
        while (end < n && chat.generated[end] != 0) { ++end; }
        if (!reasons(i)) { std::fill(chat.generated.begin() + static_cast<std::ptrdiff_t>(i),
                                     chat.generated.begin() + static_cast<std::ptrdiff_t>(end), std::uint8_t{0}); }
        i = end;
    }
}

infernix::PromptInput prompt_of(const json& record) {
    infernix::PromptInput input;
    for (const json& message : record.at("messages")) {
        infernix::ChatMessage out;
        out.role = message.at("role").get<std::string>() == "assistant" ? infernix::ChatRole::Assistant
                                                                        : infernix::ChatRole::User;
        out.parts.push_back(infernix::MessagePart{.kind  = infernix::MessagePartKind::Text,
                                                  .text  = message.at("content").get<std::string>(),
                                                  .media = {}});
        input.messages.push_back(std::move(out));
    }
    return input;
}

void replace_file(const fs::path& path, const void* data, std::size_t bytes) {
    fs::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out) { throw std::runtime_error("cannot write " + temporary.string()); }
    }
    fs::rename(temporary, path);
}

json totals(const Progress& p, const std::vector<std::int64_t>& counts, std::int32_t vocab) {
    std::uint64_t distinct = 0;
    for (std::int32_t i = 0; i < vocab; ++i) { distinct += counts[static_cast<std::size_t>(i)] != 0 ? 1U : 0U; }
    return {{"convs_seen", p.lines},
            {"convs_used", p.used},
            {"convs_skipped", p.skipped},
            {"convs_truncated", p.truncated},
            {"windows", p.windows},
            {"tokens_forwarded", p.forwarded},
            {"emit_positions", p.emitted},
            {"argmax_equals_corpus_token", p.hits},
            {"distinct_emitted_tokens", distinct},
            {"seconds", p.seconds}};
}

void save(const Options& o, const Progress& p, const std::vector<std::int64_t>& counts) {
    replace_file(with_suffix(o.output, ".counts.i64"), counts.data(), counts.size() * sizeof(std::int64_t));
    const std::string state = json{{"lines", p.lines},     {"used", p.used},       {"skipped", p.skipped},
                                   {"truncated", p.truncated}, {"windows", p.windows}, {"forwarded", p.forwarded},
                                   {"emitted", p.emitted}, {"hits", p.hits},       {"strata", p.strata},
                                   {"seconds", p.seconds}}
                                  .dump();
    replace_file(with_suffix(o.output, ".state.json"), state.data(), state.size());
}

bool resume(const Options& o, Progress& p, std::vector<std::int64_t>& counts) {
    const fs::path state_path = with_suffix(o.output, ".state.json"), counts_path = with_suffix(o.output, ".counts.i64");
    if (!fs::exists(state_path) || !fs::exists(counts_path)) { return false; }
    if (fs::file_size(counts_path) != counts.size() * sizeof(std::int64_t)) { throw std::runtime_error("the saved counts have another size"); }
    std::ifstream(counts_path, std::ios::binary).read(reinterpret_cast<char*>(counts.data()), static_cast<std::streamsize>(counts.size() * 8));
    const json s = json::parse(std::ifstream(state_path));
    p.lines = s.at("lines"), p.used = s.at("used"), p.skipped = s.at("skipped"), p.truncated = s.at("truncated");
    p.windows = s.at("windows"), p.forwarded = s.at("forwarded"), p.emitted = s.at("emitted"), p.hits = s.at("hits");
    p.strata  = s.at("strata").get<std::array<std::uint64_t, kStrata.size()>>();
    p.seconds = s.at("seconds");
    return true;
}

int run(const Options& o) {
    infernix::EngineOptions engine_options;
    engine_options.artifact_path = o.artifact;
    engine_options.purpose       = infernix::EnginePurpose::CausalScoring;
    engine_options.max_context   = o.context;
    engine_options.prefill_chunk = o.context / 128 * 128;
    if (!o.ngram_volume.empty()) { engine_options.ngram_volume_path = o.ngram_volume; }
    infernix::Engine engine(std::move(engine_options));

    const std::size_t rows = kStrata.size() + 1, vocab = static_cast<std::size_t>(o.vocab);
    std::vector<std::int64_t> counts(rows * vocab, 0);
    Progress p;
    if (resume(o, p, counts)) { std::cout << "resuming after " << p.lines << " conversations" << std::endl; }
    const std::uint64_t resume_at = p.lines;
    const double earlier          = p.seconds;

    std::ifstream corpus(o.corpus, std::ios::binary);
    if (!corpus) { throw std::runtime_error("cannot read " + o.corpus.string()); }
    const Clock::time_point started = Clock::now();
    Clock::time_point next_save = started + std::chrono::minutes(5), next_print = started + std::chrono::seconds(30);
    std::string line;
    std::uint64_t index = 0;
    while (std::getline(corpus, line)) {
        if (++index <= resume_at) { continue; }
        p.lines = index;
        const json record = json::parse(line);
        const int stratum = stratum_of(record.at("stratum").get<std::string>());
        infernix::ChatTokens chat;
        try {
            chat = engine.tokenize_chat(prompt_of(record));
        } catch (const std::exception&) {
            ++p.skipped;
            continue;
        }
        const std::size_t n = chat.ids.size();
        if (n < 2) {
            ++p.skipped;
            continue;
        }
        keep_turns_in_model_form(engine, chat);
        ++p.used;
        p.truncated += n > o.context ? 1U : 0U;
        // Each assistant position is scored once, with up to `context` tokens and at least half of them as history.
        std::size_t position = 1;
        while (position < n) {
            if (chat.generated[position] == 0) {
                ++position;
                continue;
            }
            const std::size_t begin = position <= o.context / 2 ? 0 : position - o.context / 2;
            const std::size_t end   = std::min<std::size_t>(n, begin + o.context);
            std::vector<infernix::TokenId> window(chat.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                                  chat.ids.begin() + static_cast<std::ptrdiff_t>(end));
            infernix::ScoreOptions options;
            options.argmax_only = true;
            const infernix::ScoreResult result =
                engine.score_tokens(std::move(window), static_cast<std::uint32_t>(position - begin), std::move(options));
            ++p.windows;
            p.forwarded += end - begin;
            for (std::size_t target = position; target < end; ++target) {
                if (chat.generated[target] == 0) { continue; }
                const infernix::TokenId id = result.top_ids[target - position];
                if (id < 0 || id >= o.vocab) { throw std::runtime_error("a scored id lies outside the vocabulary"); }
                ++counts[static_cast<std::size_t>(id)];
                ++counts[(static_cast<std::size_t>(stratum) + 1) * vocab + static_cast<std::size_t>(id)];
                ++p.strata[static_cast<std::size_t>(stratum)];
                ++p.emitted;
                p.hits += id == chat.ids[target] ? 1U : 0U;
            }
            position = end;
        }
        const Clock::time_point now = Clock::now();
        p.seconds = earlier + std::chrono::duration<double>(now - started).count();
        if (now >= next_print) {
            std::cout << p.lines << " conversations, " << p.emitted << " positions, "
                      << static_cast<std::uint64_t>(static_cast<double>(p.forwarded) / std::max(p.seconds, 1e-3)) << " tok/s" << std::endl;
            next_print = now + std::chrono::seconds(30);
        }
        if (now >= next_save) {
            save(o, p, counts);
            next_save = now + std::chrono::minutes(5);
        }
    }
    p.seconds = earlier + std::chrono::duration<double>(Clock::now() - started).count();
    save(o, p, counts);
    engine.stop();

    json strata = json::object();
    for (std::size_t i = 0; i < kStrata.size(); ++i) { strata[std::string(kStrata[i])] = p.strata[i]; }
    const json manifest{
        {"artifact_type", "freq_measure_ranking"},
        {"schema_version", 1},
        {"status", "complete"},
        {"mode", "teacher_forced_argmax"},
        {"serving_form", "chat-template conversations; the model's argmax at each assistant token, teacher-forced on the corpus"},
        {"corpus", o.corpus.filename().string()},
        {"weights", o.artifact.filename().string()},
        {"counts_file", with_suffix(o.output, ".counts.i64").filename().string()},
        {"counts_layout", "int64 LE; row 0 = total[vocab]; rows 1..5 = strata[vocab] in strata_order"},
        {"counts_rows", rows},
        {"vocab", o.vocab},
        {"context", o.context},
        {"strata_order", kStrata},
        {"strata_emit_counts", strata},
        {"totals", totals(p, counts, o.vocab)}};
    const std::string text = manifest.dump(1);
    replace_file(with_suffix(o.output, ".manifest.json"), text.data(), text.size());
    std::cout << "done: " << manifest.at("totals").dump() << std::endl;
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
