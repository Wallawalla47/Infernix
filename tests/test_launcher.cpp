// infernix-launcher: launch-parameter edits, serve log and --help parsing, the --help cache, the
// server's log files, artifact and n-gram volume recognition, default launch parameters, the
// router's drain-then-switch order, the advice for loads that do not fit, the event log, GitHub's
// release answer and the exit-status wording.

#include "launcher/args.h"
#include "launcher/artifact_scan.h"
#include "launcher/chats.h"
#include "launcher/event_log.h"
#include "launcher/events.h"
#include "launcher/launch_script.h"
#include "launcher/launcher.h"
#include "launcher/load_advice.h"
#include "launcher/log_buffer.h"
#include "launcher/model_defaults.h"
#include "launcher/router.h"
#include "launcher/routing.h"
#include "launcher/serve_help.h"
#include "launcher/sha256.h"
#include "launcher/supervisor.h"
#include "launcher/telemetry.h"
#include "launcher/updates.h"
#include "launcher/usage.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using namespace infernix::launcher;
using namespace std::chrono_literals;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << "FAILED: " << message << '\n';
    ++failures;
}

bool contains(const Args& args, std::initializer_list<std::string> sequence) {
    return std::search(args.begin(), args.end(), sequence.begin(), sequence.end()) != args.end();
}

void test_args() {
    Args args = {"--max-context", "8192", "--kv-dtype", "int8", "--max-context", "4096", "--preserve-thinking"};
    set_flag(args, "--max-context", "262144");
    check((args == Args{"--max-context", "262144", "--kv-dtype", "int8", "--preserve-thinking"}),
          "set_flag replaces the first occurrence in place and drops later duplicates");
    set_flag(args, "--spec", "auto");
    check(contains(args, {"--spec", "auto"}), "set_flag appends an absent flag");
    remove_flag(args, "--preserve-thinking");
    check(!has_flag(args, "--preserve-thinking"), "remove_flag removes a switch");
    remove_flag(args, "--kv-dtype");
    check(!has_flag(args, "--kv-dtype") && std::find(args.begin(), args.end(), "int8") == args.end(),
          "remove_flag removes a flag with its value");
    check(flag_value({"--draft-tokens=auto"}, "--draft-tokens") == "auto", "flag_value reads --flag=value");
    check(context_length({"--max-context", "240000"}) == 240000, "context_length reads --max-context");
    check(context_length({}) == 8192, "context_length falls back to infernix-serve's default");

    Args managed = {"--port", "9000", "--any-model-name", "--kv-dtype", "int8", "--model-id", "x", "--host", "0.0.0.0"};
    const std::vector<std::string> removed = strip_managed_flags(managed);
    check((managed == Args{"--kv-dtype", "int8"}), "managed flags and their values are stripped");
    check(removed.size() == 4, "every stripped flag is reported");

    check(windows_quote("plain") == "plain", "plain arguments stay bare");
    check(windows_quote("C:\\Program Files\\x.infernix") == "\"C:\\Program Files\\x.infernix\"", "spaces are quoted");
    check(windows_quote("a\"b") == "\"a\\\"b\"", "quotes are escaped");
    check(windows_quote("C:\\my dir\\") == "\"C:\\my dir\\\\\"", "a trailing backslash is doubled before the quote");
    check(windows_quote("") == "\"\"", "an empty argument is kept");
    check(batch_command_line({"plain", "a b"}) == "plain \"a b\"", "a batch line quotes as a program's");
    check(batch_command_line({"R&D\\"}) == "\"R&D\\\\\"", "cmd's metacharacters are quoted, even with no space");
    // The first argument's inner quote leaves cmd in its quotes at each later opening one.
    check(batch_command_line({"a\"b&c", "R&D", "x|y"}) == "\"a\\\"b^&c\" \"R^&D\" \"x^|y\"",
          "metacharacters the line's quotes leave outside cmd's are escaped");
}

void test_records() {
    const auto info = parse_serve_record("2026-10-30 11:55:04.605  INFO  build windows-2026.10.18");
    check(info && info->level == "info" && info->message == "build windows-2026.10.18", "INFO record");
    const auto warn = parse_serve_record("2026-10-30 11:55:07.233  WARN  engine | n-gram volume reads are slow");
    check(warn && warn->level == "warn" && warn->message == "engine | n-gram volume reads are slow", "WARN record");
    const auto debug = parse_serve_record("2026-10-30 11:55:04.637  DEBUG initializing CUDA");
    check(debug && debug->level == "debug" && debug->message == "initializing CUDA", "DEBUG record (one space)");
    check(!parse_serve_record("E:\\x.cpp:688: CUDA_CHECK(cudaMemcpyAsync) failed"), "a crash line is not a record");
    check(strip_ansi("\x1b[1mbold\x1b[0m\r") == "bold", "ANSI sequences are stripped");
}

void test_help() {
    const std::string help =
        "usage: infernix-serve <model.infernix> [options]\n"
        "\n"
        "Serves the OpenAI Responses/Chat Completions and Anthropic Messages APIs.\n"
        "  --help, -h                 show this help and exit\n"
        "\n"
        "MODEL & CONTEXT\n"
        "  --max-context N            max context tokens per request (default 8192)\n"
        "  --use-original-int8-prefill-kernel\n"
        "                             prefill INT8 KV with the original prompt kernel at\n"
        "                             the requested chunk\n"
        "  --reproducible             run the reproducible route\n"
        "  --spec auto|off|mtp|dflash|dflash2\n"
        "                             speculative decoding\n";
    const ServeOptions serve{.found = true, .options = parse_serve_help(help)};
    const std::vector<ServeOption>& options = serve.options;
    check(options.size() == 5, "five options parsed");
    if (options.size() == 5) {
        check(options[0].flag == "--help" && !options[0].arg, "--help with its alias, no argument");
        check(options[1].flag == "--max-context" && options[1].arg == "N" && options[1].section == "MODEL & CONTEXT",
              "argument and section");
        check(options[2].description == "prefill INT8 KV with the original prompt kernel at the requested chunk",
              "a description on the following lines is joined");
        check(options[3].flag == "--reproducible" && !options[3].arg, "a switch takes no value");
    }
    check(serve.lists_value("--spec", "auto") && serve.lists_value("--spec", "off") &&
              serve.lists_value("--spec", "dflash2"),
          "the first, a middle and the last choice are listed");
    check(!serve.lists_value("--spec", "dflash2x") && !serve.lists_value("--spec", "of") &&
              !serve.lists_value("--max-context", "N|") && !serve.lists_value("--no-such-flag", "off"),
          "a prefix, an unlisted value and an unknown flag are not");
}

void test_help_cache() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-help";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir, ignored);
    // Files standing for infernix-serve: the cache keys on the path and the modification time.
    const std::filesystem::path serve = dir / "infernix-serve.exe";
    const std::filesystem::path other = dir / "other-serve.exe";
    write_file_atomic(serve, "x");
    write_file_atomic(other, "x");
    const std::string help = "  --max-context N            max context tokens per request\n";

    std::atomic<int> runs{0};
    std::atomic<bool> works{false};
    ServeHelpCache cache([&](const std::filesystem::path&) -> std::optional<std::string> {
        ++runs;
        if (!works) { return std::nullopt; }
        return help;
    });
    const ServeOptions failed = cache.get(serve);
    check(failed.found && failed.options.empty() && failed.error, "a --help that did not run is reported");
    works = true;
    check(cache.get(serve).options.size() == 1 && runs == 2, "a failed run is not kept: the next call runs it again");
    works = false;
    check(cache.get(serve).options.size() == 1 && runs == 2, "a help that lists options is kept");

    // A capture that takes long (infernix-serve's is up to 20 s): a caller asking for the same
    // executable meanwhile shares it, and the cache's lock is not held, so another executable's help
    // is read in the meantime.
    std::atomic<bool> release{false};
    std::atomic<int> slow_runs{0};
    std::atomic<int> other_runs{0};
    const auto slow_capture = [&](const std::filesystem::path& path) -> std::optional<std::string> {
        if (path == serve) {
            ++slow_runs;
            for (int i = 0; i < 500 && !release; ++i) { std::this_thread::sleep_for(10ms); }
        } else {
            ++other_runs;
        }
        return help;
    };
    const auto wait_for_capture = [&](int runs_before) {
        for (int i = 0; i < 400 && slow_runs == runs_before; ++i) { std::this_thread::sleep_for(5ms); }
    };
    {
        ServeHelpCache shared(slow_capture);
        ServeOptions first, second;
        std::thread first_caller([&] { first = shared.get(serve); });
        wait_for_capture(0);
        std::thread second_caller([&] { second = shared.get(serve); });
        std::this_thread::sleep_for(100ms);
        release = true;
        first_caller.join();
        second_caller.join();
        check(slow_runs == 1 && first.options.size() == 1 && second.options.size() == 1,
              "callers asking during a capture share it");
    }
    {
        release = false;
        ServeHelpCache unlocked(slow_capture);
        ServeOptions blocked;
        std::thread blocked_caller([&] { blocked = unlocked.get(serve); });
        wait_for_capture(1);
        auto unrelated = std::async(std::launch::async, [&] { return unlocked.get(other); });
        const bool answered = unrelated.wait_for(2s) == std::future_status::ready;
        check(answered && other_runs == 1, "another executable's help is read while a capture runs");
        release = true;
        blocked_caller.join();
        check(unrelated.get().options.size() == 1 && blocked.options.size() == 1, "both listed");
    }
    std::filesystem::remove_all(dir, ignored);
}

void test_serve_log_files() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-serve-logs";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    LogBuffer logs(dir);
    logs.begin_instance(1, "first model");
    logs.append_output(1, "2026-10-30 11:55:04.605  INFO  build windows-2026.10.18\nraw ");
    logs.append_output(1, "line\n");
    check(logs.page(0, 10)["lines"].size() == 2, "the server's lines are records at once");
    logs.end_instance();
    logs.begin_instance(2, "second");
    logs.append_output(2, "second output\n");
    check(!std::filesystem::exists(dir, ignored), "reading the server's output writes nothing (write_files does)");

    logs.write_files();
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ignored)) { files.push_back(entry.path()); }
    std::sort(files.begin(), files.end());
    check(files.size() == 2, "one log file per server instance");
    if (files.size() == 2) {
        check(files[0].filename().string().ends_with("-first_model.log") && files[1].filename().string().ends_with("-second.log"),
              "serve-<time>-<model>.log");
        check(read_file(files[0]).value_or("") == "2026-10-30 11:55:04.605  INFO  build windows-2026.10.18\nraw line\n",
              "an instance's output as it came, in its own file");
        check(read_file(files[1]).value_or("") == "second output\n", "the next instance's in the next file");
        logs.append_output(2, "more\n");
        logs.write_files();
        check(read_file(files[1]).value_or("") == "second output\nmore\n", "later output joins the open file");
    }
    logs.end_instance();
    logs.write_files();
    std::filesystem::remove_all(dir, ignored);
}

std::string volume_header(std::uint32_t row_bytes, const std::uint8_t id[16], std::uint16_t scale, std::uint16_t tag) {
    std::string header(68, '\0');
    std::memcpy(header.data(), "NINFERNG", 8);
    const auto put = [&](std::size_t at, auto value) { std::memcpy(header.data() + at, &value, sizeof(value)); };
    put(8, std::uint32_t{1});
    put(12, std::uint32_t{4096});
    put(16, std::uint64_t{320001536});
    put(24, row_bytes);
    put(28, std::uint32_t{4096 / row_bytes});
    put(32, std::uint32_t{4096});
    put(36, std::uint64_t{(320001536 + 4096 / row_bytes - 1) / (4096 / row_bytes)});
    std::memcpy(header.data() + 44, id, 16);
    put(60, scale);
    put(62, tag);
    return header;
}

Json flash_next_directory() {
    return {{"components",
             {{"text",
               {{"config",
                 {{"model_type", "qwen4_exp_text"},
                  {"hidden_size", 2560},
                  {"num_hidden_layers", 48},
                  {"vocab_size", 248320},
                  {"max_position_embeddings", 262144},
                  {"num_experts", 512},
                  {"ngram_table",
                   {{"format", "fp8_e4m3fn"},
                    {"rows", 320001536},
                    {"row_bytes", 160},
                    {"volume_id", "59e5d77e802ac95d8faf0a800b8f2301"}}}}},
                {"proposal", {{"domain", "indexed"}, {"rows", 131072}}}}},
              {"vision", {{"config", Json::object()}, {"target", "text"}}},
              {"mtp", {{"config", Json::object()}, {"target", "text"}}}}},
            {"objects",
             Json::array({{{"id", "a"}, {"kind", "tensor"}, {"format", "nvfp4"}, {"bytes", 900}},
                          {{"id", "b"}, {"kind", "tensor"}, {"format", "fp8_e4m3fn_row_bf16"}, {"bytes", 1000}},
                          {{"id", "n"}, {"kind", "tensor"}, {"format", "fp32"}, {"bytes", 4000}},
                          {{"id", "c"}, {"kind", "resource"}, {"encoding", "raw_bytes_v1"}, {"bytes", 5000}}})},
            {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 1000}},
                                   {{"path", "Next-00002-of-00002.infernix"}, {"payload_bytes", 2000}}})},
            {"metadata", {{"name", "Qwen3.8-Flash-Next-Test"}}},
            {"provenance", {{"recipe", "qwen3_8_flash_next_dense8"}}}};
}

void test_library() {
    const std::array<std::uint8_t, 16> id{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const std::filesystem::path entry = "Z:/models/Next-00001-of-00002.infernix";
    ParsedEntry parsed = parse_artifact_directory(entry, id, flash_next_directory(), 4096);
    check(parsed.model.has_value() && !parsed.drafter.has_value(), "a text component makes a model");
    if (!parsed.model) { return; }
    ScannedModel& model = *parsed.model;
    check(model.key == "0102030405060708090a0b0c0d0e0f10", "the artifact id is the key");
    check(model.family == "qwen3.8-flash-next" && model.architecture == "Qwen4ExpForCausalLM", "Flash-Next family");
    check(model.needs_ngram_volume && model.accepts_draft_model && model.offloaded_experts, "Flash-Next needs");
    check(model.has_proposal && model.has_vision && model.has_mtp && !model.has_dflash2, "components");
    // 1,600 NVFP4 parameters in 900 bytes outnumber 1,000 FP8 ones in more bytes; FP32 norms are not named.
    check(model.quantization == "NVFP4 + FP8", "quantization names the formats holding most parameters");
    check(model.recipe == "qwen3_8_flash_next_dense8", "recipe from provenance");
    check(model.bytes == 3000 + 4096 + 4096, "payload plus framing");

    Json drafter_directory = {
        {"components",
         {{"dflash2",
           {{"config",
             {{"model_type", "qwen3"},
              {"target_config",
               {{"architectures", Json::array({"Qwen4ExpForCausalLM"})},
                {"hidden_size", 2560},
                {"num_hidden_layers", 48},
                {"vocab_size", 248320}}}}},
            {"target", "text"}}}}},
        {"objects", Json::array()},
        {"files", Json::array({{{"path", nullptr}, {"payload_bytes", 776051200}}})},
        {"metadata", {{"name", "Qwen3.8-Flash-Next-DFlash2-Infernix"}}}};
    const std::array<std::uint8_t, 16> drafter_id{9};
    ParsedEntry drafter = parse_artifact_directory("Z:/models/drafter.infernix", drafter_id, drafter_directory, 4096);
    check(drafter.drafter && drafter.drafter->kind == "dflash2" &&
              drafter.drafter->target_architecture == "Qwen4ExpForCausalLM",
          "a text-less artifact is a companion drafter with its target");

    const std::uint8_t nvidia[16] = {0x59, 0xE5, 0xD7, 0x7E, 0x80, 0x2A, 0xC9, 0x5D,
                                     0x8F, 0xAF, 0x0A, 0x80, 0x0B, 0x8F, 0x23, 0x01};
    const std::uint8_t other[16]  = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    const auto own  = parse_ngram_header("own.ngram", volume_header(160, nvidia, 0, 0));
    const auto iq4  = parse_ngram_header("iq4.ngram", volume_header(90, other, 0, 0));
    const auto bf16 = parse_ngram_header("bf16.ngram", volume_header(320, other, 0, 3));
    const auto fp8_unscaled = parse_ngram_header("fp8.ngram", volume_header(160, other, 0, 1));
    check(own && iq4 && bf16 && fp8_unscaled, "n-gram headers parse");
    check(!parse_ngram_header("x", std::string(68, 'x')), "other files are not n-gram volumes");
    if (own && iq4 && bf16 && fp8_unscaled) {
        check(iq4->format == "iq4_nl" && bf16->format == "bf16" && own->format == "fp8", "formats from tag or row size");
        check(ngram_rank(model, *own) == 0, "the artifact's own volume ranks first");
        check(ngram_rank(model, *iq4) == 3 && ngram_rank(model, *bf16) == 4, "IQ4_NL before BF16");
        check(!ngram_rank(model, *fp8_unscaled), "an FP8 volume without a scale serves only its own artifact");
    }

    LibraryScan scan;
    model.path = entry;
    scan.models.push_back(model);
    ScannedDrafter dflash2         = *drafter.drafter;
    dflash2.compatible_models = {model.key};
    scan.drafters.push_back(dflash2);
    ScannedNgramVolume volume = *iq4;
    volume.path               = "Z:/volumes/iq4.ngram";
    scan.ngram_volumes.push_back(volume);
    const Args args = default_args(scan, model, HostFacts{96ull << 30});
    check(contains(args, {"--max-context", "262144"}), "Flash-Next default context");
    check(contains(args, {"--spec", "auto"}) && contains(args, {"--draft-tokens", "auto"}), "drafter chosen by the Engine");
    check(has_flag(args, "--lm-head-draft"), "proposal head used");
    check(contains(args, {"--draft-model", path_utf8(dflash2.path)}), "the DFlash2 drafter is added");
    check(contains(args, {"--ngram-volume", "Z:/volumes/iq4.ngram"}), "the best n-gram volume is added");
    check(!has_flag(args, "--host-context-mib"), "Flash-Next sizes its own Host tier");
    check(default_alias(scan, model) == "Next", "the alias drops the part numbering");

    ScannedModel dense;
    dense.key          = "ff";
    dense.path         = "Z:/models/qwen3_8_27b_nvfp4.ninfer";
    dense.model_type   = "qwen3_5_text";
    dense.hidden_size  = 5120;
    dense.has_dflash2  = true;
    dense.has_mtp      = true;
    dense.has_proposal = true;
    dense.family       = "qwen3.8-27b";
    dense.bytes        = 23719715844; // neroued/Qwen3.8-27B-nvfp4-NInfer
    const HostFacts rtx5090{96ull << 30, 32607ull << 20};
    const Args dense_args = default_args(scan, dense, rtx5090);
    check(contains(dense_args, {"--ngram-draft-tokens", "auto"}) && contains(dense_args, {"--prefill-chunk", "4096"}) &&
              contains(dense_args, {"--host-context-mib", "52000"}),
          "Qwen3.8-27B defaults follow the README's recommended settings");
    check(!has_flag(dense_args, "--draft-model"), "bundled drafters take no --draft-model");
    // On a 32 GB RTX 5090 this artifact started with at most ~196,000 tokens beside the Engine's
    // 1 GiB headroom (240,000 failed): the default stays inside that, with room for other programs.
    const std::uint64_t fit = fitting_27b_context(dense, rtx5090);
    check(fit >= 180000 && fit <= 196000 && contains(dense_args, {"--max-context", std::to_string(fit)}),
          "the 27B context fits beside the weights on a 32 GB GPU");
    ScannedModel small = dense;
    small.bytes        = 20437521664;
    check(fitting_27b_context(small, rtx5090) == 240000, "a smaller artifact gets the full 240,000");
    check(fitting_27b_context(dense, HostFacts{96ull << 30, 0}) == 188416, "an unknown GPU gets the 32 GB value");
    check(fitting_27b_context(dense, HostFacts{96ull << 30, 24ull << 30}) == 32768, "a too-small GPU gets the floor");
    // The Engine's automatic headroom follows the display (VM-07 b): 256 MiB headless, 1 GiB with a
    // display or when NVML cannot tell (the case above).
    check(fitting_27b_context(dense, HostFacts{96ull << 30, 32607ull << 20, false}) == 208896 &&
              fitting_27b_context(dense, HostFacts{96ull << 30, 32607ull << 20, true}) == 188416,
          "the 27B context does not follow the GPU's display");
}

void test_router() {
    std::mutex mutex;
    std::vector<std::string> events;
    std::atomic<std::uint32_t> instance{0};
    std::atomic<bool> fail_next{false};
    std::atomic<bool> end_next{false};
    Router* routed = nullptr;
    const auto record = [&](const std::string& event) {
        std::lock_guard lock(mutex);
        events.push_back(event);
    };
    const auto snapshot = [&] {
        std::lock_guard lock(mutex);
        return events;
    };
    Router router(Router::Hooks{
        .load =
            [&](const std::string& key) {
                record("load:" + key);
                std::this_thread::sleep_for(30ms);
                Router::Loaded loaded;
                if (fail_next.exchange(false)) {
                    loaded.error = "boom";
                    return loaded;
                }
                loaded.instance = ++instance;
                loaded.port     = 1000 + static_cast<int>(loaded.instance);
                // The server exits between becoming ready and the load returning.
                if (end_next.exchange(false)) { routed->instance_ended(loaded.instance); }
                return loaded;
            },
        .stop            = [&] { record("stop"); },
        .changed         = [] {},
        .drain_timeout_s = [] { return 60; },
        .load_timeout_s  = [] { return 30; },
        .idle_unload_min = [](const std::string&) { return 0; },
    });
    routed = &router;
    Router::Refusal refusal;
    auto a1 = router.acquire("a", nullptr, refusal);
    check(a1 && a1->port() == 1001, "the first request loads its model");
    auto a2 = router.acquire("a", nullptr, refusal);
    check(a2 && a2->instance() == a1->instance(), "a request for the loaded model is admitted at once");

    std::shared_ptr<Router::Lease> b, a3;
    std::thread wants_b([&] { b = router.acquire("b", nullptr, refusal); });
    std::this_thread::sleep_for(300ms);
    // A request for the loaded model arriving after the switch was asked for waits its turn.
    std::thread wants_a([&] { a3 = router.acquire("a", nullptr, refusal); });
    std::this_thread::sleep_for(300ms);
    check(snapshot() == std::vector<std::string>{"load:a"}, "the switch waits for running requests (drain)");
    check(router.pending_model() == "b", "the pending switch is reported");
    a1.reset();
    a2.reset();
    wants_b.join();
    check(b && b->instance() == 2, "the next model loads once the running requests finish");
    std::this_thread::sleep_for(300ms);
    check(!a3, "a later request for the previous model waits behind the switch");
    b.reset();
    wants_a.join();
    check(a3 && a3->instance() == 3, "then its model loads again, in arrival order");
    check((snapshot() == std::vector<std::string>{"load:a", "load:b", "load:a"}), "load order");
    a3.reset();

    fail_next = true;
    auto c = router.acquire("c", nullptr, refusal);
    check(!c && refusal.code == "model_load_failed" && refusal.message == "boom", "a failed load refuses its requests");
    check(refusal.retry_after_s > 60 && refusal.retry_after_s <= 120, "...with a Retry-After over a minute");
    // Coding tools retry 5xx answers: inside the hold a retry is refused without another load.
    const std::size_t loads_before_retry = snapshot().size();
    auto retried = router.acquire("c", nullptr, refusal);
    check(!retried && refusal.code == "model_load_failed" && refusal.message.starts_with("boom (") &&
              refusal.retry_after_s > 60 && snapshot().size() == loads_before_retry,
          "a retry inside the hold is refused without loading again");
    Router::Refusal peek;
    check(router.held("c", peek) && !router.held("a", peek), "only the failed model is held");
    router.clear_hold("c");
    check(!router.held("c", peek), "a changed configuration ends the hold");
    fail_next = true;
    check(!router.acquire("c", nullptr, refusal) && router.held("c", peek), "a second failure holds it again");
    router.request_load("c", false);
    auto loaded_c = router.acquire("c", nullptr, refusal);
    check(loaded_c != nullptr, "Load in the UI ends the hold and loads the model");
    loaded_c.reset();

    auto held = router.acquire("a", nullptr, refusal);
    check(held != nullptr, "the model loads again after a failure");
    router.request_load("d", true);
    std::this_thread::sleep_for(700ms);
    const std::vector<std::string> forced = snapshot();
    check(!forced.empty() && forced.back() == "load:d", "a forced load does not wait for running requests");
    held.reset();

    // Restart (unload, then load the same model) loads it again rather than keeping the old instance.
    const std::size_t before_restart = snapshot().size();
    router.request_unload();
    router.request_load("d", true);
    auto d = router.acquire("d", nullptr, refusal);
    const std::vector<std::string> restarted = snapshot();
    check(d && restarted.size() == before_restart + 2 && restarted[before_restart] == "stop" &&
              restarted.back() == "load:d",
          "a restart stops the model and loads it again");
    d.reset();

    end_next = true;
    auto e = router.acquire("e", nullptr, refusal);
    check(!e && refusal.code == "model_load_failed", "a server that exits as soon as it is ready refuses its requests");
    check(!router.acquire("e", nullptr, refusal) && refusal.retry_after_s > 60, "...and is held like any failed load");
    router.request_load("e", false);
    auto e2 = router.acquire("e", nullptr, refusal);
    check(e2 != nullptr, "Load in the UI loads it again");
    e2.reset();

    // The watchdog's restart of a crashed instance yields to an Unload of it, whether the Unload
    // lands before the crash is seen or after.
    const auto instance_of = [&](const std::string& key) {
        auto lease = router.acquire(key, nullptr, refusal);
        return lease ? lease->instance() : 0U;
    };
    const std::uint32_t f = instance_of("f");
    router.request_unload();
    router.instance_ended(f);
    check(f != 0 && router.unloaded(f) && !router.restart("f", f), "a crashed instance unloaded first is not restarted");
    const std::uint32_t g = instance_of("g");
    router.instance_ended(g);
    router.request_unload();
    check(g > f && !router.restart("g", g), "nor one unloaded after its crash was seen");
    const std::uint32_t h = instance_of("h");
    router.instance_ended(h);
    check(h > g && !router.unloaded(h) && router.restart("h", h), "a crash with no Unload restarts its model");
    check(instance_of("h") > h, "...as a new instance");
    router.shutdown();
}

void test_load_advice() {
    // infernix-serve's words when the official 27B NVFP4 did not fit at 240,000 tokens (RTX 5090).
    const std::string kv_error =
        "infernix-serve failed to start: server failed during startup | automatic KV capacity requires 11215218944 "
        "bytes total (10141477120 bytes minimum Engine runtime reservation + 1073741824 bytes automatic headroom), "
        "but only 9738125312 bytes are available after weights";
    check(classify_load_failure(kv_error) == "vram", "a KV reservation shortfall is a VRAM failure");
    check(classify_load_failure("SSD tier: the OS let Infernix page-lock only 512 MiB") == "ram", "pinning is RAM");
    check(classify_load_failure("the SSD expert tier needs RAM for at least 4096 experts") == "ram", "the SSD tier is RAM");
    check(!classify_load_failure("startup failed | cannot open the artifact"), "other failures are not resource ones");
    // infernix-serve's refusal of a port another program holds (it binds exclusively).
    check(classify_load_failure("infernix-serve failed to start: cannot bind 127.0.0.1:61000: the port is in use "
                                "(another server, an earlier infernix-serve say, holds it): stop it or choose "
                                "another --port") == "port",
          "a port in use is a port failure");
    const std::vector<LoadSuggestion> fixed_port = suggest_port_fixes(61000, true);
    check(!fixed_port.empty() && fixed_port.front().title == "Let the launcher choose the server's port" &&
              std::none_of(fixed_port.begin(), fixed_port.end(), [](const LoadSuggestion& s) { return s.args.has_value(); }),
          "a fixed port's advice starts with the automatic port, and changes no launch parameter");
    const std::vector<LoadSuggestion> automatic_port = suggest_port_fixes(61000, false);
    check(!automatic_port.empty() && automatic_port.front().title == "Stop the program that holds port 61000",
          "an automatic port's advice names the port's holder");

    ScannedModel dense;
    dense.family = "qwen3.8-27b";
    const Args args = {"--max-context", "240000", "--max-concurrency", "2", "--kv-dtype", "int8", "--host-context-mib", "52000"};
    const ResourceFacts facts{96ull << 30, 60ull << 30, 32607ull << 20, 2ull << 30};
    const std::vector<LoadSuggestion> fixes = suggest_load_fixes("vram", dense, args, kv_error, facts);
    // Measured: the artifact starts at up to ~196,000 tokens beside the 1 GiB headroom.
    const auto lower = std::find_if(fixes.begin(), fixes.end(), [](const LoadSuggestion& s) {
        return s.args && flag_value(*s.args, "--max-context") != "240000";
    });
    check(lower != fixes.end(), "a smaller context is suggested");
    if (lower != fixes.end()) {
        const std::uint64_t context = std::stoull(*flag_value(*lower->args, "--max-context"));
        check(context >= 160000 && context <= 196000 && context % 4096 == 0, "the suggested context fits, in 4,096 steps");
        check(flag_value(*lower->args, "--kv-dtype") == "int8", "a suggestion changes one thing");
    }
    check(std::any_of(fixes.begin(), fixes.end(),
                      [](const LoadSuggestion& s) { return s.args && flag_value(*s.args, "--vram-headroom-mib") == "0"; }),
          "the headroom can be given up");
    check(std::any_of(fixes.begin(), fixes.end(),
                      [](const LoadSuggestion& s) { return s.args && flag_value(*s.args, "--max-concurrency") == "1"; }),
          "one lane is suggested");
    check(std::none_of(fixes.begin(), fixes.end(),
                       [](const LoadSuggestion& s) { return s.args && flag_value(*s.args, "--kv-dtype") != "int8"; }),
          "INT8 KV is not suggested again");
    check(std::any_of(fixes.begin(), fixes.end(), [](const LoadSuggestion& s) { return !s.args; }),
          "other programs' GPU memory is named");

    ScannedModel offloaded;
    offloaded.family            = "qwen3.8-flash-next";
    offloaded.offloaded_experts = true;
    const std::vector<LoadSuggestion> ram = suggest_load_fixes(
        "ram", offloaded, {"--max-context", "262144", "--host-context-mib", "8192"}, "free RAM fell", facts);
    check(std::any_of(ram.begin(), ram.end(),
                      [](const LoadSuggestion& s) {
                          return s.args && flag_value(*s.args, "--ram-free-mib") == "16384" &&
                                 flag_value(*s.args, "--expert-ram-mib") == "auto";
                      }),
          "more RAM left free with automatic expert sizing");
    check(std::any_of(ram.begin(), ram.end(),
                      [](const LoadSuggestion& s) { return s.args && flag_value(*s.args, "--host-context-mib") == "4096"; }),
          "a smaller Host prefix cache");
}

void test_event_log() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-events";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    EventLog log(dir);
    std::vector<std::string> heard;
    log.set_listener([&](const EventRecord& record) { heard.push_back(record.message); });
    log.add("info", "launcher", "first");
    log.add("error", "watchdog", "second line\nwith a break", Json{{"kind", "crash"}});
    const Json page = log.page(0, 10);
    check(page["lines"].size() == 2 && page["lines"][1]["fields"]["kind"] == "crash", "records keep their fields");
    check(log.page(1, 10)["lines"].size() == 1, "a page after a sequence number");
    check(heard == std::vector<std::string>{"first", "second line\nwith a break"}, "the listener hears every record");
    const std::optional<std::string> text = read_file(dir / EventLog::kEventsFile);
    check(text && text->find("  INFO   launcher  first\n") != std::string::npos, "the file has one text line per record");
    check(text && text->find("ERROR  watchdog  second line with a break\n") != std::string::npos, "one record, one line");
    log.add_request_body({{"id", 1}});
    check(read_file(dir / EventLog::kRequestsFile) == std::optional<std::string>("{\"id\":1}\n"), "request bodies, one JSON line each");

    const std::filesystem::path moved = dir / "moved";
    std::string error;
    check(log.set_dir(moved, error), "the log folder can change");
    log.add("info", "settings", "third");
    check(read_file(moved / EventLog::kEventsFile).value_or("").find("third") != std::string::npos, "later records go there");
    check(log.page(0, 10)["lines"].size() == 3, "the memory keeps the earlier records");
#ifdef _WIN32
    {
        // A full file another program holds open without sharing delete (a tail tool) cannot be
        // rotated: it keeps its records and grows until a later try moves it.
        const std::filesystem::path held_dir = dir / "held";
        std::filesystem::create_directories(held_dir, ignored);
        const std::filesystem::path held = held_dir / EventLog::kEventsFile;
        {
            std::ofstream fill(held, std::ios::binary);
            fill << "oldest record\n" << std::string(EventLog::kEventsLimit - 20, 'x') << '\n';
        }
        HANDLE tail = CreateFileW(held.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        check(tail != INVALID_HANDLE_VALUE, "a tail tool holds the file");
        check(log.set_dir(held_dir, error), "the log folder can change to the held file's");
        log.add("info", "launcher", "past the limit");
        const std::string head = read_file(held).value_or("").substr(0, 14);
        check(head == "oldest record\n" && std::filesystem::file_size(held, ignored) > EventLog::kEventsLimit &&
                  !std::filesystem::exists(held_dir / (std::string(EventLog::kEventsFile) + ".1")),
              "a file that cannot be moved keeps its records");
        if (tail != INVALID_HANDLE_VALUE) { CloseHandle(tail); }
        const std::string block(64 << 10, 'y');
        for (int i = 0; i < 20; ++i) { log.add("info", "launcher", block); }
        check(read_file(held_dir / (std::string(EventLog::kEventsFile) + ".1")).value_or("").starts_with("oldest record\n") &&
                  std::filesystem::file_size(held, ignored) < EventLog::kEventsLimit,
              "a later try rotates it once it is free");
    }
#endif
    std::filesystem::remove_all(dir, ignored);
}

void test_routing() {
    check(wildcard_match("claude-*haiku*", "claude-3-5-HAIKU-20241022"), "wildcards match any run, ignoring case");
    check(wildcard_match("*", "") && wildcard_match("gpt-*", "gpt-") && !wildcard_match("gpt-*", "chatgpt-4"),
          "a pattern matches the whole name");
    check(wildcard_match("a*b*c", "aXbYbZc") && !wildcard_match("a*b*c", "aXbYbZ"), "backtracking");
    const std::vector<ModelRoute> routes = {{"claude-*haiku*", "loaded"}, {"claude-*", "k@00000001"}};
    const ModelRoute* route = match_route(routes, "claude-haiku-4-5");
    check(route != nullptr && route->target == "loaded", "the first matching route wins");
    check(match_route(routes, "gpt-5") == nullptr, "no route, no match");
    check(artifact_key_of("0123@abcd0123") == "0123" && artifact_key_of("0123") == "0123" &&
              is_named_config("0123@abcd0123") && !is_named_config("0123"),
          "configuration ids");
    const std::string id = new_config_id("ff");
    check(id.starts_with("ff@") && id.size() == 11, "a new configuration id is <key>@<8 hex>");
}

void test_launch_scripts() {
    const TakesValue takes = [](std::string_view flag) -> std::optional<bool> {
        if (flag == "--lm-head-draft" || flag == "--preserve-thinking" || flag == "--tolerant-tool-calls") { return false; }
        return true;
    };
    // The owner's NInfer launch script, abridged.
    const std::string bat =
        "@echo off\r\nrem a comment\r\nset NINFER_CUDA_SYNC=blocking\r\n"
        "\"ninfer-serve.exe\" \"E:\\NInfer-Deploy-V3\\qwen3_8_27b_nvfp4-official.ninfer\" --host 127.0.0.1 --port 8080 "
        "--max-context 220000 --lm-head-draft ^\r\n  --thinking-budget-message \"Time to act:\" --tolerant-tool-calls\r\npause\r\n";
    check(script_shell(bat, "Launch.bat") == ScriptShell::Cmd && script_shell("export A=1\n", "") == ScriptShell::Sh,
          "the script's shell");
    const std::vector<ScriptCommand> commands = parse_launch_script(bat, ScriptShell::Cmd, takes);
    check(commands.size() == 1, "one server command");
    if (commands.size() == 1) {
        const ScriptCommand& command = commands[0];
        check(command.artifact == "E:\\NInfer-Deploy-V3\\qwen3_8_27b_nvfp4-official.ninfer", "the artifact");
        check(contains(command.args, {"--max-context", "220000"}) && contains(command.args, {"--thinking-budget-message", "Time to act:"}) &&
                  has_flag(command.args, "--tolerant-tool-calls"),
              "arguments, quoted and continued");
        check(command.env.size() == 1 && command.env[0].first == "NINFER_CUDA_SYNC" && command.env[0].second == "blocking",
              "set VAR=value");
    }
    const std::string sh = "#!/bin/sh\nexport CUDA_VISIBLE_DEVICES=0\nMODEL=/models/a.infernix\n"
                           "INFERNIX_X=1 exec ./infernix-serve \"$MODEL\" --max-context 8192 \\\n  --kv-dtype int8 2>&1 | tee log\n";
    const std::vector<ScriptCommand> posix = parse_launch_script(sh, ScriptShell::Sh, takes);
    check(posix.size() == 1 && posix[0].artifact == "/models/a.infernix" &&
              contains(posix[0].args, {"--kv-dtype", "int8"}) && posix[0].args.size() == 4,
          "a shell script's command, variables expanded, redirections cut");
    check(posix.size() == 1 && posix[0].env.size() == 2, "exported and per-command variables; plain ones stay in the script");

    const std::string out = write_launch_script(ScriptShell::Cmd, "C:\\Infernix\\infernix-serve.exe", "D:\\m.infernix",
                                                {"--thinking-budget-message", "100% done"}, {{"A", "1"}}, "model-a");
    check(out.find("set \"A=1\"") != std::string::npos && out.find("100%% done") != std::string::npos &&
              out.find("--model-id model-a") != std::string::npos,
          "a batch script: variables, %% doubled, a model id");
    check(out.find("chcp 65001") != std::string::npos, "a batch script reads in UTF-8");
    const std::vector<ScriptCommand> batch_round = parse_launch_script(
        write_launch_script(ScriptShell::Cmd, "C:\\R&D\\infernix-serve.exe", "D:\\R&D\\m.infernix",
                            {"--thinking-budget-message", "say \"go\" & act|now"}, {}, "m"),
        ScriptShell::Cmd, takes);
    check(batch_round.size() == 1 && batch_round[0].artifact == "D:\\R&D\\m.infernix" &&
              contains(batch_round[0].args, {"--thinking-budget-message", "say \"go\" & act|now"}),
          "a batch script with cmd's metacharacters written and read back");
    const std::vector<ScriptCommand> round = parse_launch_script(
        write_launch_script(ScriptShell::Sh, "/opt/infernix-serve", "/m/x y.infernix", {"--max-context", "4096"}, {{"A", "it's"}}, "x"),
        ScriptShell::Sh, takes);
    check(round.size() == 1 && round[0].artifact == "/m/x y.infernix" && round[0].env.size() == 1 &&
              round[0].env[0].second == "it's",
          "a shell script written and read back");
    // A launcher whose server has another file name reads back its own exports.
    const std::string renamed =
        write_launch_script(ScriptShell::Cmd, "D:\\builds\\serve-dev.exe", "D:\\m.infernix", {"--max-context", "4096"}, {}, "m");
    check(parse_launch_script(renamed, ScriptShell::Cmd, takes).empty() &&
              parse_launch_script(renamed, ScriptShell::Cmd, takes, "serve-dev.exe").size() == 1,
          "the launcher's own server is recognised by its file name");
}

void test_usage_book() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-usage";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir, ignored);
    const std::int64_t now = now_ms();
    {
        UsageBook book(dir);
        UsageSample sample;
        sample.t             = now;
        sample.config_id     = "k";
        sample.model         = "model-k";
        sample.client        = "hermes";
        sample.prompt_tokens = 1000;
        sample.cached_tokens = 600;
        sample.output_tokens = 50;
        book.add(sample);
        sample.failed = true;
        book.add(sample);
    }
    UsageBook reread(dir);
    const Json rows = reread.rows(local_day(now), local_day(now));
    check(rows.size() == 1 && rows[0]["requests"] == 2 && rows[0]["failed"] == 1 && rows[0]["cached_tokens"] == 1200 &&
              rows[0]["client"] == "hermes",
          "usage adds up per day, configuration and client, and survives a restart");
    check(reread.rows("1999-01-01", "1999-12-31").empty(), "outside the range nothing");
    std::filesystem::remove_all(dir, ignored);
}

void test_release_versions() {
    check(release_version("windows-2026.10.18-174-g52bcb024-dirty") == "2026.10.18", "a build id's release");
    check(release_version("windows-2026.10.09.2") == "2026.10.09.2" && !release_version("g52bcb024"), "tags");
    check(compare_versions("2026.10.18", "2026.10.09.2") > 0 && compare_versions("2026.10.09.2", "2026.10.09") > 0 &&
              compare_versions("2026.10.09", "2026.10.09.0") == 0,
          "versions compare part by part");
}

void test_release_answer() {
    // A release published without notes whose zip GitHub did not hash: both fields are null.
    const std::string unhashed = R"({"tag_name": "windows-2026.11.03", "published_at": "2026-11-03T09:00:00Z",
        "html_url": "https://github.com/Wallawalla47/Infernix/releases/tag/windows-2026.11.03", "body": null,
        "assets": [{"name": "ffmpeg-2026.11.03-source.zip", "size": 10, "digest": null, "browser_download_url": "https://a/f.zip"},
                   {"name": "infernix-2026.11.03-windows-x64.zip", "size": 123, "digest": null,
                    "browser_download_url": "https://a/i.zip"}]})";
    std::optional<Json> release = parse_release(unhashed);
    check(release && (*release)["version"] == "2026.11.03" && (*release)["notes"] == "",
          "a release without notes is read, not refused");
    check(release && (*release)["asset"]["name"] == "infernix-2026.11.03-windows-x64.zip" &&
              (*release)["asset"]["bytes"] == 123 && (*release)["asset"]["sha256"].is_null() &&
              (*release)["asset"]["url"] == "https://a/i.zip",
          "the Windows zip without a digest has no SHA-256");
    release = parse_release(R"({"tag_name": "windows-2026.11.03", "body": "notes", "published_at": null, "html_url": null,
        "assets": [{"name": "infernix-2026.11.03-windows-x64.zip", "size": null, "browser_download_url": "https://a/i.zip",
                    "digest": "sha256:00ff"}]})");
    check(release && (*release)["notes"] == "notes" && (*release)["published_at"] == "" &&
              (*release)["asset"]["sha256"] == "00ff" && (*release)["asset"]["bytes"] == 0,
          "GitHub's sha256: digest is kept; other null fields read as empty");
    release = parse_release(R"({"tag_name": "windows-2026.11.03",
        "assets": [{"name": "infernix-x\\..\\..\\evil-windows-x64.zip", "browser_download_url": "https://a/e.zip"}]})");
    check(release && (*release)["asset"].is_null(), "an asset name that is not a plain file name is no update");
    check(!parse_release("not json") && !parse_release("[1, 2]"), "an answer that is not a JSON object is unreadable");

    check(is_windows_release_asset("infernix-2026.11.03-windows-x64.zip") &&
              is_windows_release_asset("infernix-2026.11.03.2-windows-x64.zip"),
          "release zips");
    check(!is_windows_release_asset("infernix-a/b-windows-x64.zip") && !is_windows_release_asset("infernix-a b-windows-x64.zip") &&
              !is_windows_release_asset("infernix-windows-x64.zip") && !is_windows_release_asset("ffmpeg-1-windows-x64.zip") &&
              !is_windows_release_asset("infernix-1-windows-x64.zip.exe") && !is_windows_release_asset("infernix-1:x-windows-x64.zip"),
          "separators, spaces, other programs and other suffixes are refused");
}

void test_scan_folder() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-scan";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir / "sub", ignored);
    const LibraryScan scan = scan_library({dir}, 2);
    check(scan.dirs.size() == 1 && scan.dirs[0].exists && !scan.dirs[0].error, "an empty folder scans whole");
    // The drive's room comes with the scan, so the library routes never ask the drive.
    const Json dirs = to_json(scan)["dirs"];
    check(scan.dirs.size() == 1 && scan.dirs[0].free_bytes && scan.dirs[0].total_bytes &&
              dirs[0]["free_bytes"].is_number() && dirs[0]["total_bytes"].is_number(),
          "the scan records its drive's free and total bytes");
    std::filesystem::remove_all(dir, ignored);
}

void test_exit_meaning() {
#ifdef _WIN32
    check(exit_status_meaning(static_cast<int>(0xC0000005u), true).find("access violation") != std::string::npos, "0xC0000005");
    check(exit_status_meaning(static_cast<int>(0xC0000409u), true).find("fail-fast") != std::string::npos, "0xC0000409");
    // The loader's statuses: the server never started, and the note names the release to take the DLLs from.
    for (const unsigned status : {0xC0000135u, 0xC0000139u, 0xC000007Bu, 0xC0000142u}) {
        const std::string meaning = exit_status_meaning(static_cast<int>(status), false);
        check(meaning.starts_with("could not start: a DLL") &&
                  meaning.find("https://github.com/Wallawalla47/Infernix/releases/latest") != std::string::npos &&
                  meaning.find("crashed") == std::string::npos,
              "a DLL load failure names the release instead of a crash");
    }
#else
    check(exit_status_meaning(128 + SIGSEGV, true).find("segmentation fault") != std::string::npos, "SIGSEGV");
    const std::string killed = exit_status_meaning(128 + SIGKILL, true);
    check(killed.find("out-of-memory") != std::string::npos && killed.find("systemd-oomd") != std::string::npos,
          "SIGKILL names both memory-pressure killers");
#endif
    check(exit_status_meaning(1, false) == "failed to start", "infernix-serve's own statuses first");
    // taskkill /F ends a serving server with status 1: not a startup failure.
    check(exit_status_meaning(1, true).find("exited unexpectedly") != std::string::npos, "status 1 while serving");
}

void test_events() {
    check(EventHub::format("toast", Json{{"a", 1}}) == "event: toast\ndata: {\"a\":1}\n\n", "SSE framing");
    EventHub hub;
    auto subscription = hub.subscribe();
    hub.publish("x", Json{{"n", 1}});
    const auto event = subscription->next(100ms);
    check(event && event->find("event: x") == 0, "a subscriber receives published events");
    hub.close_all();
    check(subscription->closed() && !subscription->next(10ms), "closing ends subscriptions");
}

} // namespace

// Download checks: git blob ids (`git hash-object`, Python's hashlib for the longer ones).
void test_blob_ids() {
    check(git_blob_id("") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" &&
              git_blob_id("hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a",
          "git's own blob ids");
    check(git_blob_id(std::string(55, 'b')) == "13836c2549eab2ac757d0279ebbb2641a6a6e1ac",
          "a header and contents one byte short of a block pad into another");
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "infernix-launcher-test-blob";
    {
        std::ofstream out(path, std::ios::binary);
        out << std::string(1000, 'a');
    }
    const std::atomic<bool> cancel{false};
    check(git_blob_id_file(path, cancel) == std::optional<std::string>("a50be72b20f0e3f078d252e8e56b11b4bec67509"),
          "a file's blob id, over several blocks");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

// The chat list reads each file once until it changes, a save through the store or a file
// replaced beside it.
void test_chats() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-chats";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    ChatStore store(dir);
    store.put("a", Json{{"title", "first"}, {"messages", Json::array()}, {"updated_at", 1}});
    store.put("b", Json{{"title", "second"}, {"messages", Json::array({1})}, {"updated_at", 2}});
    Json list = store.list();
    check(list.size() == 2 && list[0]["id"] == "b" && list[0]["messages"] == 1 && list[1]["title"] == "first",
          "chat summaries, newest first");
    store.put("a", Json{{"title", "renamed"}, {"messages", Json::array()}, {"updated_at", 3}});
    list = store.list();
    check(list.size() == 2 && list[0]["title"] == "renamed", "a save shows in the next list");
    write_file_atomic(dir / "b.json", Json{{"id", "b"}, {"title", "edited beside it"}, {"messages", Json::array()},
                                           {"updated_at", 4}}
                                          .dump());
    list = store.list();
    check(list.size() == 2 && list[0]["title"] == "edited beside it", "a file replaced outside the store is read again");
    check(store.remove("a") && store.list().size() == 1 && store.get("b").has_value(), "a removed chat leaves the list");
    std::filesystem::remove_all(dir, ignored);
}

// With a second NVIDIA card the inference GPU is the one with the most VRAM, listed first (the
// dashboard shows the first, default launch parameters and load advice size from it); the others
// keep NVML's order, and equals keep it too.
void test_gpu_order() {
    const auto gpu = [](std::uint32_t index, const char* name, std::uint64_t gib) {
        GpuReading reading;
        reading.index      = index;
        reading.name       = name;
        reading.vram_total = gib << 30U;
        return reading;
    };
    check(gpu_listing_order({gpu(0, "NVIDIA GeForce RTX 3060", 12), gpu(1, "NVIDIA GeForce RTX 5090", 32)}) ==
              std::vector<std::size_t>{1, 0},
          "a larger second NVML device is listed first as the inference GPU");
    check(gpu_listing_order({gpu(0, "NVIDIA GeForce RTX 5090", 32), gpu(1, "NVIDIA GeForce RTX 3060", 12)}) ==
              std::vector<std::size_t>{0, 1},
          "the inference GPU as NVML device 0 keeps the order");
    check(gpu_listing_order({gpu(0, "NVIDIA T400", 2), gpu(1, "NVIDIA GeForce RTX 3060", 12),
                             gpu(2, "NVIDIA GeForce RTX 5090", 32)}) == std::vector<std::size_t>{2, 0, 1},
          "the other GPUs keep NVML's order behind the inference GPU");
    check(gpu_listing_order({gpu(0, "NVIDIA GeForce RTX 5090", 32), gpu(1, "NVIDIA GeForce RTX 5090", 32)}) ==
              std::vector<std::size_t>{0, 1},
          "equal GPUs keep NVML's order");
    check(gpu_listing_order({}).empty() && gpu_listing_order({gpu(0, "NVIDIA GeForce RTX 5090", 32)}) ==
                                               std::vector<std::size_t>{0},
          "no GPU or one GPU");
}

// /v1/models carries its page under data and again as a top-level models (Codex CLI's catalog).
void test_models_list_body() {
    const Json page = Json::array({Json{{"id", "a"}, {"object", "model"}}, Json{{"id", "b"}, {"object", "model"}}});
    const Json body = models_list_body(page, true);
    check(body["object"] == "list" && body["data"] == page && body["models"] == page,
          "the models list carries the same entries under data and models");
    check(body["has_more"] == true && body["first_id"] == "a" && body["last_id"] == "b",
          "the models list keeps Anthropic's paging fields");
    const Json empty = models_list_body(Json::array(), false);
    check(empty["data"].empty() && empty["models"].is_array() && empty["models"].empty() &&
              empty["first_id"].is_null() && empty["has_more"] == false,
          "an empty models list");
}

int main() {
    test_gpu_order();
    test_models_list_body();
    test_args();
    test_blob_ids();
    test_chats();
    test_records();
    test_help();
    test_help_cache();
    test_serve_log_files();
    test_library();
    test_scan_folder();
    test_router();
    test_load_advice();
    test_event_log();
    test_routing();
    test_launch_scripts();
    test_usage_book();
    test_release_versions();
    test_release_answer();
    test_exit_meaning();
    test_events();
    if (failures != 0) {
        std::cerr << failures << " launcher checks failed\n";
        return 1;
    }
    return 0;
}
