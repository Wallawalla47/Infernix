// infernix-launcher: launch-parameter edits, serve log and --help parsing, the --help cache, the
// server's log files, artifact, n-gram volume and LoRA adapter recognition, the Hub's adapter search,
// default launch parameters, the router's drain-then-switch order, the advice for loads that do not
// fit, the event log, GitHub's release answer and the exit-status wording.

#include "launcher/args.h"
#include "launcher/artifact_scan.h"
#include "launcher/chats.h"
#include "launcher/downloads.h"
#include "launcher/event_log.h"
#include "launcher/events.h"
#include "launcher/hf_hub.h"
#include "launcher/http_access.h"
#include "launcher/launch_script.h"
#include "launcher/launcher.h"
#include "launcher/load_advice.h"
#include "launcher/log_buffer.h"
#include "launcher/model_defaults.h"
#include "launcher/option_support.h"
#include "launcher/process.h"
#include "launcher/router.h"
#include "launcher/routing.h"
#include "launcher/serve_help.h"
#include "launcher/sha256.h"
#include "launcher/store.h"
#include "launcher/supervisor.h"
#include "launcher/telemetry.h"
#include "launcher/updates.h"
#include "launcher/usage.h"
#include "launcher/util.h"

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
#else
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
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
    check(flag_value({"--max-context", "auto", "--kv-dtype", "int8", "--max-context", "250000"}, "--max-context") ==
                  "250000" &&
              !context_auto({"--max-context", "auto", "--max-context", "250000"}) &&
              context_length({"--max-context=4096", "--max-context", "250000"}) == 250000,
          "a repeated flag's value is its last, as infernix-serve parses it");
    check(context_length({"--max-context", "240000"}) == 240000, "context_length reads --max-context");
    check(context_length({}) == 8192, "context_length falls back to infernix-serve's default");
    check(context_length({"--max-context", "auto"}) == 0 && context_auto({"--max-context", "auto"}) &&
              !context_auto({"--max-context", "240000"}) && !context_auto({}),
          "--max-context auto is the server's to size");

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

// The update hand-over's command line (audit 2026-11-18 F-5): cmd /d /s /c ""script"" runs a batch file from a folder
// whose name holds a space, parentheses and & (a plain /c "script" split it at the first space and ran nothing).
void test_batch_spawn() {
    check(windows_command_line("C:\\Program Files\\x.exe", {"a b", "c"}) == "\"C:\\Program Files\\x.exe\" \"a b\" c",
          "windows_command_line quotes the program and each argument");
    const std::string script = "C:\\Users\\Jo Bloggs (work) & co\\Infernix\\apply-update.bat";
    check(cmd_batch_command_line("C:\\Windows\\System32\\cmd.exe", script) ==
              "C:\\Windows\\System32\\cmd.exe /d /s /c \"\"" + script + "\"\"",
          "cmd_batch_command_line wraps the script in one outer quote pair for /s");
#ifdef _WIN32
    const std::filesystem::path folder =
        std::filesystem::temp_directory_path() / ("infernix launcher (spawn) & test " + std::to_string(GetCurrentProcessId()));
    std::filesystem::remove_all(folder);
    std::filesystem::create_directories(folder);
    const std::filesystem::path marker = folder / "ran.txt";
    {
        std::ofstream bat(folder / "apply-update.bat", std::ios::binary);
        bat << "@echo off\r\necho ran> \"%~dp0ran.txt\"\r\n";
    }
    wchar_t system[MAX_PATH] = {};
    const UINT length        = GetSystemDirectoryW(system, MAX_PATH);
    const std::filesystem::path cmd = std::filesystem::path(std::wstring(system, length)) / L"cmd.exe";
    const bool spawned = spawn_detached_command_line(
        cmd, cmd_batch_command_line(path_utf8(cmd), path_utf8(folder / "apply-update.bat")));
    const auto limit = std::chrono::steady_clock::now() + 10s;
    while (spawned && !std::filesystem::exists(marker) && std::chrono::steady_clock::now() < limit) {
        std::this_thread::sleep_for(20ms);
    }
    check(spawned && std::filesystem::exists(marker), "a batch file in a folder with ( ) & did not run");
    std::this_thread::sleep_for(200ms); // cmd has closed the file
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
#endif
}

// Host and Origin checks of the management API (audit 2026-11-18 F-1): a DNS name that begins "127." is not this
// computer, so a DNS-rebound page on 127.evil.com is refused in Host and in Origin.
void test_http_access() {
    check(!loopback_host("127.evil.com"), "127.evil.com is not loopback");
    check(!loopback_host("127.0.0.1.evil.com"), "a dotted name past an address is not loopback");
    check(loopback_host("127.0.0.1") && loopback_host("127.1.2.3") && loopback_host("localhost") &&
              loopback_host("a.localhost") && loopback_host("::1") && loopback_host("::ffff:127.0.0.1"),
          "numeric 127/8, localhost and its subdomains, ::1 and mapped 127 are loopback");
    check(authority_host("127.evil.com:8000") == "127.evil.com" && authority_host("[::1]:80") == "::1",
          "authority_host drops the port and the brackets");
    check(page_refusal("127.evil.com:8000", "", false).has_value(), "Host 127.evil.com is refused");
    check(page_refusal("127.0.0.1:8000", "http://127.evil.com:8000", false).has_value(),
          "Origin 127.evil.com is refused beside a numeric Host");
    check(!page_refusal("127.0.0.1:8000", "http://127.0.0.1:8000", false) &&
              !page_refusal("localhost:8000", "http://localhost:8000", false) &&
              !page_refusal("[::1]:8000", "", false),
          "the launcher's own page by address or localhost is allowed");
    check(!page_refusal("box.lan:8000", "", true) && page_refusal("box.lan:8000", "", false).has_value(),
          "a DNS Host needs an admin key");
    check(page_refusal("127.0.0.1:8000", "http://evil.example", true).has_value(),
          "another page's Origin is refused even with an admin key");
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
        "                             speculative decoding\n"
        "\n"
        "NOTES\n"
        "  --reproducible forces the reproducible kernels.\n";
    const ServeOptions serve{.found = true, .options = parse_serve_help(help)};
    const std::vector<ServeOption>& options = serve.options;
    check(options.size() == 5, "five options parsed (a NOTES line naming one again is not another)");
    check(serve.takes_value("--reproducible") == false, "the first listing gives the arity");
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
              "serve-<time>-<run>-<model>.log");
        check(read_file(files[0]).value_or("") == "2026-10-30 11:55:04.605  INFO  build windows-2026.10.18\nraw line\n",
              "an instance's output as it came, in its own file");
        check(read_file(files[1]).value_or("") == "second output\n", "the next instance's in the next file");
        logs.append_output(2, "more\n");
        logs.write_files();
        check(read_file(files[1]).value_or("") == "second output\nmore\n", "later output joins the open file");
    }
    logs.end_instance();
    // Another start of that model straight away (audit 2026-11-17 LN-15: a retry on another port, within the same
    // second) writes a file of its own instead of truncating the log of the start it retries.
    logs.begin_instance(3, "second");
    logs.append_output(3, "retried\n");
    logs.end_instance();
    logs.write_files();
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ignored)) {
        (void)entry;
        ++count;
    }
    check(count == 3 && files.size() == 2 && read_file(files[1]).value_or("") == "second output\nmore\n",
          "a restart of the same model keeps the previous run's log");
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
    const auto q8   = parse_ngram_header("q8.ngram", volume_header(170, other, 0, 5));
    const auto p12  = parse_ngram_header("p12.ngram", volume_header(256, other, 0, 6));
    const auto fp8_unscaled = parse_ngram_header("fp8.ngram", volume_header(160, other, 0, 1));
    check(own && iq4 && bf16 && q8 && p12 && fp8_unscaled, "n-gram headers parse");
    check(!parse_ngram_header("x", std::string(68, 'x')), "other files are not n-gram volumes");
    if (own && iq4 && bf16 && q8 && p12 && fp8_unscaled) {
        check(iq4->format == "iq4_nl" && bf16->format == "bf16" && q8->format == "q8_0" &&
                  p12->format == "bf16_p12" && own->format == "fp8",
              "formats from tag or row size");
        check(ngram_rank(model, *own) == 0, "the artifact's own volume ranks first");
        check(ngram_rank(model, *iq4) == 3 && ngram_rank(model, *bf16) == 6, "IQ4_NL before BF16");
        check(ngram_rank(model, *q8) == 4 && ngram_rank(model, *p12) == 5, "Q8_0 and BF16 P12 before BF16");
        check(!ngram_rank(model, *fp8_unscaled), "an FP8 volume without a scale serves only its own artifact");

        // An artifact bound to a Q8_0 table (a GGUF or .basalt import): its own Q8_0 volume first, then the
        // other formats of its width (160 values, 170-byte Q8_0 rows).
        Json q8_directory = flash_next_directory();
        q8_directory["components"]["text"]["config"]["ngram_table"] = {
            {"format", "q8_0"}, {"rows", 320001536}, {"row_bytes", 170}, {"volume_id", "07070707070707070707070707070707"}};
        const ParsedEntry q8_parsed = parse_artifact_directory(entry, id, q8_directory, 4096);
        check(q8_parsed.model && q8_parsed.model->ngram_table && q8_parsed.model->ngram_table->format == "q8_0",
              "a Q8_0 table is an artifact's own format");
        if (q8_parsed.model) {
            const ScannedModel& q8_model = *q8_parsed.model;
            check(ngram_rank(q8_model, *q8) == 0, "a Q8_0 artifact's own volume ranks first");
            check(ngram_rank(q8_model, *iq4) == 3 && ngram_rank(q8_model, *p12) == 5 && ngram_rank(q8_model, *bf16) == 6,
                  "a Q8_0 artifact takes the other formats of its width");
        }
        // BF16 P12 is a run-time volume only: an artifact naming it as its own table is not recognised.
        Json p12_directory = flash_next_directory();
        p12_directory["components"]["text"]["config"]["ngram_table"] = {
            {"format", "bf16_p12"}, {"rows", 320001536}, {"row_bytes", 256}, {"volume_id", "07070707070707070707070707070707"}};
        const ParsedEntry p12_parsed = parse_artifact_directory(entry, id, p12_directory, 4096);
        check(p12_parsed.model && p12_parsed.model->ngram_table && p12_parsed.model->ngram_table->format == "unknown" &&
                  !ngram_rank(*p12_parsed.model, *p12),
              "no artifact binds a BF16 P12 table");
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
    const Args args = default_args(scan, model);
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
    const Args dense_args = default_args(scan, dense);
    check(contains(dense_args, {"--ngram-draft-tokens", "auto"}) && contains(dense_args, {"--prefill-chunk", "4096"}),
          "Qwen3.8-27B defaults follow the README's recommended settings");
    check(!has_flag(dense_args, "--host-context-mib"), "the Engine sizes every model's Host tier (auto)");
    check(!has_flag(dense_args, "--draft-model"), "bundled drafters take no --draft-model");
    // Every KV page stays on the GPU: the server sizes the context to what its KV cache holds.
    check(contains(dense_args, {"--max-context", "auto"}), "the 27B context is the server's to size");
}

// Which options each model type takes (option_support.h): the table, the default arguments and a saved configuration.
void test_option_support() {
    const auto kind = [](const char* type, bool offloaded = false) {
        ScannedModel m;
        m.key               = type;
        m.model_type        = type;
        m.family_label      = type;
        m.offloaded_experts = offloaded;
        return m;
    };
    const ScannedModel dense = kind("qwen3_5_text"), moe = kind("qwen3_5_moe_text"), moe_off = kind("qwen3_5_moe_text", true);
    ScannedModel next         = kind("qwen4_exp_text", true);
    next.accepts_draft_model  = true;
    ScannedModel inkling = kind("inkling2_loop_text");
    inkling.family       = "inkling2-mini";
    inkling.num_experts  = 128;
    inkling.has_dspark   = true;
    check(model_kind(dense) == ModelKind::Qwen35Dense && model_kind(moe) == ModelKind::Qwen35Moe &&
              model_kind(moe_off) == ModelKind::Qwen35MoeOffloaded && model_kind(next) == ModelKind::FlashNext &&
              model_kind(inkling) == ModelKind::Inkling2 && model_kind(kind("other")) == ModelKind::Unknown,
          "the model kinds");
    // One row each of what the Engine accepts and refuses.
    const auto row = [&](const char* flag, bool d, bool m, bool o, bool f, bool i) {
        check(supports_option(dense, flag) == d && supports_option(moe, flag) == m &&
                  supports_option(moe_off, flag) == o && supports_option(next, flag) == f &&
                  supports_option(inkling, flag) == i,
              (std::string("option support: ") + flag).c_str());
    };
    row("--lora", true, false, false, false, false);
    row("--ablate", true, true, true, true, false);
    row("--q8-weights", false, false, false, false, true);
    row("--expert-state", false, false, true, true, false);
    row("--expert-tail-skip", false, false, true, true, false);
    row("--embedding-host", true, true, false, false, true);
    row("--rope-yarn-factor", true, true, true, true, false);
    row("--logits-dtype", true, true, true, true, false);
    row("--cache-host-state", true, true, true, false, false);
    row("--device-snapshot-slots", true, true, true, false, false);
    row("--cache-next-turn-ms", true, true, true, true, false);
    row("--draft-model", false, false, false, true, false);
    row("--draft-tree-nodes", true, true, true, false, false);
    row("--ngram-draft-tokens", true, true, true, true, false);
    row("--ngram-archive-mib", true, true, true, false, false);
    row("--ngram-volume", false, false, false, true, false);
    row("--preserve-thinking", true, true, true, true, false);
    row("--no-cuda-graph", true, true, true, false, true);
    row("--fast-path", true, true, true, true, false);
    row("--max-context", true, true, true, true, true);
    row("--temperature", true, true, true, true, true);
    row("--vision", false, false, false, false, false); // none of these holds a vision component
    check(!supports_option(inkling, "--lm-head-draft"), "--lm-head-draft needs Inkling's proposal/token_rank");
    ScannedModel ranked = inkling;
    ranked.has_proposal = true;
    check(supports_option(ranked, "--lm-head-draft"), "an Inkling artifact with its ranking takes --lm-head-draft");
    check(!supports_context_auto(next) && !supports_context_auto(moe_off) && supports_context_auto(dense) &&
              supports_context_auto(inkling),
          "--max-context auto where experts stay in VRAM");
    const std::vector<std::string> specs = spec_values(inkling);
    check(specs == std::vector<std::string>{"off", "auto", "dspark"}, "Inkling's --spec values");

    // A saved configuration: what the model does not take is gone, and its neighbours and values stay.
    Args saved = {"--kv-dtype", "int8", "--preserve-thinking", "--ngram-draft-tokens", "12", "--spec", "mtp",
                  "--ablate",   "x.json", "--rope-yarn-factor=1.5", "--temperature", "0.7"};
    const std::vector<std::string> removed = strip_unsupported_options(saved, inkling);
    check(saved == Args{"--kv-dtype", "int8", "--temperature", "0.7"}, "unsupported options leave the list");
    check(removed.size() == 5 && removed.back() == "--spec", "the removed options are named");
    Args auto_context = {"--max-context", "auto", "--spec", "dflash2"};
    check(strip_unsupported_options(auto_context, next).size() == 1 && !has_flag(auto_context, "--max-context") &&
              has_flag(auto_context, "--spec"),
          "Flash-Next refuses --max-context auto and takes dflash2");
    Args untouched = {"--anything", "1"};
    check(strip_unsupported_options(untouched, kind("other")).empty() && untouched.size() == 2,
          "a model type the launcher does not know is not restricted");

    // Inkling's defaults: auto context and DSpark, nothing the model does not take.
    LibraryScan scan;
    Args defaults = default_args(scan, inkling);
    check(contains(defaults, {"--max-context", "auto"}), "Inkling's context is the server's to size");
    check(contains(defaults, {"--spec", "auto"}) && contains(defaults, {"--draft-tokens", "auto"}), "Inkling drafts with DSpark");
    check(!has_flag(defaults, "--lm-head-draft"), "the ranking is offered, not defaulted");
    check(strip_unsupported_options(defaults, inkling).empty(), "Inkling's defaults hold only what it takes");
    Json library_row = option_support_json(inkling);
    check(library_row["context_auto"].get<bool>() && library_row["unsupported"].is_array(), "the form's view");
}

// A part file an artifact header names outside the artifact's folder (audit 2026-11-17 LN-10: an absolute path or
// "..") is reported missing and is not one of the model's files, which "Delete model files" deletes.
void test_library_part_paths() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "infernix-launcher-test-parts";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    const std::filesystem::path models = root / "models";
    std::filesystem::create_directories(models);
    const std::filesystem::path outside = root / "outside.bin";
    { std::ofstream(outside, std::ios::binary) << std::string(4096 + 100, 'x'); }
    { std::ofstream(models / "Next-00002-of-00002.infernix", std::ios::binary) << std::string(4096 + 2000, 'x'); }
    Json directory     = flash_next_directory();
    directory["files"] = Json::array({{{"path", nullptr}, {"payload_bytes", 1000}},
                                      {{"path", "Next-00002-of-00002.infernix"}, {"payload_bytes", 2000}},
                                      {{"path", "../outside.bin"}, {"payload_bytes", 100}},
                                      {{"path", path_utf8(outside)}, {"payload_bytes", 100}}});
    const std::string text = directory.dump();
    std::string entry(32, '\0');
    std::memcpy(entry.data(), "NINFER\0\x03", 8);
    for (int i = 0; i < 8; ++i) { entry[8 + i] = static_cast<char>((text.size() >> (8 * i)) & 0xFF); }
    for (int i = 0; i < 16; ++i) { entry[16 + i] = static_cast<char>(i + 1); }
    entry += text;
    entry.resize((entry.size() + 4095) / 4096 * 4096 + 1000, '\0');
    { std::ofstream(models / "Next-00001-of-00002.infernix", std::ios::binary) << entry; }

    const LibraryScan scan = scan_library({models}, 2);
    const auto model = std::find_if(scan.models.begin(), scan.models.end(),
                                    [](const ScannedModel& m) { return m.key == "0102030405060708090a0b0c0d0e0f10"; });
    check(model != scan.models.end(), "the test artifact is scanned");
    if (model != scan.models.end()) {
        const bool kept_outside = std::any_of(model->part_files.begin(), model->part_files.end(),
                                              [&](const std::filesystem::path& p) { return p.filename() == "outside.bin"; });
        const auto outside_noted = std::count_if(model->missing_files.begin(), model->missing_files.end(),
                                                 [](const std::string& f) { return f.find("outside") != std::string::npos; });
        check(model->part_files.size() == 1 && !kept_outside, "only the part inside the folder is a model file");
        check(outside_noted == 2, "both parts outside the folder are reported missing");
    }
    std::filesystem::remove_all(root, ignored);
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

    // Waits (up to 30 s) for what another thread does next, rather than sleeping a fixed time a loaded machine can
    // outlast before that thread runs.
    const auto settle = [](const auto& done) {
        const auto until = std::chrono::steady_clock::now() + 30s;
        while (!done() && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(2ms); }
        return done();
    };
    std::shared_ptr<Router::Lease> b, a3;
    std::thread wants_b([&] { b = router.acquire("b", nullptr, refusal); });
    check(settle([&] { return router.pending_model() == "b"; }), "the switch is asked for");
    // A request for the loaded model arriving after the switch was asked for waits its turn.
    std::thread wants_a([&] { a3 = router.acquire("a", nullptr, refusal); });
    std::this_thread::sleep_for(300ms); // time for a wrongly admitted request or an early load to show
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
    check(settle([&] {
              const std::vector<std::string> forced = snapshot();
              return !forced.empty() && forced.back() == "load:d";
          }),
          "a forced load does not wait for running requests");
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
                          return s.args && flag_value(*s.args, "--ram-free-mib") == "12288" &&
                                 flag_value(*s.args, "--expert-ram-mib") == "auto";
                      }),
          "more RAM left free with automatic expert sizing");
    check(std::any_of(ram.begin(), ram.end(),
                      [](const LoadSuggestion& s) { return s.args && flag_value(*s.args, "--host-context-mib") == "4096"; }),
          "a smaller Host prefix cache");
}

void check_event_log(const std::filesystem::path& dir) {
    std::error_code ignored;
    EventLog log(dir);
    std::vector<std::string> heard;
    log.set_listener([&](const EventRecord& record) { heard.push_back(record.message); });
    log.add("info", "launcher", "first");
    log.add("error", "watchdog", "second line\nwith a break", Json{{"kind", "crash"}});
    const Json page = log.page(0, 10);
    check(page["lines"].size() == 2 && page["lines"][1]["fields"]["kind"] == "crash", "records keep their fields");
    check(log.page(1, 10)["lines"].size() == 1, "a page after a sequence number");
    check(heard == std::vector<std::string>{"first", "second line\nwith a break"}, "the listener hears every record");
    // A record is kept and queued; the file is written by write_files, on the launcher's log thread (audit 2026-11-17
    // P4D-02: a slow log folder held up every thread that logged, the request threads among them).
    check(read_file(dir / EventLog::kEventsFile).value_or("x").empty(), "logging a record writes nothing");
    log.write_files();
    const std::optional<std::string> text = read_file(dir / EventLog::kEventsFile);
    check(text && text->find("  INFO   launcher  first\n") != std::string::npos, "the file has one text line per record");
    check(text && text->find("ERROR  watchdog  second line with a break\n") != std::string::npos, "one record, one line");
    log.add_request_body({{"id", 1}});
    log.write_files();
    check(read_file(dir / EventLog::kRequestsFile) == std::optional<std::string>("{\"id\":1}\n"), "request bodies, one JSON line each");

    const std::filesystem::path moved = dir / "moved";
    std::string error;
    log.add("info", "settings", "queued before the move");
    check(log.set_dir(moved, error), "the log folder can change");
    log.add("info", "settings", "third");
    log.write_files();
    check(read_file(moved / EventLog::kEventsFile).value_or("").find("third") != std::string::npos, "later records go there");
    check(read_file(dir / EventLog::kEventsFile).value_or("").find("queued before the move") != std::string::npos,
          "a record queued before the move is written where it was logged");
    check(log.page(0, 10)["lines"].size() == 4, "the memory keeps the earlier records");
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
        log.write_files();
        const std::string head = read_file(held).value_or("").substr(0, 14);
        check(head == "oldest record\n" && std::filesystem::file_size(held, ignored) > EventLog::kEventsLimit &&
                  !std::filesystem::exists(held_dir / (std::string(EventLog::kEventsFile) + ".1")),
              "a file that cannot be moved keeps its records");
        if (tail != INVALID_HANDLE_VALUE) { CloseHandle(tail); }
        const std::string block(64 << 10, 'y');
        for (int i = 0; i < 20; ++i) { log.add("info", "launcher", block); }
        log.write_files();
        check(read_file(held_dir / (std::string(EventLog::kEventsFile) + ".1")).value_or("").starts_with("oldest record\n") &&
                  std::filesystem::file_size(held, ignored) < EventLog::kEventsLimit,
              "a later try rotates it once it is free");
    }
#endif
}

void test_event_log() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-events";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    check_event_log(dir);
    // After the log, which holds its files open (Windows deletes no open file).
    std::filesystem::remove_all(dir, ignored);
    check(!std::filesystem::exists(dir), "the event log test leaves no files behind");
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

// Ablation sidecars (--ablate) in the model folders: an ablate-direction-v2 sidecar.json with its direction file, the
// models whose geometry it fits and the one it was fitted on; a base direction (no cut), another program's
// sidecar.json and a tools/ablate run's trial sidecars are not listed.
void test_library_sidecars() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "infernix-launcher-test-sidecars";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    const auto write = [](const std::filesystem::path& path, const std::string& text) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << text;
    };
    const std::string id = "00112233445566778899aabbccddeeff";
    const Json cut = {{"format", "ablate-direction-v2"},
                      {"direction_dtype", "float32"},
                      {"direction_dim", 4},
                      {"direction_file", "d.f32.bin"},
                      {"default_alpha", 0.0},
                      {"alpha_attn_per_layer", {{"1", 1.0}, {"2", 0.0}}},
                      {"alpha_mlp_per_layer", {{"3", 0.5}}},
                      {"artifact_id", id},
                      {"generation", {{"source", "test prompts"}, {"fitted_under", {{"artifact", "M-00001-of-00001.infernix"}}}}}};
    write(root / "refusal-cut" / "sidecar.json", cut.dump());
    write(root / "refusal-cut" / "d.f32.bin", std::string(16, '\0'));
    Json base = cut;
    base["alpha_attn_per_layer"] = Json::object();
    base["alpha_mlp_per_layer"]  = Json::object();
    write(root / "base" / "sidecar.json", base.dump());
    write(root / "base" / "d.f32.bin", std::string(16, '\0'));
    Json short_file = cut;
    short_file["direction_layout"] = "per_layer";
    write(root / "short" / "sidecar.json", short_file.dump());
    write(root / "short" / "d.f32.bin", std::string(10, '\0'));
    write(root / "other-tool" / "sidecar.json", Json{{"format", "something-else"}}.dump());
    // A tools/ablate run directory: its trial sidecars stay out of the library, its export does not.
    write(root / "run" / "run.json", Json{{"schema", "infernix-ablate-run-2"}}.dump());
    write(root / "run" / "fit" / "s1" / "t0" / "sidecar.json", cut.dump());
    write(root / "run" / "fit" / "s1" / "t0" / "d.f32.bin", std::string(16, '\0'));
    write(root / "run" / "export" / "share" / "sidecar.json", cut.dump());
    write(root / "run" / "export" / "share" / "d.f32.bin", std::string(16, '\0'));

    const LibraryScan scan = scan_library({root}, 5);
    std::vector<std::string> names;
    for (const ScannedSidecar& sidecar : scan.sidecars) { names.push_back(sidecar.name); }
    check(names == std::vector<std::string>{"refusal-cut", "share"},
          "a sidecar that cuts, and a run's export, are listed; a base direction, another file and trials are not");
    check(scan.dirs.size() == 1 && scan.dirs[0].sidecars == 2, "the folder counts its sidecars");
    check(std::any_of(scan.issues.begin(), scan.issues.end(),
                      [](const ScanIssue& issue) { return issue.path.parent_path().filename() == "short"; }),
          "a direction file of the wrong size is a scan issue");
    check(std::none_of(scan.issues.begin(), scan.issues.end(),
                       [](const ScanIssue& issue) { return issue.path.parent_path().filename() == "other-tool"; }),
          "another program's sidecar.json is no issue");
    if (!scan.sidecars.empty()) {
        const ScannedSidecar& sidecar = scan.sidecars[0];
        check(sidecar.direction_dim == 4 && sidecar.rows == 1 && sidecar.attn_layers == 1 && sidecar.mlp_layers == 1 &&
                  sidecar.layer_bound == 4 && sidecar.bytes == 16 && sidecar.artifact_id == id &&
                  sidecar.fitted_artifact == "M-00001-of-00001.infernix" && sidecar.source == "test prompts",
              "a sidecar's geometry, cut layers and provenance");
        const Json json = to_json(scan)["sidecars"];
        check(json.size() == 2 && json[0]["direction_dim"] == 4 && json[0]["artifact_id"] == id,
              "the library lists the sidecars");

        ScannedModel model;
        model.key         = id;
        model.hidden_size = 4;
        model.num_layers  = 4;
        check(sidecar_fits(model, sidecar), "a sidecar fits a model of its hidden size and layers");
        ScannedModel fewer = model;
        fewer.num_layers   = 3;
        ScannedModel wider = model;
        wider.hidden_size  = 8;
        check(!sidecar_fits(fewer, sidecar) && !sidecar_fits(wider, sidecar),
              "not one with fewer layers than its tables name, nor another hidden size");
        ScannedSidecar stack = sidecar;
        stack.rows           = 2;
        check(!sidecar_fits(model, stack), "a per-layer stack needs one row a layer");
    }
    std::filesystem::remove_all(root, ignored);
}

// LoRA adapters (--lora) in the model folders: a folder with adapter_config.json and adapter_model.safetensors, the
// models of the base it names, and the Hub's adapter search.
void test_library_adapters() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "infernix-launcher-test-adapters";
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    const auto write = [](const std::filesystem::path& path, const std::string& text) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary) << text;
    };
    const Json pi_config = {{"peft_type", "LORA"},
                            {"base_model_name_or_path", "Qwen/Qwen3.8-27B"},
                            {"r", 32},
                            {"lora_alpha", 64},
                            {"rank_pattern", {{"model.layers.3.mlp.down_proj", 64}}},
                            {"target_modules", Json::array({"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})},
                            {"use_rslora", false}};
    write(root / "Qwen3.8-27B-pi-adapted" / "adapter_config.json", pi_config.dump());
    write(root / "Qwen3.8-27B-pi-adapted" / "adapter_model.safetensors", std::string(100, 'x'));
    // A GGUF adapter's folder, or a download still running: no weights, so no adapter.
    write(root / "gguf-only" / "adapter_config.json", pi_config.dump());
    write(root / "gguf-only" / "adapter.gguf", std::string(10, 'x'));
    write(root / "broken" / "adapter_config.json", "{not json");
    write(root / "broken" / "adapter_model.safetensors", std::string(10, 'x'));

    const LibraryScan scan = scan_library({root}, 2);
    check(scan.adapters.size() == 1 && scan.dirs.size() == 1 && scan.dirs[0].adapters == 1,
          "an adapter folder needs both files");
    if (scan.adapters.size() == 1) {
        const ScannedAdapter& adapter = scan.adapters[0];
        check(adapter.name == "Qwen3.8-27B-pi-adapted" && adapter.bytes == 100 &&
                  path_utf8(adapter.path.filename()) == adapter.name,
              "an adapter is its folder, sized by its weights");
        check(adapter.config.base_model == "Qwen/Qwen3.8-27B" && adapter.config.rank == 32u &&
                  adapter.config.max_rank == 64u && adapter.config.alpha == 64.0 &&
                  adapter.config.target_modules.size() == 7 && !adapter.config.rslora,
              "the config's base model, ranks, alpha and modules");
        const Json json = to_json(scan)["adapters"];
        check(json.size() == 1 && json[0]["config"]["max_rank"] == 64 && json[0]["config"]["base_model"] == "Qwen/Qwen3.8-27B",
              "the library lists the adapter with its config");
    }
    check(std::any_of(scan.issues.begin(), scan.issues.end(),
                      [](const ScanIssue& issue) { return issue.path.parent_path().filename() == "broken"; }),
          "an unreadable config is a scan issue");

    ScannedModel qwen38;
    qwen38.key         = "38";
    qwen38.model_type  = "qwen3_5_text";
    qwen38.hidden_size = 5120;
    qwen38.family      = "qwen3.8-27b";
    ScannedModel qwen36 = qwen38;
    qwen36.key          = "36";
    qwen36.family       = "qwen3.6-27b";
    ScannedModel moe    = qwen38;
    moe.model_type      = "qwen3_5_moe_text";
    moe.num_experts     = 256;
    moe.family          = "qwen3.6-35b-a3b";
    ScannedAdapter adapter;
    adapter.config.base_model = "unsloth/Qwen3.8-27B-bnb-4bit";
    check(adapter_matches(qwen38, adapter), "an adapter of a Qwen3.8-27B repackaging fits Qwen3.8-27B");
    check(!adapter_matches(qwen36, adapter) && !adapter_matches(moe, adapter), "nor another base or a MoE model");
    adapter.config.base_model.reset();
    check(!adapter_matches(qwen38, adapter), "an adapter that names no base fits no model");
    // Inkling 2 Mini refuses --lora, so no adapter fits it.
    ScannedModel inkling = qwen38;
    inkling.model_type   = "inkling2_loop_text";
    inkling.num_experts  = 128;
    inkling.family       = "inkling2-mini";
    adapter.config.base_model = "thinkingmachines/Inkling-2-Mini";
    check(!adapter_matches(inkling, adapter), "an adapter does not fit Inkling");
    std::filesystem::remove_all(root, ignored);

    // The Hub's model list: only repositories with both files at their root.
    const Json models = Json::array(
        {{{"id", "harness-hub/Qwen3.8-27B-pi-adapted"},
          {"downloads", 2383},
          {"likes", 40},
          {"lastModified", "2026-11-21T04:31:16.000Z"},
          {"gated", false},
          {"siblings", Json::array({{{"rfilename", "adapter_config.json"}}, {{"rfilename", "adapter_model.safetensors"}}})}},
         {{"id", "someone/gguf-lora"},
          {"siblings", Json::array({{{"rfilename", "adapter_config.json"}}, {{"rfilename", "lora.gguf"}}})}},
         {{"id", "someone/checkpoints"},
          {"siblings",
           Json::array({{{"rfilename", "step-100/adapter_config.json"}}, {{"rfilename", "step-100/adapter_model.safetensors"}}})}},
         {{"id", "someone/gated-lora"},
          {"gated", "manual"},
          {"siblings", Json::array({{{"rfilename", "adapter_model.safetensors"}}, {{"rfilename", "adapter_config.json"}}})}}});
    const std::vector<HfAdapter> found = parse_adapter_search(models.dump());
    check(found.size() == 2 && found[0].repo == "harness-hub/Qwen3.8-27B-pi-adapted" && found[0].downloads == 2383 &&
              !found[0].gated && found[1].repo == "someone/gated-lora" && found[1].gated,
          "the Hub search keeps the adapters --lora reads, in order, with their gating");
    bool refused = false;
    try {
        (void)parse_adapter_search("{\"error\": \"x\"}");
    } catch (const std::runtime_error&) { refused = true; }
    check(refused, "an answer that is not a list is an error");
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

// An unload, or a forced load of another model, while a model loads ends that load at once instead of once it is
// ready (up to the load timeout; audit 2026-11-17 LN-03): the router calls cancel_load and load_cancelled turns
// true for the load in progress. A cancelled load holds nothing: its model loads at its next request.
void test_router_cancel() {
    std::atomic<bool> cancelled{false};
    std::atomic<int> loads{0};
    Router* routed = nullptr;
    Router router(Router::Hooks{
        .load =
            [&](const std::string& key) {
                ++loads;
                Router::Loaded loaded;
                if (key == "slow") { // a long load, until the router cancels it (10 s at most)
                    const auto until = std::chrono::steady_clock::now() + 10s;
                    while (!routed->load_cancelled() && std::chrono::steady_clock::now() < until) {
                        std::this_thread::sleep_for(5ms);
                    }
                    if (routed->load_cancelled()) {
                        loaded.error     = "cancelled";
                        loaded.cancelled = true;
                        return loaded;
                    }
                }
                loaded.instance = static_cast<std::uint32_t>(loads.load());
                loaded.port     = 2000 + loads.load();
                return loaded;
            },
        .cancel_load     = [&] { cancelled = true; },
        .stop            = [] {},
        .changed         = [] {},
        .drain_timeout_s = [] { return 60; },
        .load_timeout_s  = [] { return 30; },
        .idle_unload_min = [](const std::string&) { return 0; },
    });
    routed = &router;
    const auto wait_for = [](const auto& done) {
        const auto until = std::chrono::steady_clock::now() + 5s;
        while (!done() && std::chrono::steady_clock::now() < until) { std::this_thread::sleep_for(5ms); }
        return done();
    };
    router.request_load("slow", false);
    check(wait_for([&] { return router.loading_model() == std::optional<std::string>("slow"); }), "the slow load starts");
    const auto unload_at = std::chrono::steady_clock::now();
    router.request_unload();
    check(wait_for([&] { return !router.loading_model().has_value(); }) && cancelled &&
              std::chrono::steady_clock::now() - unload_at < 2s,
          "an unload ends the load in progress at once");
    Router::Refusal peek;
    check(!router.held("slow", peek), "a cancelled load holds nothing");

    cancelled = false;
    router.request_load("slow", false);
    check(wait_for([&] { return router.loading_model() == std::optional<std::string>("slow"); }),
          "the slow load starts again (no hold)");
    router.request_load("fast", true);
    Router::Refusal refusal;
    auto lease = router.acquire("fast", nullptr, refusal);
    check(cancelled && lease && lease->port() > 2000, "a forced load of another model ends the load in progress");
    lease.reset();
    router.request_load("fast", true); // the model loaded: nothing to cancel
    check(!router.load_cancelled(), "a forced load of the loaded model cancels nothing");
}

// A cancel that races a failure (audit 2026-11-18 F-7): the job the user cancelled stays cancelled (and is not
// reported failed), whatever failure the worker meets after the cancel. Here the job's folder cannot be created (it
// lies under a file) and the cancel lands as the job starts, before run_job fails it.
void test_download_cancel_race() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-cancel";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    std::filesystem::create_directories(dir);
    { std::ofstream(dir / "blocker") << "a file, so no folder under it"; }
    std::atomic<Downloads*> self{nullptr};
    std::atomic<bool> cancelled{false};
    std::atomic<int> failures_reported{0};
    Downloads::Hooks hooks;
    hooks.load_state = [&] {
        return Json::array({{{"id", "job"}, {"repo", "org/model"}, {"state", "queued"},
                             {"target_dir", path_utf8(dir / "blocker" / "model")},
                             {"files", Json::array({{{"path", "weights.bin"}, {"bytes", 1}}})}}});
    };
    hooks.failed  = [&](const std::string&, const std::string&) { ++failures_reported; };
    hooks.changed = [&](const Json& jobs) {
        for (const Json& job : jobs) {
            if (job.value("id", "") != "job" || job.value("state", "") != "running" || cancelled.exchange(true)) {
                continue;
            }
            const auto limit = std::chrono::steady_clock::now() + 10s;
            while (self.load() == nullptr && std::chrono::steady_clock::now() < limit) { std::this_thread::yield(); }
            if (Downloads* downloads = self.load()) { downloads->cancel("job"); }
        }
    };
    std::string state;
    {
        Downloads downloads(hooks);
        self.store(&downloads);
        const auto limit = std::chrono::steady_clock::now() + 10s;
        do {
            std::this_thread::sleep_for(20ms);
            const Json jobs = downloads.list();
            state           = jobs.empty() ? std::string() : jobs[0].value("state", "");
        } while ((state == "queued" || state == "running" || !cancelled.load()) && std::chrono::steady_clock::now() < limit);
        downloads.shutdown();
    }
    check(cancelled.load() && state == "cancelled" && failures_reported.load() == 0,
          ("a cancel racing a failure left the job " + state).c_str());
    std::filesystem::remove_all(dir, ignored);
}

// A download error in the system's code page (audit 2026-11-18 F-6: std::error_code::message() on a non-English
// Windows) is not valid UTF-8; the job holding it must not stop every later save of the memory file.
void test_store_invalid_utf8() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-utf8";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    {
        Store store(dir);
        store.set_downloads(Json::array({{{"id", "job"}, {"state", "failed"},
                                          {"error", std::string("cannot rename x: Zugriff verweigert \xE4")}}}));
        store.set_last_model(std::string("after"));
        check(!store.take_save_failure().has_value(), "a download error that is not UTF-8 failed the save");
    }
    Store store(dir);
    check(store.last_model() == std::optional<std::string>("after"),
          "a change saved after a download error that is not UTF-8 was lost");
    std::filesystem::remove_all(dir, ignored);
}

// The memory file survives a torn write (audit 2026-11-17 LN-01): each save keeps the previous file as its backup,
// and a memory file that is not valid JSON (a power loss after the rename, before the data reached the disk) or is
// missing (a crash between the backup's rename and the new file's) loads the backup instead of defaults. The
// temporary files of the atomic write leave nothing behind.
void test_store_durability() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "infernix-launcher-test-store";
    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    const std::filesystem::path memory = dir / Store::kMemoryFile;
    {
        Store store(dir);
        store.set_last_model(std::string("first"));
        store.set_last_model(std::string("second"));
    }
    check(std::filesystem::exists(backup_path(memory)), "a save keeps the previous memory file as its backup");
    bool temporary_left = false;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        temporary_left = temporary_left || entry.path().filename().string().find(".tmp") != std::string::npos;
    }
    check(!temporary_left, "the atomic write leaves no temporary file");
    {
        std::ofstream torn(memory, std::ios::binary | std::ios::trunc);
        torn << "{\"schema\": 1, \"last_mo";
    }
    {
        Store store(dir);
        check(store.last_model() == std::optional<std::string>("first"),
              "a torn memory file loads the backup (the save before the last)");
    }
    std::filesystem::remove(memory, ignored);
    {
        Store store(dir);
        check(store.last_model().has_value(), "a missing memory file loads the backup");
    }
    // A file read while it is replaced (audit 2026-11-17 LN-09: a chat the UI reads while the next message's save
    // renames over it): the read shares delete, so the replace is never refused, and the read gets one whole version.
    const std::filesystem::path chat = dir / "chat.json";
    write_file_atomic(chat, std::string(1 << 18, 'a'));
    std::atomic<bool> writing{true};
    std::atomic<int> torn_reads{0};
    std::thread reader([&] {
        while (writing) {
            if (const std::optional<std::string> text = read_file(chat);
                text && (text->size() != (1U << 18) || text->find_first_not_of(text->front()) != std::string::npos)) {
                ++torn_reads;
            }
        }
    });
    int refused = 0;
    for (int i = 0; i < 100; ++i) {
        try {
            write_file_atomic(chat, std::string(1 << 18, static_cast<char>('a' + i % 26)));
        } catch (const std::exception&) { ++refused; }
    }
    writing = false;
    reader.join();
    if (refused != 0) { std::cerr << refused << " of 100 replaces refused\n"; }
    check(refused == 0, "an atomic replace is not refused while the file is read");
    check(torn_reads == 0, "a read during replaces gets one whole version");
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
#ifdef __linux__
// Whether `pid` is a running process (a zombie, killed and not yet reaped, is not).
bool running(pid_t pid) {
    std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
    std::string line;
    if (!std::getline(stat, line)) { return false; }
    const std::size_t close = line.rfind(')');
    return close != std::string::npos && close + 2 < line.size() && line[close + 2] != 'Z' && line[close + 2] != 'X';
}

// --hold-child FD: the stand-in launcher of test_child_ends_with_launcher. Starts `sleep 300` from a thread that
// then ends, writes the child's pid to FD and waits to be killed.
int hold_child(int fd) {
    std::unique_ptr<ChildProcess> child;
    std::thread([&] { child = ChildProcess::start("/bin/sleep", {"300"}, {}, {}); }).join();
    const pid_t pid = static_cast<pid_t>(child->pid());
    if (write(fd, &pid, sizeof(pid)) != static_cast<ssize_t>(sizeof(pid))) { return 1; }
    for (;;) { pause(); }
}

// A server ends with the launcher, even one killed by SIGKILL (Windows: the job object), and not with the
// thread that started it (PR_SET_PDEATHSIG follows the forking thread: ChildProcess forks from one that lives as
// long as the launcher).
void test_child_ends_with_launcher() {
    {
        std::unique_ptr<ChildProcess> child;
        std::thread([&] { child = ChildProcess::start("/bin/sleep", {"30"}, {}, {}); }).join();
        std::this_thread::sleep_for(300ms);
        check(!child->exit_code() && running(static_cast<pid_t>(child->pid())),
              "a child outlives the thread that started it");
        child->kill();
        check(child->wait_for(5s), "a killed child ends");
    }
    int fds[2];
    if (pipe(fds) != 0) {
        check(false, "a pipe to the stand-in launcher");
        return;
    }
    const pid_t launcher = fork();
    if (launcher == 0) {
        close(fds[0]);
        const std::string fd = std::to_string(fds[1]);
        execl("/proc/self/exe", "test_launcher", "--hold-child", fd.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(fds[1]);
    pid_t held = -1;
    const bool started = read(fds[0], &held, sizeof(held)) == static_cast<ssize_t>(sizeof(held)) && held > 0;
    close(fds[0]);
    check(started && running(held), "the stand-in launcher's child runs");
    ::kill(launcher, SIGKILL);
    int raw = 0;
    waitpid(launcher, &raw, 0);
    bool ended = !started;
    for (int poll = 0; poll < 100 && !ended; ++poll) {
        ended = !running(held);
        if (!ended) { std::this_thread::sleep_for(50ms); }
    }
    check(started && ended, "a child ends with a launcher killed by SIGKILL");
}
#endif

// A client body nested past kMaxJsonNesting is refused before the launcher builds a DOM it would
// serialize or copy recursively; brackets inside strings do not count.
void test_json_nesting() {
    const auto nested = [](std::size_t depth) { return std::string(depth, '[') + std::string(depth, ']'); };
    check(json_text_nesting(R"({"a":[1,{"b":"[[[{{{"}],"c":"\"]"})") == 3, "nesting counts arrays and objects, not string text");
    check(json_text_nesting("") == 0 && json_text_nesting("\"[\"") == 0, "a scalar has no nesting");
    const Json deepest = parse_client_json(nested(kMaxJsonNesting));
    check(deepest.dump() == nested(kMaxJsonNesting), "a body at the bound parses and serializes");
    bool refused = false;
    try {
        (void)parse_client_json(nested(kMaxJsonNesting + 1));
    } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "a body past the bound is refused");
    refused = false;
    try {
        (void)parse_client_json(R"({"a": )");
    } catch (const Json::exception&) { refused = true; }
    check(refused, "malformed JSON still fails as a parse error");
}

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

int main(int argc, char** argv) {
#ifdef __linux__
    if (argc == 3 && std::string(argv[1]) == "--hold-child") { return hold_child(std::stoi(argv[2])); }
    test_child_ends_with_launcher();
#else
    (void)argc;
    (void)argv;
#endif
    test_gpu_order();
    test_models_list_body();
    test_json_nesting();
    test_args();
    test_http_access();
    test_batch_spawn();
    test_blob_ids();
    test_chats();
    test_store_durability();
    test_store_invalid_utf8();
    test_download_cancel_race();
    test_records();
    test_help();
    test_help_cache();
    test_serve_log_files();
    test_library();
    test_option_support();
    test_library_part_paths();
    test_scan_folder();
    test_library_adapters();
    test_library_sidecars();
    test_router();
    test_router_cancel();
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
