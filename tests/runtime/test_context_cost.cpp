#include "runtime/engine/context_cache/context_cost.h"

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

using Json = nlohmann::json;

long long test_process_id() {
#ifdef _WIN32
    return _getpid();
#else
    return ::getpid();
#endif
}

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

template <class Function>
void expect_throw(Function&& function, const char* message) {
    try {
        function();
    } catch (const std::exception&) { return; }
    expect(false, message);
}

std::array<infernix::runtime::ContextTransferCost, 3> transfer(std::uint64_t batch_ns = 10) {
    std::array<infernix::runtime::ContextTransferCost, 3> result;
    for (auto& direction : result) {
        direction = {.batch_ns        = batch_ns,
                     .operation_ns    = 3,
                     .ns_per_byte_q32 = infernix::runtime::kContextCostQ32One / 2};
    }
    return result;
}

infernix::runtime::ContextPrefillCost prefill(std::uint64_t chunk_ns = 3) {
    return {
        .chunk_ns              = chunk_ns,
        .token_ns_q32          = infernix::runtime::kContextCostQ32One / 2,
        .attention_pair_ns_q32 = infernix::runtime::kContextCostQ32One / 4,
        .vision_item_ns        = 5,
        .vision_patch_ns_q32   = 2 * infernix::runtime::kContextCostQ32One,
    };
}

Json direction_json(const infernix::runtime::ContextTransferCost& value) {
    return Json{{"batch_ns", value.batch_ns},
                {"operation_ns", value.operation_ns},
                {"ns_per_byte_q32", value.ns_per_byte_q32}};
}

Json transfer_json(const std::array<infernix::runtime::ContextTransferCost, 3>& value) {
    return Json{{"d2h", direction_json(value[0])},
                {"h2d", direction_json(value[1])},
                {"d2d", direction_json(value[2])}};
}

Json prefill_json(const infernix::runtime::ContextPrefillCost& value) {
    return Json{{"chunk_ns", value.chunk_ns},
                {"token_ns_q32", value.token_ns_q32},
                {"attention_pair_ns_q32", value.attention_pair_ns_q32},
                {"vision_item_ns", value.vision_item_ns},
                {"vision_patch_ns_q32", value.vision_patch_ns_q32}};
}

Json document(Json machines) {
    return Json{{"schema_version", 3},
                {"artifact_type", "ninfer_context_cost_presets"},
                {"machines", std::move(machines)}};
}

Json machine(std::string hardware, Json transfer_value, Json prefill_values) {
    return Json{{"hardware_class", std::move(hardware)},
                {"transfer", std::move(transfer_value)},
                {"prefill", std::move(prefill_values)}};
}

Json prefill_entry(std::string signature, const infernix::runtime::ContextPrefillCost& value) {
    return Json{{"prefill_signature", std::move(signature)}, {"coefficients", prefill_json(value)}};
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void test_hardware_class() {
    expect(infernix::runtime::context_cost_hardware_class("NVIDIA GeForce RTX 5090", 12, 0) ==
               "nvidia-geforce-rtx-5090-sm120",
           "hardware class duplicated or lost the NVIDIA vendor");
    expect(infernix::runtime::context_cost_hardware_class("Example Accelerator", 9, 0) ==
               "nvidia-example-accelerator-sm90",
           "hardware class did not canonicalize a vendorless CUDA device name");
}

void test_schema_validation() {
    Json entries = Json::array();
    entries.push_back(prefill_entry("config-and-formats", prefill()));
    const Json valid =
        document(Json::array({machine("machine", transfer_json(transfer()), entries)}));
    const auto parsed = infernix::runtime::parse_context_cost_presets(valid.dump(), "test");
    expect(parsed.size() == 1 && parsed[0].transfer == transfer() &&
               parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost == prefill(),
           "valid layered preset did not parse");

    Json duplicate_machine = valid;
    duplicate_machine["machines"].push_back(duplicate_machine["machines"][0]);
    expect_throw(
        [&] {
            (void)infernix::runtime::parse_context_cost_presets(duplicate_machine.dump(),
                                                              "duplicate-machine");
        },
        "duplicate hardware entry was accepted");

    Json duplicate_prefill = valid;
    duplicate_prefill["machines"][0]["prefill"].push_back(
        duplicate_prefill["machines"][0]["prefill"][0]);
    expect_throw(
        [&] {
            (void)infernix::runtime::parse_context_cost_presets(duplicate_prefill.dump(),
                                                              "duplicate-prefill");
        },
        "duplicate model prefill entry was accepted");

    Json invalid                                              = valid;
    invalid["machines"][0]["transfer"]["d2h"]["operation_ns"] = -1;
    expect_throw(
        [&] { (void)infernix::runtime::parse_context_cost_presets(invalid.dump(), "negative"); },
        "negative transfer coefficient was accepted");

    Json prefill_only                       = valid;
    prefill_only["machines"][0]["transfer"] = nullptr;
    expect(
        infernix::runtime::parse_context_cost_presets(prefill_only.dump(), "prefill-only").size() ==
            1,
        "independent prefill-only machine entry was rejected");
    prefill_only["machines"][0]["prefill"] = Json::array();
    expect_throw(
        [&] { (void)infernix::runtime::parse_context_cost_presets(prefill_only.dump(), "empty"); },
        "empty machine cost entry was accepted");
}

void test_resolution_and_atomic_upserts() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("infernix-context-cost-test-" + std::to_string(test_process_id()));
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / "presets.json";
    try {
        const infernix::runtime::ContextCostIdentity compiled_identity{
            .hardware_class    = "nvidia-geforce-rtx-5090-sm120",
            .prefill_signature = "unmeasured-config-and-formats",
        };
        const auto compiled = infernix::runtime::resolve_context_machine_cost(compiled_identity);
        expect(
            compiled.summary.transfer_source == infernix::ContextCostPresetSource::CompiledDefault &&
                compiled.summary.prefill_source == infernix::ContextCostPresetSource::GenericDefault,
            "compiled transfer and prefill defaults did not resolve independently");

        const auto measured = infernix::runtime::resolve_context_machine_cost({
            .hardware_class    = compiled_identity.hardware_class,
            .prefill_signature = "e6eae48276e11c15c932cb90d258b51b81e144dc13fb461e7ffa202d2caa440a",
        });
        expect(measured.summary.prefill_source ==
                       infernix::ContextCostPresetSource::CompiledDefault &&
                   measured.model.prefill.token_ns_q32 == 375'800'765'711'778,
               "measured bindings did not select their compiled prefill cost");

        const infernix::runtime::ContextCostIdentity unknown{
            .hardware_class    = "unmeasured-machine",
            .prefill_signature = "unmeasured-config-and-formats",
        };
        const auto generic = infernix::runtime::resolve_context_machine_cost(unknown);
        expect(
            generic.summary.transfer_source == infernix::ContextCostPresetSource::GenericDefault &&
                generic.summary.prefill_source == infernix::ContextCostPresetSource::GenericDefault &&
                generic.model.transfer[0].ns_per_byte_q32 > 0 &&
                generic.model.prefill.token_ns_q32 > 0,
            "unknown identity did not retain a numerical cost model");

        infernix::runtime::upsert_context_transfer_cost_atomic(path, unknown.hardware_class,
                                                             transfer(77), R"({"run":1})");
        auto resolved = infernix::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 &&
                   resolved.summary.transfer_source == infernix::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source ==
                       infernix::ContextCostPresetSource::GenericDefault,
               "external transfer did not layer over generic prefill");

        infernix::runtime::upsert_context_prefill_cost_atomic(path, unknown, prefill(91),
                                                            R"({"run":2})");
        resolved = infernix::runtime::resolve_context_machine_cost(unknown, path);
        expect(resolved.model.transfer[0].batch_ns == 77 && resolved.model.prefill.chunk_ns == 91 &&
                   resolved.summary.transfer_source == infernix::ContextCostPresetSource::External &&
                   resolved.summary.prefill_source == infernix::ContextCostPresetSource::External,
               "independent external transfer/prefill entries did not compose");

        const infernix::runtime::ContextCostIdentity other_model{
            .hardware_class    = unknown.hardware_class,
            .prefill_signature = "other-config-and-formats",
        };
        const auto layered_miss = infernix::runtime::resolve_context_machine_cost(other_model, path);
        expect(layered_miss.summary.transfer_source == infernix::ContextCostPresetSource::External &&
                   layered_miss.summary.prefill_source ==
                       infernix::ContextCostPresetSource::GenericDefault,
               "external model miss incorrectly discarded the matching machine transfer");

        infernix::runtime::upsert_context_transfer_cost_atomic(path, unknown.hardware_class,
                                                             transfer(123), R"({"run":3})");
        const auto parsed = infernix::runtime::parse_context_cost_presets(read_file(path), "file");
        expect(parsed.size() == 1 && parsed[0].transfer->at(0).batch_ns == 123 &&
                   parsed[0].prefill.size() == 1 && parsed[0].prefill[0].cost.chunk_ns == 91,
               "transfer replacement did not preserve the model prefill list");

        const std::string before = read_file(path);
        expect_throw(
            [&] {
                infernix::runtime::upsert_context_prefill_cost_atomic(path, unknown, prefill(),
                                                                    "not-json");
            },
            "invalid provenance was accepted");
        expect(read_file(path) == before, "failed upsert modified the preset file");
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    test_hardware_class();
    test_schema_validation();
    test_resolution_and_atomic_upserts();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
