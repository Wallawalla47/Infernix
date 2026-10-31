// infernix-serve --plan's JSON (serve/plan_report.h): the key names and types the launcher reads, null for
// the quantities an architecture does not have, and the error object of a failed plan.

#include "serve/plan_report.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<std::string> keys(const Json& object) {
    std::vector<std::string> out;
    for (const auto& item : object.items()) { out.push_back(item.key()); }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace

int main() {
    using namespace infernix::serve;
    int failures = 0;

    // A resident Qwen3.5 plan that fits: no expert or n-gram figures.
    infernix::EnginePlan resident;
    resident.architecture                   = "Qwen3_5ForCausalLM";
    resident.gpu.name                       = "NVIDIA GeForce RTX 5090";
    resident.gpu.total_bytes                = 34190917632ULL;
    resident.gpu.free_bytes                 = 32000000000ULL;
    resident.vram.weights_bytes             = 18000000000ULL;
    resident.vram.kv_bytes                  = 9000000000ULL;
    resident.vram.kv_tokens                 = 262144;
    resident.vram.minimum_reservation_bytes = 3000000000ULL;
    resident.vram.bytes_per_page_group      = 2228224ULL;
    resident.vram.page_group_tokens         = 64U;
    resident.vram.workspace_bytes           = 1500000000ULL;
    resident.vram.graph_allowance_bytes     = 300000000ULL;
    resident.vram.headroom_bytes            = 1073741824ULL;
    resident.vram.planned_bytes             = 29000000000ULL;
    resident.vram.left_bytes                = 1900000000ULL;
    resident.ram.total_bytes                = 102000000000ULL;
    resident.ram.available_bytes            = 80000000000ULL;
    resident.ram.ram_free_bytes             = 4294967296ULL;
    resident.ram.host_context_bytes         = 8589934592ULL;
    resident.ram.planned_bytes              = 8589934592ULL;
    resident.ram.left_bytes                 = 67000000000ULL;
    resident.max_context                    = 262144U;
    const Json root = Json::parse(render_plan_json(resident));

    failures += check(keys(root) == std::vector<std::string>{"architecture", "error", "estimated", "fits", "gpu",
                                                              "max_context", "ram", "schema", "vram"},
                      "top-level keys");
    failures += check(root["schema"] == 1 && root["fits"] == true && root["estimated"] == true &&
                          root["error"].is_null() && root["architecture"] == "Qwen3_5ForCausalLM",
                      "a fitting plan: schema, fits, no error");
    failures += check(root["gpu"]["name"] == "NVIDIA GeForce RTX 5090" &&
                          root["gpu"]["total_bytes"] == 34190917632ULL && root["gpu"]["budget_bytes"].is_null(),
                      "gpu: name, sizes as integers, no OS budget");
    const Json& vram = root["vram"];
    failures += check(keys(vram) == std::vector<std::string>{"bytes_per_page_group", "expert_cache_bytes",
                                                              "expert_frames", "graph_allowance_bytes",
                                                              "headroom_bytes", "kv_bytes", "kv_tokens", "left_bytes",
                                                              "minimum_reservation_bytes", "page_group_tokens",
                                                              "planned_bytes", "weights_bytes", "workspace_bytes"},
                      "vram keys");
    failures += check(vram["weights_bytes"].is_number_unsigned() && vram["kv_tokens"] == 262144 &&
                          vram["bytes_per_page_group"] == 2228224 && vram["page_group_tokens"] == 64 &&
                          vram["expert_frames"].is_null() && vram["expert_cache_bytes"].is_null(),
                      "vram: the curve's slope, no expert cache");
    const Json& ram = root["ram"];
    failures += check(keys(ram) == std::vector<std::string>{"available_bytes", "expert_ram_layout", "experts_bytes",
                                                             "host_context_bytes", "left_bytes", "ngram_bytes",
                                                             "ngram_residency", "pinned_bytes", "planned_bytes",
                                                             "ram_free_bytes", "tier", "total_bytes"},
                      "ram keys");
    failures += check(ram["host_context_bytes"] == 8589934592ULL && ram["experts_bytes"].is_null() &&
                          ram["pinned_bytes"].is_null() && ram["tier"].is_null() &&
                          ram["ngram_residency"].is_null() && ram["expert_ram_layout"].is_null(),
                      "ram: the Host tier; no experts, pins or n-gram table");
    failures += check(root["max_context"] == 262144, "max_context");

    // A Qwen3.8-Flash-Next plan that does not fit: the refusal, the expert figures and the n-gram table.
    infernix::EnginePlan offloaded       = resident;
    offloaded.fits                       = false;
    offloaded.error                      = "Qwen3.8-Flash-Next does not fit in device memory: 1200 MiB free";
    offloaded.architecture               = "Qwen4ExpForCausalLM";
    offloaded.gpu.budget_bytes           = 31000000000ULL;
    offloaded.vram.bytes_per_page_group  = std::nullopt;
    offloaded.vram.page_group_tokens     = std::nullopt;
    offloaded.vram.expert_frames         = 0U;
    offloaded.vram.expert_cache_bytes    = 0ULL;
    offloaded.ram.experts_bytes          = 60000000000ULL;
    offloaded.ram.pinned_bytes           = 61000000000ULL;
    offloaded.ram.ngram_bytes            = 2000000000ULL;
    offloaded.ram.ngram_residency        = infernix::NgramResidency::Ram;
    offloaded.ram.tier                   = infernix::EnginePlanExpertTier::Ssd;
    offloaded.ram.expert_ram_layout      = infernix::ExpertRamLayout::Exclusive;
    offloaded.max_context                = std::nullopt;
    const Json refused = Json::parse(render_plan_json(offloaded));
    failures += check(refused["fits"] == false && refused["error"] == offloaded.error,
                      "a refused plan carries the start's message");
    failures += check(refused["gpu"]["budget_bytes"] == 31000000000ULL, "the OS budget when known");
    failures += check(refused["vram"]["bytes_per_page_group"].is_null() && refused["vram"]["expert_frames"] == 0,
                      "Flash-Next: no page-group slope; zero frames is a number, not null");
    failures += check(refused["ram"]["tier"] == "ssd" && refused["ram"]["ngram_residency"] == "ram" &&
                          refused["ram"]["expert_ram_layout"] == "exclusive" &&
                          refused["ram"]["experts_bytes"] == 60000000000ULL,
                      "the experts' RAM, tier, layout and n-gram table");
    failures += check(refused["max_context"].is_null(), "no max_context when none was computed");
    offloaded.ram.tier            = infernix::EnginePlanExpertTier::Full;
    offloaded.ram.ngram_residency = infernix::NgramResidency::Volume;
    const Json full = Json::parse(render_plan_json(offloaded));
    failures += check(full["ram"]["tier"] == "full" && full["ram"]["ngram_residency"] == "volume",
                      "tier full and a table read from its volume");

    // A plan that could not be made.
    const Json error = Json::parse(render_plan_error_json("Infernix accepts only .infernix and .ninfer artifacts"));
    failures += check(error["schema"] == 1 && error["error"] == "Infernix accepts only .infernix and .ninfer artifacts",
                      "the error object");
    failures += check(!Json::parse(render_plan_error_json(std::string("bad \xff byte")))["error"].get<std::string>().empty(),
                      "a message that is not UTF-8 still renders");

    return failures == 0 ? 0 : 1;
}
