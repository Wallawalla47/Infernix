#include "runtime/expert_offload/placement.h"

#include "core/cpu_topology.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace infernix::runtime::expert_offload {
namespace {

// Strata's fit (src/program/generate.cpp PcieAuto, 0.1.42): the best measured shares of three
// machines lie within one 0.05 step of link / (link + 1.85 x CPU pool rate), kept in [0.10, 0.60].
constexpr double kPoolFactor = 1.85;
constexpr double kShareStep = 0.05, kShareMin = 0.10, kShareMax = 0.60;

std::string cpu_list(std::uint64_t cpus) {
    std::string out;
    for (int c = 0; c < 64; ++c) {
        if (((cpus >> c) & 1U) == 0) { continue; }
        if (!out.empty()) { out += ','; }
        out += std::to_string(c);
    }
    return out.empty() ? "none" : out;
}

} // namespace

std::int32_t link_share_256(const OffloadSwitches& switches, double link_bytes_per_second, double cpu_expert_rate,
                            double record_bytes) noexcept {
    double share = 0.0;
    switch (switches.link_share) {
    case OffloadSwitches::LinkShare::kDivisor: return 0;
    case OffloadSwitches::LinkShare::kFixed: share = switches.link_share_fixed; break;
    case OffloadSwitches::LinkShare::kAuto: {
        const double cpu = cpu_expert_rate * record_bytes; // bytes/s the CPU team serves
        if (!(link_bytes_per_second > 0.0) || !(cpu > 0.0)) { return 0; }
        share = link_bytes_per_second / (link_bytes_per_second + kPoolFactor * cpu);
        share = std::clamp(std::round(share / kShareStep) * kShareStep, kShareMin, kShareMax);
        break;
    }
    }
    return static_cast<std::int32_t>(std::clamp<long>(std::lround(share * 256.0), 1, 256));
}

std::int32_t plain_divisor(const OffloadSwitches& switches, std::int32_t position) noexcept {
    const bool long_context = position >= kPlainDivisorContext;
    if (switches.plain_divisor_all) { return long_context ? switches.plain_long : switches.plain_short; }
    return long_context ? 2 : 1;
}

CpuPlacement cpu_placement(const OffloadSwitches& switches, std::uint32_t configured) {
    CpuPlacement out;
    out.workers = switches.workers > 0 ? switches.workers : configured;
    if (switches.placement == OffloadSwitches::Placement::kDefault && !switches.engine_isolate) {
        if (switches.workers > 0) { out.text = "CPU expert team: " + std::to_string(out.workers) + " workers (INFERNIX_TMP_CPU_PLACEMENT), unpinned"; }
        return out;
    }
    std::vector<PhysicalCore> cores = physical_cores();
    if (cores.empty()) {
        out.text = "CPU placement switches ignored: the CPU topology is unknown (or the machine has more than 64 logical CPUs)";
        return out;
    }
    // The fastest class first (hybrid CPUs: the P-cores), the OS's order within a class.
    std::stable_sort(cores.begin(), cores.end(),
                     [](const PhysicalCore& a, const PhysicalCore& b) { return a.efficiency > b.efficiency; });
    if (switches.engine_isolate && cores.size() > 1) {
        // The last core of the fastest class: away from CPU 0, where Windows delivers the GPU's interrupts.
        std::size_t last = 0;
        for (std::size_t i = 0; i < cores.size(); ++i) {
            if (cores[i].efficiency == cores.front().efficiency) { last = i; }
        }
        out.engine = cores[last].cpus;
        cores.erase(cores.begin() + static_cast<std::ptrdiff_t>(last));
    }
    const auto first_thread = [](std::uint64_t cpus) { return cpus & (~cpus + 1U); };
    std::string mode;
    switch (switches.placement) {
    case OffloadSwitches::Placement::kPhysical: {
        mode = "one worker pinned per physical core, fastest cores first";
        for (std::uint32_t w = 0; w < out.workers; ++w) {
            out.cpus.push_back(w < cores.size() ? first_thread(cores[w].cpus) : 0U);
        }
        break;
    }
    case OffloadSwitches::Placement::kPhysicalSet: {
        mode = "every worker on the first logical CPU of any physical core";
        std::uint64_t set = 0;
        for (const PhysicalCore& core : cores) { set |= first_thread(core.cpus); }
        out.cpus.assign(out.workers, set);
        break;
    }
    case OffloadSwitches::Placement::kDefault: {
        mode = "workers unpinned except for the engine's core";
        std::uint64_t set = 0;
        for (const PhysicalCore& core : cores) { set |= core.cpus; }
        out.cpus.assign(out.workers, set);
        break;
    }
    }
    out.text = "CPU expert team placement (INFERNIX_TMP_CPU_PLACEMENT / INFERNIX_TMP_ENGINE_ISOLATE): " +
               std::to_string(out.workers) + " workers, " + mode + ";";
    if (switches.placement == OffloadSwitches::Placement::kPhysical) {
        for (std::uint32_t w = 0; w < out.workers; ++w) { out.text += " w" + std::to_string(w) + ":" + cpu_list(out.cpus[w]); }
        if (out.workers > cores.size()) { out.text += " (workers past the physical cores run unpinned)"; }
    } else {
        out.text += " CPUs " + cpu_list(out.cpus.empty() ? 0 : out.cpus.front());
    }
    out.text += "; engine thread " + (out.engine != 0 ? "on CPUs " + cpu_list(out.engine) : std::string("unpinned"));
    return out;
}

void place_engine_thread() noexcept {
    try {
        const OffloadSwitches& switches = temporary_switches::offload();
        if (!switches.engine_isolate) { return; }
        const std::uint64_t cpus = cpu_placement(switches, 0).engine;
        if (cpus != 0 && set_current_thread_cpus(cpus) == 0) {
            std::fprintf(stderr, "[engine] INFERNIX_TMP_ENGINE_ISOLATE: the engine thread could not be pinned\n");
        }
    } catch (...) {}
}

} // namespace infernix::runtime::expert_offload
