// The CPU expert routes on CSF records (ops/offloaded_sparse_moe/csf_record.h): every route (W4A4, W4A16 exact,
// two-term A4, the 8-bit activations), every supported ISA and every width class (one pass, two passes, two column
// groups) computes from a compressed record exactly the bits it computes from the expanded record, at both
// registered geometries, for scales as NVIDIA's exports have them, with outlying codes, with every group raw and with
// raw and compressed groups mixed; and the worker team does the same for jobs of both layouts in one round.
// Host-only: needs no GPU.

#include "ops/op_family.h"
#include "ops/offloaded_sparse_moe/cpu/a4r_expert.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/w4a16_expert.h"
#include "ops/offloaded_sparse_moe/cpu/w4a8_expert.h"
#include "ops/offloaded_sparse_moe/csf_record.h"
#include "ops/offloaded_moe_fixtures.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace moe      = infernix::ops::offloaded_moe;
namespace csf      = infernix::ops::offloaded_moe::csf;
namespace fixtures = infernix::test::offloaded_moe;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

enum class Scales { Realistic, Outliers, Mixed, Raw };

const char* name(Scales s) {
    switch (s) {
    case Scales::Realistic: return "realistic";
    case Scales::Outliers: return "outliers";
    case Scales::Mixed: return "mixed";
    case Scales::Raw: return "raw";
    }
    return "?";
}

// A record's scale bytes: each row's codes near its own centre (NVIDIA's exports), some 30 codes above it, every
// third group's codes spread (raw groups), or the fixture's spread codes everywhere (every group raw).
void set_scales(const moe::ExpertShape& shape, Scales kind, std::mt19937& rng, std::vector<std::uint8_t>& record) {
    if (kind == Scales::Raw) { return; }
    const csf::Geometry g = csf::Geometry::of(shape);
    for (int rg = 0; rg < g.rgs(); ++rg) {
        int centre[16];
        for (auto& c : centre) { c = 30 + static_cast<int>(rng() % 60); }
        for (int b = 0; b < g.blocks(rg); ++b) {
            std::uint8_t* s = record.data() + g.expanded_offset(rg) + static_cast<std::size_t>(b) * moe::kUnitBytes + 128;
            for (int r = 0; r < 16; ++r) {
                const int near = centre[r] + static_cast<int>(rng() % 9) - 4;
                int code       = near;
                if (kind == Scales::Outliers && rng() % 40 == 0) { code = near + 30; }
                if (kind == Scales::Mixed && rg % 3 == 1) { code = 40 + static_cast<int>(rng() % 41); }
                s[r] = static_cast<std::uint8_t>(code);
            }
        }
    }
}

struct Case {
    fixtures::Expert expert;            // expanded
    std::vector<std::uint8_t> compressed; // the same record compressed in place
    std::vector<std::uint32_t> table;
    [[nodiscard]] moe::Rg16Record csf_record() const { return {compressed.data(), table.data()}; }
};

Case make_case(const moe::ExpertShape& shape, Scales kind, bool a16, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Case c;
    c.expert = a16 ? fixtures::random_a16_expert(rng, shape) : fixtures::random_expert(rng, false, shape);
    set_scales(shape, kind, rng, c.expert.record);
    c.compressed = c.expert.record;
    c.table.resize(static_cast<std::size_t>(csf::Geometry::of(shape).table_words()));
    (void)csf::compress_record(shape, c.compressed.data(), c.compressed.data(), c.table.data());
    check(csf::valid_table(shape, c.table.data(), c.compressed.size()), std::string(name(kind)) + ": table");
    return c;
}

std::vector<moe::CpuIsa> isas() {
    std::vector<moe::CpuIsa> out;
    for (const moe::CpuIsa isa : {moe::CpuIsa::kScalar, moe::CpuIsa::kAvx2, moe::CpuIsa::kAvxVnni, moe::CpuIsa::kAvx512Vnni}) {
        if (moe::cpu_isa_supported(isa)) { out.push_back(isa); }
    }
    return out;
}

// One route's outputs at `ncols` columns from a record of either layout.
using Forward = void (*)(moe::CpuIsa, const moe::ExpertShape&, const moe::Rg16Record&, const moe::ExpertScales&, int,
                         const std::uint16_t* const*, std::uint16_t* const*);

std::vector<std::uint16_t> run(Forward forward, moe::CpuIsa isa, const moe::ExpertShape& shape,
                               const moe::Rg16Record& record, const moe::ExpertScales& scales,
                               const std::vector<std::uint16_t>& x, int ncols) {
    const auto hidden = static_cast<std::size_t>(shape.hidden);
    std::vector<std::uint16_t> y(static_cast<std::size_t>(ncols) * hidden, 0xFFFF);
    const std::uint16_t* xp[moe::kMaxColumns];
    std::uint16_t* yp[moe::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = x.data() + static_cast<std::size_t>(c) * hidden;
        yp[c] = y.data() + static_cast<std::size_t>(c) * hidden;
    }
    forward(isa, shape, record, scales, ncols, xp, yp);
    return y;
}

void a16_forward(moe::CpuIsa isa, const moe::ExpertShape& shape, const moe::Rg16Record& record,
                 const moe::ExpertScales& scales, int ncols, const std::uint16_t* const* x, std::uint16_t* const* y) {
    moe::expert_forward_a16(isa, shape, record, scales, ncols, x, y);
}
void a8_forward(moe::CpuIsa isa, const moe::ExpertShape& shape, const moe::Rg16Record& record,
                const moe::ExpertScales& scales, int ncols, const std::uint16_t* const* x, std::uint16_t* const* y) {
    moe::expert_forward_a8(isa, shape, record, scales, ncols, x, y);
}
void a4_forward(moe::CpuIsa isa, const moe::ExpertShape&, const moe::Rg16Record& record,
                const moe::ExpertScales& scales, int ncols, const std::uint16_t* const* x, std::uint16_t* const* y) {
    moe::expert_forward(isa, record, scales, ncols, x, y);
}
void a4r_forward(moe::CpuIsa isa, const moe::ExpertShape&, const moe::Rg16Record& record,
                 const moe::ExpertScales& scales, int ncols, const std::uint16_t* const* x, std::uint16_t* const* y) {
    moe::expert_forward_a4r(isa, record, scales, ncols, x, y);
}

struct Route {
    const char* name;
    Forward forward;
    bool a16;        // the expert has no activation scales
    bool flash_next; // Flash-Next's geometry only
};

constexpr Route kRoutes[] = {
    {"w4a16", a16_forward, true, false},
    {"w4a8", a8_forward, true, false},
    {"w4a4", a4_forward, false, true},
    {"a4r", a4r_forward, true, true},
};

void test_routes(const moe::ExpertShape& shape, const char* geometry) {
    std::mt19937 rng(17);
    for (const Route& route : kRoutes) {
        if (route.flash_next && shape != moe::kFlashNextShape) { continue; }
        for (const Scales kind : {Scales::Realistic, Scales::Outliers, Scales::Mixed, Scales::Raw}) {
            const Case c = make_case(shape, kind, route.a16, 101U + static_cast<std::uint32_t>(kind));
            for (const moe::CpuIsa isa : isas()) {
                // Scalar is slow: one and three columns; the vector ISAs every width class.
                const std::vector<int> widths = isa == moe::CpuIsa::kScalar ? std::vector<int>{1, 3}
                                                                            : std::vector<int>{1, 2, 3, 5, 6, 8};
                for (const int ncols : widths) {
                    const auto x = route.a16 ? fixtures::wide_range_activations(rng, ncols, shape.hidden)
                                             : fixtures::random_activations(rng, ncols, shape.hidden);
                    const auto expanded = run(route.forward, isa, shape, c.expert.record.data(), c.expert.scales, x, ncols);
                    const auto compressed = run(route.forward, isa, shape, c.csf_record(), c.expert.scales, x, ncols);
                    check(expanded == compressed, std::string(geometry) + " " + route.name + " " + name(kind) + " " +
                                                      moe::cpu_isa_name(isa) + " ncols " + std::to_string(ncols) +
                                                      ": CSF record's outputs differ");
                }
            }
        }
    }
}

// A team round of jobs of both layouts gives every job its single-thread bits.
void test_team(const moe::ExpertShape& shape, const char* geometry, moe::ExpertActivation activation,
               moe::NarrowA16 narrow, bool a8) {
    const bool a16 = activation == moe::ExpertActivation::kA16;
    std::mt19937 rng(23);
    std::vector<Case> cases;
    for (const Scales kind : {Scales::Realistic, Scales::Mixed, Scales::Outliers}) {
        cases.push_back(make_case(shape, kind, a16, 301U + static_cast<std::uint32_t>(kind)));
    }
    moe::CpuExpertTeam team({.workers = 3, .max_jobs = 8, .activation = activation, .narrow_a16 = narrow,
                             .nvfp4_a8 = a8, .shape = shape});
    const bool a4r          = narrow == moe::NarrowA16::kA4Residual;
    const Forward forward   = a8 ? a8_forward : !a16 ? a4_forward : a4r ? a4r_forward : a16_forward;
    const std::string label = std::string(geometry) + (a8 ? " a8" : !a16 ? " w4a4" : a4r ? " a4r" : " w4a16");
    const auto hidden       = static_cast<std::size_t>(shape.hidden);
    std::vector<std::vector<std::uint16_t>> xs, ys, refs;
    std::vector<moe::CpuExpertJob> jobs;
    for (int j = 0; j < 6; ++j) {
        const Case& c   = cases[static_cast<std::size_t>(j) % cases.size()];
        const int ncols = 1 + j;
        xs.push_back(a16 ? fixtures::wide_range_activations(rng, ncols, shape.hidden)
                         : fixtures::random_activations(rng, ncols, shape.hidden));
        refs.push_back(
            run(forward, moe::best_cpu_isa(), shape, c.expert.record.data(), c.expert.scales, xs.back(), ncols));
        ys.emplace_back(static_cast<std::size_t>(ncols) * hidden, 0xFFFF);
    }
    for (int j = 0; j < 6; ++j) {
        const Case& c   = cases[static_cast<std::size_t>(j) % cases.size()];
        const bool comp = j % 2 == 0;
        moe::CpuExpertJob job;
        job.record = comp ? c.compressed.data() : c.expert.record.data();
        job.csf    = comp ? c.table.data() : nullptr;
        job.scales = c.expert.scales;
        job.a8     = a8;
        job.ncols  = 1 + j;
        for (int col = 0; col < job.ncols; ++col) {
            job.x[col] = xs[static_cast<std::size_t>(j)].data() + static_cast<std::size_t>(col) * hidden;
            job.y[col] = ys[static_cast<std::size_t>(j)].data() + static_cast<std::size_t>(col) * hidden;
        }
        jobs.push_back(job);
    }
    team.run(jobs);
    for (std::size_t j = 0; j < jobs.size(); ++j) {
        check(ys[j] == refs[j], label + " team job " + std::to_string(j) +
                                    ": outputs differ from the single-thread expanded record's");
    }
}

} // namespace

INFERNIX_OP_TEST(offloaded_moe_csf_cpu) {
    for (const auto& [shape, geometry] :
         {std::pair{moe::kFlashNextShape, "flash-next"}, std::pair{moe::kQwen3_5_122BShape, "122b"}}) {
        test_routes(shape, geometry);
        test_team(shape, geometry, moe::ExpertActivation::kA16, moe::NarrowA16::kExact, false);
        test_team(shape, geometry, moe::ExpertActivation::kA16, moe::NarrowA16::kExact, true);
    }
    test_team(moe::kFlashNextShape, "flash-next", moe::ExpertActivation::kA4, moe::NarrowA16::kExact, false);
    test_team(moe::kFlashNextShape, "flash-next", moe::ExpertActivation::kA16, moe::NarrowA16::kA4Residual, false);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all offloaded_moe CSF CPU checks passed\n");
    return 0;
}
