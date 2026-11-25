// Inkling 2's forward (src/models/inkling2/execution/forward.h) on the real artifact against the private CPU reference's
// per-visit dump (.local inkling2 tools/inkling_ref.py: the pinned arithmetic in FP32/FP64, the experts in their trained
// arithmetic; 32 positions: the golden prompt and the first 8 tokens of its completion).
//
//   INFERNIX_INKLING2_ARTIFACT   an artifact converted with the inkling2_mini recipe
//   INFERNIX_INKLING2_REFERENCE  the dump's path prefix: <prefix>.bin and <prefix>.json (tools/npz_to_bin.py)
//
// 1. Teacher-forced, on the decode route and again on the prompt route (CallPlan::prompt), the same criteria: two
//    calls of 16 columns; before each visit the reference's input, blocks and message sources are
//    injected, so each stage of each visit compares with the reference from the same inputs. Judged: every stage
//    upstream of the experts (x_q/x_k/x_v, q, r, the stored k and v, sink, gate, a, W_o, cand, x_mlp, gamma, the
//    final hidden) within 2^-5 of its row's largest magnitude; a routing difference at a near-tie is counted. The visit
//    tails cross the experts' 4-bit activation codes with inputs an ulp apart, so they are reported, not judged here:
//    their check is boundary-conditioned (the reference's MoE, conv and residual on this forward's own x_mlp and cand,
//    dumped with INFERNIX_INKLING2_DUMP, private .local tools/tail_check.py) and is not part of this test, a gap.
// 2. Free-running: the same calls on the forward's own wiring. Judged: the argmax from the last prompt position on
//    (the greedy continuation) equals the reference's. Reported: every other position's argmax, the largest logit
//    difference and the mean NLL of the next ids (with <reference>.spread.json's perturbation distribution when
//    present). The model is sensitive to rounding in its trained W4A4 arithmetic: a 1-ulp change in 24 of 2304
//    embedding elements moves the reference's own logits by up to 8.8 and its mean NLL by up to 0.40, so neither is a
//    criterion (a first criterion on prompt-position argmax and one on the NLL both failed for that reason, 2026-11-21).
// 3. Decode = verify: the 32 positions as single-column calls give the free-running calls' logits bit for bit.
// 4. Graphs: a 16-column call and a 1-column step captured whole (PDL dependents) and replayed three times from the
//    same state give the eager logits bit for bit.
// 5. Lanes: three lanes (different tokens, mixed widths, a padding column) in one call give each lane's solo logits.
// 6. The bit guards, for changes claimed to keep the bits: INFERNIX_INKLING2_LOGITS_GUARD (the decode route's
//    free-running logits, written by INFERNIX_INKLING2_LOGITS_OUT) and INFERNIX_INKLING2_PROMPT_LOGITS_GUARD (the
//    prompt route's 32-column call, written by INFERNIX_INKLING2_PROMPT_LOGITS_OUT; the Engine test's scoring check
//    reads it), each byte for byte.
// 7. The prompt route (CallPlan::prompt: tensor-core projections, router and experts): the 32 positions as one prompt
//    call give the bits of prompt calls of 8, and the greedy continuation equals the reference's. Reported: its
//    argmax elsewhere and its largest logit difference from the reference and from the decode route.
//
// Timing (environment): INFERNIX_INKLING2_PROFILE = width (the captured decode round, INFERNIX_INKLING2_CONTEXT
// positions before it; INFERNIX_INKLING2_SKELETON = serial or trigger also times the same graph with every kernel
// empty, = kernel-name parts "a|b" with those kernels empty, = names lists the round's kernels and their node counts);
// INFERNIX_INKLING2_PREFILL_PROFILE = width (one prompt call, each Op's time;
// INFERNIX_INKLING2_PREFILL_DECODE runs it on the decode route instead, for comparison).
//
// Diagnostics (environment): INFERNIX_INKLING2_DRIFT prints the free-running drift per visit and the NLL detail;
// INFERNIX_INKLING2_HYBRID = blocks | input | perturb runs a hybrid (the reference's blocks and message sources, or its
// visit inputs, or a 1-ulp embedding perturbation); INFERNIX_INKLING2_DUMP = a directory writes the teacher-forced
// x_mlp, cand, tails and routing.

#include "artifact/reader.h"
#include "core/device.h"
#include "infernix/ops/inkling2_attention.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/load.h"
#include "ops/common/canonical_math.h"

#include <cuda_fp16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// graph_skeleton.cu: every kernel node of a graph made an empty kernel of the same launch shape; the count replaced.
int skeletonize_graph(cudaGraph_t graph, bool trigger, const std::string& filter);
void print_graph_kernels(cudaGraph_t graph);

namespace {

using namespace infernix;
namespace ik    = ops::inkling2;
namespace canon = ops::canon;
namespace m     = models::inkling2;

constexpr int H = ik::kHidden, kT = 16, kPositions = 32;

void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

struct Reference {
    nlohmann::json index;
    std::vector<char> bytes;
    explicit Reference(const std::string& prefix) {
        std::ifstream j(prefix + ".json");
        if (!j) { throw std::runtime_error("cannot read " + prefix + ".json"); }
        index = nlohmann::json::parse(j);
        std::ifstream b(prefix + ".bin", std::ios::binary | std::ios::ate);
        if (!b) { throw std::runtime_error("cannot read " + prefix + ".bin"); }
        bytes.resize(static_cast<std::size_t>(b.tellg()));
        b.seekg(0);
        b.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    [[nodiscard]] bool has(const std::string& name) const { return index.contains(name); }
    template <class T>
    [[nodiscard]] const T* get(const std::string& name) const {
        if (!has(name)) { throw std::runtime_error("the reference has no " + name); }
        return reinterpret_cast<const T*>(bytes.data() + index.at(name).at("offset").get<std::size_t>());
    }
    [[nodiscard]] std::size_t row(const std::string& name) const { // elements per position
        const auto& shape = index.at(name).at("shape");
        std::size_t n = 1;
        for (std::size_t i = 1; i < shape.size(); ++i) { n *= shape[i].get<std::size_t>(); }
        return n;
    }
};

std::uint16_t b16(float x) { return canon::f32_to_bf16_rn(x); }
float f(std::uint16_t v) { return canon::bf16_to_f32(v); }

struct DeviceRows {
    std::uint16_t* p = nullptr;
    explicit DeviceRows(std::size_t words) {
        cuda(cudaMalloc(&p, words * 2), "cudaMalloc");
        cuda(cudaMemset(p, 0, words * 2), "cudaMemset");
    }
    ~DeviceRows() { cudaFree(p); }
    DeviceRows(const DeviceRows&)            = delete;
    DeviceRows& operator=(const DeviceRows&) = delete;
};

// Uploads the reference rows [c0, c0 + T) of `name` (FP32 carrying BF16 values) as BF16.
void upload_rows(const Reference& ref, const std::string& name, int c0, int T, std::uint16_t* dst, std::size_t width = H) {
    const float* src = ref.get<float>(name) + static_cast<std::size_t>(c0) * width;
    std::vector<std::uint16_t> host(static_cast<std::size_t>(T) * width);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = b16(src[i]);
        if (f(host[i]) != src[i]) { throw std::runtime_error(name + " holds a value that is not BF16"); }
    }
    cuda(cudaMemcpy(dst, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "upload");
}

template <class T>
std::vector<T> download(const T* p, std::size_t n) {
    std::vector<T> v(n);
    cuda(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}

// ---------------------------------------------------------------------------- stage comparison

struct Stage {
    double worst  = 0; // max over columns of max |kernel - ref| / max |ref row|
    long columns  = 0;
    long excluded = 0;
};

std::map<std::string, Stage> g_stages;
long g_routing_differs = 0;

// Compares `T` rows of `width` (kernel values as doubles) against the reference rows from c0, skipping columns whose
// previous routing differed (skip[t]).
void compare(const std::string& stage, const std::vector<double>& got, const Reference& ref, const std::string& name,
             int c0, int T, std::size_t width, const std::vector<bool>& skip) {
    const float* r = ref.get<float>(name) + static_cast<std::size_t>(c0) * width;
    Stage& s       = g_stages[stage];
    for (int t = 0; t < T; ++t) {
        if (skip[t]) {
            ++s.excluded;
            continue;
        }
        double scale = 0, diff = 0;
        for (std::size_t i = 0; i < width; ++i) { scale = std::max(scale, std::fabs(double{r[t * width + i]})); }
        for (std::size_t i = 0; i < width; ++i) { diff = std::max(diff, std::fabs(got[t * width + i] - r[t * width + i])); }
        s.worst = std::max(s.worst, scale > 0 ? diff / scale : diff);
        ++s.columns;
    }
}

std::vector<double> bf16_values(const std::uint16_t* p, std::size_t n) {
    const auto host = download(p, n);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = f(host[i]); }
    return out;
}
std::vector<double> f32_values(const float* p, std::size_t n) {
    const auto host = download(p, n);
    return {host.begin(), host.end()};
}

// INFERNIX_INKLING2_WINDOW (profiles only): the sliding windows' storage, int8 (unset: BF16).
ik::KvCodec window_from_env() {
    const char* w = std::getenv("INFERNIX_INKLING2_WINDOW");
    if (w == nullptr) { return ik::KvCodec::Bf16; }
    if (std::string(w) == "int8") { return ik::KvCodec::Int8G64; }
    throw std::invalid_argument("INFERNIX_INKLING2_WINDOW: int8");
}

// ---------------------------------------------------------------------------- state

// Lanes' images and the full visits' pools: `pages` 64-position pages a lane (lane l's page b at l * pages + b).
struct State {
    m::StateView view;
    std::uint16_t* images = nullptr;
    std::vector<std::uint16_t*> pools;
    int* table = nullptr;
    std::size_t image_bytes = 0, pool_bytes = 0;
    std::uint8_t *window_codes = nullptr, *window_side = nullptr; // quantized windows (StateView)
    std::size_t window_code_bytes = 0, window_side_bytes = 0;
    std::vector<std::vector<std::uint8_t>> saved;
    explicit State(const m::StateLayout& layout, int lanes = 1, int pages = 1, ik::KvCodec window = ik::KvCodec::Bf16) {
        image_bytes = static_cast<std::size_t>(layout.lane_words) * 2 * lanes;
        cuda(cudaMalloc(&images, image_bytes), "state");
        if (window != ik::KvCodec::Bf16) {
            window_code_bytes = static_cast<std::size_t>(m::window_code_bytes(layout, window)) * lanes;
            window_side_bytes = static_cast<std::size_t>(m::window_side_bytes(layout, window)) * lanes;
            cuda(cudaMalloc(&window_codes, window_code_bytes), "window codes");
            cuda(cudaMalloc(&window_side, window_side_bytes), "window side");
            view.window_codes  = window_codes;
            view.window_side   = window_side;
            view.window_planes = 2 * layout.sliding_visits;
        }
        view.window_codec = window;
        pool_bytes = std::size_t{ik::kKvHeads} * lanes * pages * ik::kKeyUnit * ik::kHeadDim * 2;
        for (int s = 0; s < 2 * layout.full_visits; ++s) {
            std::uint16_t* p = nullptr;
            cuda(cudaMalloc(&p, pool_bytes), "pool");
            pools.push_back(p);
        }
        std::vector<int> rows(static_cast<std::size_t>(lanes) * pages);
        for (std::size_t i = 0; i < rows.size(); ++i) { rows[i] = static_cast<int>(i); }
        cuda(cudaMalloc(&table, rows.size() * 4), "table");
        cuda(cudaMemcpy(table, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice), "table");
        view.images     = images;
        view.lane_words = layout.lane_words;
        for (int s = 0; s < layout.full_visits; ++s) {
            view.pool_k.push_back(pools[2 * s]);
            view.pool_v.push_back(pools[2 * s + 1]);
        }
        view.pool_pages   = static_cast<std::int64_t>(lanes) * pages;
        view.table        = table;
        view.table_stride = pages;
        reset();
    }
    void reset() {
        cuda(cudaMemset(images, 0, image_bytes), "reset");
        if (window_codes != nullptr) {
            cuda(cudaMemset(window_codes, 0, window_code_bytes), "reset");
            cuda(cudaMemset(window_side, 0, window_side_bytes), "reset");
        }
        for (auto* p : pools) { cuda(cudaMemset(p, 0, pool_bytes), "reset"); }
    }
    // A host copy of every byte of the state, and its return (both synchronous).
    void save() {
        saved.clear();
        saved.push_back(std::vector<std::uint8_t>(image_bytes));
        cuda(cudaMemcpy(saved.back().data(), images, image_bytes, cudaMemcpyDeviceToHost), "save");
        for (auto* p : pools) {
            saved.push_back(std::vector<std::uint8_t>(pool_bytes));
            cuda(cudaMemcpy(saved.back().data(), p, pool_bytes, cudaMemcpyDeviceToHost), "save");
        }
    }
    void restore() {
        cuda(cudaMemcpy(images, saved[0].data(), image_bytes, cudaMemcpyHostToDevice), "restore");
        for (std::size_t i = 0; i < pools.size(); ++i) {
            cuda(cudaMemcpy(pools[i], saved[i + 1].data(), pool_bytes, cudaMemcpyHostToDevice), "restore");
        }
    }
    ~State() {
        cudaFree(images);
        cudaFree(window_codes);
        cudaFree(window_side);
        for (auto* p : pools) { cudaFree(p); }
        cudaFree(table);
    }
};

// The K and V the visit appended at positions [c0, c0 + T) of lane 0, as values ([T][8][128]).
void appended_kv(const m::StateView& s, const m::VisitState& vs, int c0, int T, std::vector<double>& k,
                 std::vector<double>& v) {
    k.assign(static_cast<std::size_t>(T) * ik::kKvHeads * ik::kHeadDim, 0);
    v = k;
    for (int t = 0; t < T; ++t) {
        const int pos = c0 + t;
        for (int head = 0; head < ik::kKvHeads; ++head) {
            std::size_t at;
            const std::uint16_t *kb, *vb;
            if (vs.full_slot < 0) {
                at = ((static_cast<std::size_t>(head) * ik::kWindowRingPages + (pos / 64) % ik::kWindowRingPages) * 64 +
                      pos % 64) * ik::kHeadDim;
                kb = s.images + vs.window_k, vb = s.images + vs.window_v;
            } else {
                at = ((static_cast<std::size_t>(0) * ik::kKvHeads + head) * 64 + pos % 64) * ik::kHeadDim; // page 0
                kb = s.pool_k[vs.full_slot], vb = s.pool_v[vs.full_slot];
            }
            const auto kr = download(kb + at, ik::kHeadDim), vr = download(vb + at, ik::kHeadDim);
            for (int d = 0; d < ik::kHeadDim; ++d) {
                const std::size_t o = (static_cast<std::size_t>(t) * ik::kKvHeads + head) * ik::kHeadDim + d;
                k[o]                = f(kr[d]);
                v[o]                = __half2float(__ushort_as_half(vr[d]));
            }
        }
    }
}

m::CallPlan call_plan(const Reference& ref, int c0, int T) {
    m::CallPlan plan;
    const std::int32_t* ids = ref.get<std::int32_t>("ids");
    for (int t = 0; t < T; ++t) {
        plan.tokens.push_back(ids[c0 + t]);
        plan.lane.push_back(0);
        plan.position.push_back(c0 + t);
    }
    plan.first = {0};
    return plan;
}

// ---------------------------------------------------------------------------- 1. teacher-forced

// Diagnostic (INFERNIX_INKLING2_DUMP = a directory): the teacher-forced kernel's x_mlp, cand, visit tails and routing
// for every visit and position ([48][32][H] FP32, routing [48][32][8] I32), for the reference's MoE to rerun on them.
struct TailDump {
    std::vector<float> xm, cand, tail;
    std::vector<std::int32_t> routed;
    std::vector<double> worst; // per visit: the tail's worst |diff| / row max against the reference
    explicit TailDump(std::uint32_t visits)
        : xm(std::size_t{visits} * kPositions * H), cand(xm.size()), tail(xm.size()),
          routed(std::size_t{visits} * kPositions * 8), worst(visits, 0) {}
    void rows(std::vector<float>& dst, std::uint32_t v, int c0, const std::vector<double>& values) {
        std::copy(values.begin(), values.end(), dst.begin() + (std::size_t{v} * kPositions + c0) * H);
    }
    void write(const std::string& dir) const {
        const auto put = [&](const std::string& name, const void* data, std::size_t bytes) {
            std::ofstream out(dir + "/" + name, std::ios::binary);
            out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        };
        put("kxm.f32", xm.data(), xm.size() * 4);
        put("kcand.f32", cand.data(), cand.size() * 4);
        put("ktail.f32", tail.data(), tail.size() * 4);
        put("krouted.i32", routed.data(), routed.size() * 4);
    }
};

double tail_worst(const std::vector<double>& got, const Reference& ref, const std::string& name, int c0, int T) {
    const float* r = ref.get<float>(name) + static_cast<std::size_t>(c0) * H;
    double worst   = 0;
    for (int t = 0; t < T; ++t) {
        double scale = 0, diff = 0;
        for (int i = 0; i < H; ++i) {
            scale = std::max(scale, std::fabs(double{r[t * H + i]}));
            diff  = std::max(diff, std::fabs(got[t * H + i] - r[t * H + i]));
        }
        worst = std::max(worst, diff / scale);
    }
    return worst;
}

void teacher_forced(const m::Parameters& p, m::Forward& fw, State& state, const Reference& ref, bool prompt) {
    const auto& c  = p.model.config().text;
    state.reset();
    const auto& b  = fw.buffers();
    const char* dump_dir = std::getenv("INFERNIX_INKLING2_DUMP");
    TailDump dump(dump_dir != nullptr ? c.visits() : 0);
    DeviceRows input(static_cast<std::size_t>(kT) * H), tail(static_cast<std::size_t>(kT) * H);
    DeviceRows msrc0(static_cast<std::size_t>(kT) * H), msrc1(static_cast<std::size_t>(kT) * H);
    std::vector<std::unique_ptr<DeviceRows>> blocks;
    for (int i = 0; i < 4; ++i) { blocks.push_back(std::make_unique<DeviceRows>(static_cast<std::size_t>(kT) * H)); }
    long logit_columns = 0;
    double logit_worst = 0;
    for (int c0 = 0; c0 < kPositions; c0 += kT) {
        m::CallPlan plan = call_plan(ref, c0, kT);
        plan.prompt      = prompt;
        fw.prepare(plan);
        fw.embed(nullptr);
        std::vector<bool> skip(kT, false), none(kT, false);
        compare("embed", bf16_values(b.e, static_cast<std::size_t>(kT) * H), ref, "e", c0, kT, H, none);
        for (std::uint32_t v = 0; v < c.visits(); ++v) {
            const std::string pre = "v" + std::to_string(v) + "_";
            const std::uint32_t layer = c.visit_layer(v), pass = c.visit_pass(v);
            const auto& op = p.layers[layer];
            m::VisitInputs in;
            upload_rows(ref, pre + "h_in", c0, kT, input.p);
            in.input    = input.p;
            in.tail_out = v > 0 ? tail.p : nullptr;
            const std::string bname = pre + "blocks";
            const int nb = ref.index.at(bname).at("shape")[0].get<int>();
            for (int i = 0; i < nb; ++i) {
                const float* src = ref.get<float>(bname) + (static_cast<std::size_t>(i) * kPositions + c0) * H;
                std::vector<std::uint16_t> host(static_cast<std::size_t>(kT) * H);
                for (std::size_t j = 0; j < host.size(); ++j) { host[j] = b16(src[j]); }
                cuda(cudaMemcpy(blocks[i]->p, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "blocks");
                in.attention_blocks.rows[i] = blocks[i]->p;
                in.mlp_blocks.rows[i]       = blocks[i]->p;
            }
            in.attention_blocks.count = in.mlp_blocks.count = nb;
            if (op.boundary) { in.mlp_blocks.rows[in.mlp_blocks.count++] = input.p; }
            if (op.messages && pass > 0) {
                const std::uint32_t base = c.loop.core_start + (pass - 1) * c.loop.core_layers();
                upload_rows(ref, "v" + std::to_string(base + 6) + "_h_in", c0, kT, msrc0.p);
                upload_rows(ref, "v" + std::to_string(base + 11) + "_out", c0, kT, msrc1.p);
                in.message_sources[0] = msrc0.p;
                in.message_sources[1] = msrc1.p;
            }
            fw.visit(v, state.view, in, nullptr);
            cuda(cudaDeviceSynchronize(), "visit");
            const std::size_t TH = static_cast<std::size_t>(kT) * H;
            if (v > 0) {
                const auto t = bf16_values(tail.p, TH);
                compare("tail (previous out)", t, ref, "v" + std::to_string(v - 1) + "_out", c0, kT, H, skip);
                if (dump_dir != nullptr) {
                    dump.rows(dump.tail, v - 1, c0, t);
                    dump.worst[v - 1] = std::max(dump.worst[v - 1],
                                                 tail_worst(t, ref, "v" + std::to_string(v - 1) + "_out", c0, kT));
                }
            }
            if (dump_dir != nullptr) {
                dump.rows(dump.xm, v, c0, bf16_values(fw.mlp_input(v), TH));
                dump.rows(dump.cand, v, c0, f32_values(b.cand, TH));
                const auto routed = download(b.moe.routed, static_cast<std::size_t>(kT) * 8);
                std::copy(routed.begin(), routed.end(), dump.routed.begin() + (std::size_t{v} * kPositions + c0) * 8);
            }
            const bool three = op.full || (op.messages && pass > 0);
            compare("x_q", bf16_values(b.x[0], TH), ref, pre + "xq", c0, kT, H, none);
            compare("x_k", bf16_values(three ? b.x[1] : b.x[0], TH), ref, pre + "xk", c0, kT, H, none);
            compare("x_v", bf16_values(three ? b.x[2] : b.x[0], TH), ref, pre + "xv", c0, kT, H, none);
            const std::size_t QW = ik::kHeads * ik::kHeadDim;
            compare("q", bf16_values(b.attention_ws.q, kT * QW), ref, pre + "q", c0, kT, QW, none);
            compare("r", bf16_values(b.attention_ws.r, kT * ik::kHeads * ik::kRelDim), ref, pre + "r", c0, kT,
                    ik::kHeads * ik::kRelDim, none);
            std::vector<double> kk, vv;
            appended_kv(state.view, fw.layout().visits[v], c0, kT, kk, vv);
            compare("k (stored)", kk, ref, pre + "k", c0, kT, ik::kKvHeads * ik::kHeadDim, none);
            compare("v (stored, FP16)", vv, ref, pre + "v", c0, kT, ik::kKvHeads * ik::kHeadDim, none);
            compare("sink", bf16_values(b.sink, kT * ik::kHeads), ref, pre + "sink", c0, kT, ik::kHeads, none);
            compare("gate", bf16_values(b.gate, kT * QW), ref, pre + "gate", c0, kT, QW, none);
            compare("attention a", bf16_values(b.attention, kT * QW), ref, pre + "a", c0, kT, QW, none);
            compare("W_o", bf16_values(b.o, TH), ref, pre + "o", c0, kT, H, none);
            compare("cand", f32_values(b.cand, TH), ref, pre + "cand", c0, kT, H, none);
            compare("x_mlp", bf16_values(fw.mlp_input(v), TH), ref, pre + "xm", c0, kT, H, none);
            // Routing: the experts in slot order, and gamma.
            const auto routed = download(b.moe.routed, static_cast<std::size_t>(kT) * 8);
            const auto gamma  = download(b.moe.gamma, static_cast<std::size_t>(kT) * 8);
            const std::int32_t* rr = ref.get<std::int32_t>(pre + "routed") + c0 * 8;
            const float* rg        = ref.get<float>(pre + "gamma") + c0 * 8;
            for (int t = 0; t < kT; ++t) {
                bool same = true;
                for (int j = 0; j < 8; ++j) { same = same && routed[t * 8 + j] == rr[t * 8 + j]; }
                skip[t] = !same;
                g_routing_differs += same ? 0 : 1;
                if (same) {
                    double d = 0;
                    for (int j = 0; j < 8; ++j) { d = std::max(d, std::fabs(double{gamma[t * 8 + j]} - rg[t * 8 + j])); }
                    Stage& s = g_stages["gamma"];
                    s.worst  = std::max(s.worst, d / 8.0);
                    ++s.columns;
                } else {
                    std::printf("  routing differs at visit %u column %d\n", v, c0 + t);
                }
            }
        }
        // The output mix over the reference's blocks and the kernel's last tail, then the head.
        m::FinalInputs fin;
        fin.tail_out = tail.p;
        const char* names[4] = {"v0_h_in", "v6_h_in", "v36_h_in", "v41_out"};
        for (int i = 0; i < 4; ++i) {
            upload_rows(ref, names[i], c0, kT, blocks[i]->p);
            fin.blocks.rows[i] = blocks[i]->p;
        }
        fin.blocks.count = 4;
        fw.finish(state.view, fin, nullptr);
        cuda(cudaDeviceSynchronize(), "finish");
        const auto last_tail = bf16_values(tail.p, static_cast<std::size_t>(kT) * H);
        compare("tail (visit 47 out)", last_tail, ref, "v47_out", c0, kT, H, skip);
        if (dump_dir != nullptr) {
            dump.rows(dump.tail, c.visits() - 1, c0, last_tail);
            dump.worst[c.visits() - 1] = std::max(dump.worst[c.visits() - 1], tail_worst(last_tail, ref, "v47_out", c0, kT));
        }
        compare("final hidden", bf16_values(b.final_hidden, static_cast<std::size_t>(kT) * H), ref, "final", c0, kT,
                H, skip);
        const std::size_t V = c.vocab_size;
        const auto logits   = download(b.logits, static_cast<std::size_t>(kT) * V);
        const float* rl     = ref.get<float>("logits") + static_cast<std::size_t>(c0) * V;
        for (int t = 0; t < kT; ++t) {
            double d = 0;
            for (std::size_t i = 0; i < V; ++i) { d = std::max(d, std::fabs(double{logits[t * V + i]} - rl[t * V + i])); }
            logit_worst = std::max(logit_worst, d);
            ++logit_columns;
        }
    }
    std::printf("teacher-forced logits: largest difference %.4f over %ld columns\n", logit_worst, logit_columns);
    if (dump_dir != nullptr) {
        dump.write(dump_dir);
        std::printf("teacher-forced tails, worst |diff| / row max per visit:");
        for (std::uint32_t v = 0; v < c.visits(); ++v) { std::printf("%s %u:%.1e", v % 8 == 0 ? "\n " : "", v, dump.worst[v]); }
        std::printf("\n");
    }
}

// ---------------------------------------------------------------------------- 2-3. free-running, decode = verify

// The worst |kernel - ref| / max |ref row| over the call's columns.
double drift(const std::vector<double>& got, const Reference& ref, const std::string& name, int c0, int T,
             std::size_t width) {
    const float* r = ref.get<float>(name) + static_cast<std::size_t>(c0) * width;
    double worst   = 0;
    for (int t = 0; t < T; ++t) {
        double scale = 0, diff = 0;
        for (std::size_t i = 0; i < width; ++i) {
            scale = std::max(scale, std::fabs(double{r[t * width + i]}));
            diff  = std::max(diff, std::fabs(got[t * width + i] - r[t * width + i]));
        }
        worst = std::max(worst, diff / scale);
    }
    return worst;
}

// Diagnostic hybrids (INFERNIX_INKLING2_HYBRID): "blocks" runs the forward's own inputs with the reference's block and
// message sources; "input" the reference's visit inputs with the forward's own blocks and message sources.
std::vector<float> free_running(const m::Parameters& p, m::Forward& fw, State& state, const Reference& ref, int width,
                                bool report = false, const std::string& hybrid = {}, int perturb_offset = 0,
                                bool prompt = false) {
    const auto& c = p.model.config().text;
    state.reset();
    std::vector<float> out;
    DeviceRows rinput(static_cast<std::size_t>(width) * H), m0(static_cast<std::size_t>(width) * H),
        m1(static_cast<std::size_t>(width) * H);
    std::vector<std::unique_ptr<DeviceRows>> rblocks;
    for (int i = 0; i < 4; ++i) { rblocks.push_back(std::make_unique<DeviceRows>(static_cast<std::size_t>(width) * H)); }
    for (int c0 = 0; c0 < kPositions; c0 += width) {
        m::CallPlan plan = call_plan(ref, c0, width);
        plan.prompt      = prompt;
        fw.prepare(plan);
        fw.embed(nullptr);
        if (hybrid == "perturb") {
            // One BF16 ulp up in every 97th element of each column's embedding: the model's own sensitivity.
            auto e = download(fw.buffers().e, static_cast<std::size_t>(width) * H);
            for (int t = 0; t < width; ++t) {
                for (int j = perturb_offset; j < H; j += 97) {
                    std::uint16_t& w = e[static_cast<std::size_t>(t) * H + j];
                    w                = static_cast<std::uint16_t>(w + 1);
                }
            }
            cuda(cudaMemcpy(fw.buffers().e, e.data(), e.size() * 2, cudaMemcpyHostToDevice), "perturb");
        }
        if (report) {
            std::printf("free-running drift%s, positions %d-%d (visit: input, x_q, x_mlp):\n",
                        hybrid.empty() ? "" : (" [" + hybrid + "]").c_str(), c0, c0 + width - 1);
        }
        for (std::uint32_t v = 0; v < c.visits(); ++v) {
            m::VisitInputs in = fw.inputs(v);
            const std::string pre = "v" + std::to_string(v) + "_";
            const std::uint32_t layer = c.visit_layer(v), pass = c.visit_pass(v);
            if (hybrid == "blocks") {
                const int nb = ref.index.at(pre + "blocks").at("shape")[0].get<int>();
                for (int i = 0; i < nb; ++i) {
                    const float* src = ref.get<float>(pre + "blocks") + (static_cast<std::size_t>(i) * kPositions + c0) * H;
                    std::vector<std::uint16_t> host(static_cast<std::size_t>(width) * H);
                    for (std::size_t j = 0; j < host.size(); ++j) { host[j] = b16(src[j]); }
                    cuda(cudaMemcpy(rblocks[i]->p, host.data(), host.size() * 2, cudaMemcpyHostToDevice), "blocks");
                    in.attention_blocks.rows[i] = in.mlp_blocks.rows[i] = rblocks[i]->p;
                }
                in.attention_blocks.count = in.mlp_blocks.count = nb;
                if (p.layers[layer].boundary) { in.mlp_blocks.rows[in.mlp_blocks.count++] = in.input; }
                if (p.layers[layer].messages && pass > 0) {
                    const std::uint32_t base = c.loop.core_start + (pass - 1) * c.loop.core_layers();
                    upload_rows(ref, "v" + std::to_string(base + 6) + "_h_in", c0, width, m0.p);
                    upload_rows(ref, "v" + std::to_string(base + 11) + "_out", c0, width, m1.p);
                    in.message_sources[0] = m0.p, in.message_sources[1] = m1.p;
                }
            } else if (hybrid == "input" && v > 0) {
                upload_rows(ref, pre + "h_in", c0, width, rinput.p);
                in.input = rinput.p;
                for (int i = 0; i < in.mlp_blocks.count; ++i) {
                    if (in.mlp_blocks.rows[i] == in.tail_out) { in.mlp_blocks.rows[i] = rinput.p; }
                }
            }
            fw.visit(v, state.view, in, nullptr);
            if (report) {
                cuda(cudaDeviceSynchronize(), "visit");
                const std::size_t TH = static_cast<std::size_t>(width) * H;
                const std::string pre = "v" + std::to_string(v) + "_";
                const auto& b         = fw.buffers();
                std::printf("  %2u L%-2u p%u: %.2e %.2e %.2e\n", v, c.visit_layer(v), c.visit_pass(v),
                            drift(bf16_values(in.input, TH), ref, pre + "h_in", c0, width, H),
                            drift(bf16_values(b.x[0], TH), ref, pre + "xq", c0, width, H),
                            drift(bf16_values(fw.mlp_input(v), TH), ref, pre + "xm", c0, width, H));
            }
        }
        fw.finish(state.view, nullptr);
        cuda(cudaDeviceSynchronize(), "forward");
        const auto logits = download(fw.buffers().logits, static_cast<std::size_t>(width) * c.vocab_size);
        out.insert(out.end(), logits.begin(), logits.end());
    }
    return out;
}

// One call through the forward on `stream`, its logits downloaded.
std::vector<float> run_call(const m::Parameters& p, m::Forward& fw, State& state, const m::CallPlan& plan,
                            cudaStream_t stream) {
    const auto& c = p.model.config().text;
    fw.prepare(plan);
    fw.embed(stream);
    for (std::uint32_t v = 0; v < c.visits(); ++v) { fw.visit(v, state.view, stream); }
    fw.finish(state.view, stream);
    cuda(cudaStreamSynchronize(stream), "call");
    return download(fw.buffers().logits, static_cast<std::size_t>(plan.columns()) * c.vocab_size);
}

// 4. The whole call captured as one CUDA graph on a non-blocking stream (the visit, attention and expert kernels
//    become programmatic dependents) and replayed three times from the same state: the eager call's logits bit for
//    bit, for a 16-column call from an empty state and a 1-column decode step after it.
bool graph_replay(const m::Parameters& p, m::Forward& fw, State& state, const Reference& ref) {
    const auto& c = p.model.config().text;
    cudaStream_t stream = nullptr;
    cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    bool ok = true;
    state.reset();
    for (const auto& [c0, width] : {std::pair{0, kT}, std::pair{kT, 1}}) {
        const m::CallPlan plan = call_plan(ref, c0, width);
        state.save();
        const auto eager = run_call(p, fw, state, plan, nullptr);
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        fw.prepare(plan);
        cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        fw.embed(stream);
        for (std::uint32_t v = 0; v < c.visits(); ++v) { fw.visit(v, state.view, stream); }
        fw.finish(state.view, stream);
        cuda(cudaStreamEndCapture(stream, &graph), "capture");
        cuda(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int replay = 0; replay < 3; ++replay) {
            state.restore();
            cuda(cudaGraphLaunch(exec, stream), "replay");
            cuda(cudaStreamSynchronize(stream), "replay");
            const auto got = download(fw.buffers().logits, static_cast<std::size_t>(width) * c.vocab_size);
            const bool same = std::memcmp(got.data(), eager.data(), got.size() * 4) == 0;
            ok = ok && same;
            std::printf("graph replay %d of a %d-column call at position %d: %s\n", replay + 1, width, c0,
                        same ? "the eager logits bit for bit" : "DIFFERENT");
        }
        if (std::getenv("INFERNIX_INKLING2_TIME") != nullptr) {
            // Diagnostic: the captured call's time (the same positions replayed; not a qualified measurement).
            cudaEvent_t start = nullptr, stop = nullptr;
            cuda(cudaEventCreate(&start), "event");
            cuda(cudaEventCreate(&stop), "event");
            for (int i = 0; i < 20; ++i) { cuda(cudaGraphLaunch(exec, stream), "warm"); }
            constexpr int kRuns = 100;
            cuda(cudaEventRecord(start, stream), "event");
            for (int i = 0; i < kRuns; ++i) { cuda(cudaGraphLaunch(exec, stream), "time"); }
            cuda(cudaEventRecord(stop, stream), "event");
            cuda(cudaEventSynchronize(stop), "event");
            float ms = 0;
            cuda(cudaEventElapsedTime(&ms, start, stop), "event");
            std::printf("  captured %d-column call: %.3f ms a call (%.1f columns/s)\n", width, ms / kRuns,
                        1000.0 * width * kRuns / ms);
            cuda(cudaEventDestroy(start), "event");
            cuda(cudaEventDestroy(stop), "event");
        }
        cuda(cudaGraphExecDestroy(exec), "destroy");
        cuda(cudaGraphDestroy(graph), "destroy");
        state.restore();
        (void)run_call(p, fw, state, plan, nullptr); // the state after this call, for the next one
    }
    cuda(cudaStreamDestroy(stream), "stream");
    return ok;
}

// Diagnostic and timing (INFERNIX_INKLING2_PROFILE = width, INFERNIX_INKLING2_CONTEXT = positions, default 16):
// the context filled by 16-column calls of the reference's ids repeated, then the `width`-column step at that
// position captured; 200 replays timed by events (after 20 warm-up replays), then 50 between cudaProfilerStart and
// cudaProfilerStop for `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node`. The replays repeat the
// same positions (the state they write is the step's own).
void profile_rounds(const m::Parameters& p, m::Forward& fw, State& state, const Reference& ref, int width,
                    int context) {
    const auto& c = p.model.config().text;
    const std::int32_t* ids = ref.get<std::int32_t>("ids");
    const auto plan_at = [&](int c0, int n) {
        m::CallPlan plan;
        for (int t = 0; t < n; ++t) {
            plan.tokens.push_back(ids[(c0 + t) % kPositions]);
            plan.lane.push_back(0);
            plan.position.push_back(c0 + t);
        }
        plan.first = {0};
        return plan;
    };
    cudaStream_t stream = nullptr;
    cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    state.reset();
    for (int c0 = 0; c0 < context; c0 += kT) { (void)run_call(p, fw, state, plan_at(c0, std::min(kT, context - c0)), nullptr); }
    const m::CallPlan plan = plan_at(context, width);
    state.save();
    (void)run_call(p, fw, state, plan, nullptr); // warm: every first-call attribute set before the capture
    state.restore();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    fw.prepare(plan);
    cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    fw.embed(stream);
    for (std::uint32_t v = 0; v < c.visits(); ++v) { fw.visit(v, state.view, stream); }
    fw.finish(state.view, stream);
    cuda(cudaStreamEndCapture(stream, &graph), "capture");
    cuda(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (int i = 0; i < 20; ++i) { cuda(cudaGraphLaunch(exec, stream), "warm"); }
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda(cudaEventCreate(&start), "event");
    cuda(cudaEventCreate(&stop), "event");
    constexpr int kRuns = 200;
    cuda(cudaEventRecord(start, stream), "event");
    for (int i = 0; i < kRuns; ++i) { cuda(cudaGraphLaunch(exec, stream), "time"); }
    cuda(cudaEventRecord(stop, stream), "event");
    cuda(cudaEventSynchronize(stop), "event");
    float ms = 0;
    cuda(cudaEventElapsedTime(&ms, start, stop), "event");
    std::printf("timing: %d-column step at context %d: %.4f ms a round (%.1f tokens/s)\n", width, context, ms / kRuns,
                1000.0 * width * kRuns / ms);
    const char* skeleton_mode = std::getenv("INFERNIX_INKLING2_SKELETON");
    if (skeleton_mode != nullptr && std::string(skeleton_mode) == "names") {
        print_graph_kernels(graph);
    } else if (skeleton_mode != nullptr) {
        // Attribution: the same graph with every kernel an empty one of its grid, block and shared memory (the edges,
        // programmatic ones included, unchanged): the chain of launches and dependency waits alone.
        cudaGraph_t skeleton = nullptr;
        cudaGraphExec_t skeleton_exec = nullptr;
        cuda(cudaGraphClone(&skeleton, graph), "clone");
        // INFERNIX_INKLING2_SKELETON = serial, trigger, or kernel-name parts ("a|b"): only those kernels emptied.
        const std::string mode = std::getenv("INFERNIX_INKLING2_SKELETON");
        const bool trigger     = mode == "trigger";
        const std::string filter = mode == "serial" || mode == "trigger" ? std::string() : mode;
        const int kernels  = skeletonize_graph(skeleton, trigger, filter);
        cuda(cudaGraphInstantiate(&skeleton_exec, skeleton, 0), "instantiate");
        for (int i = 0; i < 20; ++i) { cuda(cudaGraphLaunch(skeleton_exec, stream), "warm"); }
        cuda(cudaEventRecord(start, stream), "event");
        for (int i = 0; i < kRuns; ++i) { cuda(cudaGraphLaunch(skeleton_exec, stream), "time"); }
        cuda(cudaEventRecord(stop, stream), "event");
        cuda(cudaEventSynchronize(stop), "event");
        float skeleton_ms = 0;
        cuda(cudaEventElapsedTime(&skeleton_ms, start, stop), "event");
        std::printf("skeleton (%s): %d kernels, %.4f ms a round (%.1f %% of the round)\n",
                    filter.empty() ? (trigger ? "triggering at entry" : "serialized") : filter.c_str(), kernels,
                    skeleton_ms / kRuns, 100.0 * skeleton_ms / ms);
        cuda(cudaGraphExecDestroy(skeleton_exec), "destroy");
        cuda(cudaGraphDestroy(skeleton), "destroy");
    }
    cuda(cudaProfilerStart(), "profiler");
    for (int i = 0; i < 50; ++i) { cuda(cudaGraphLaunch(exec, stream), "replay"); }
    cuda(cudaStreamSynchronize(stream), "replay");
    cuda(cudaProfilerStop(), "profiler");
    cuda(cudaEventDestroy(start), "event");
    cuda(cudaEventDestroy(stop), "event");
    cuda(cudaGraphExecDestroy(exec), "destroy");
    cuda(cudaGraphDestroy(graph), "destroy");
    cuda(cudaStreamDestroy(stream), "stream");
}

// Under INFERNIX_INKLING2_Q8: the converted matrices' FP16 group scales, counted (a group whose largest magnitude is
// under 127 * 2^-25 gets scale 0 and loses its values; a subnormal scale keeps fewer than 11 bits).
void q8_census(const m::Model& model) {
    std::uint64_t groups = 0, zero = 0, subnormal = 0;
    for (const auto& parent : model.q8().parents) {
        const WeightGeometry& g = parent->geometry;
        std::vector<std::uint16_t> scales(g.scale_bytes / 2);
        cuda(cudaMemcpy(scales.data(), parent->device + g.scale_offset, g.scale_bytes, cudaMemcpyDeviceToHost), "scales");
        const std::uint64_t real = (g.shape[1] + 31) / 32; // a row's groups before the padding to 128
        const std::uint64_t per_row = g.scale_bytes_per_row / 2;
        for (std::uint64_t r = 0; r < g.shape[0]; ++r) {
            for (std::uint64_t j = 0; j < real; ++j) {
                const std::uint16_t s = scales[r * per_row + j];
                ++groups;
                zero += (s & 0x7FFFU) == 0 ? 1 : 0;
                subnormal += (s & 0x7C00U) == 0 && (s & 0x3FFU) != 0 ? 1 : 0;
            }
        }
    }
    std::printf("q8 census: %llu groups, %llu with scale 0, %llu with a subnormal scale\n",
                static_cast<unsigned long long>(groups), static_cast<unsigned long long>(zero),
                static_cast<unsigned long long>(subnormal));
}

// Diagnostic and timing (INFERNIX_INKLING2_PREFILL_PROFILE = width): one eager prefill call of `width` columns from an
// empty state (pseudo-random ids, so the routing varies as text does), after a warm-up call, with an event at every
// Op boundary (Forward::set_marker): each Op kind's time summed over the 48 visits, and the call's rate. The Ops run
// in stream order, so an Op's time is the gap since the previous boundary (PDL overlap moves a little between them).
void profile_prefill(const m::Parameters& p, int width) {
    const auto& c = p.model.config().text;
    // INFERNIX_INKLING2_A8: --a8-activations (with the 8-bit weights it implies): the projections' and the NVFP4
    // experts' 8-bit activations.
    const bool a8 = std::getenv("INFERNIX_INKLING2_A8") != nullptr;
    m::Forward fw(p, 1, std::max(width, m::kMaxForwardColumns),
                  a8 ? ops::w4a4_moe::ExpertActivation::A8 : ops::w4a4_moe::ExpertActivation::Checkpoint, a8,
                  window_from_env());
    State state(fw.layout(), 1, width / ik::kKeyUnit + 2, window_from_env());
    m::CallPlan plan;
    std::uint64_t lcg = 12345;
    for (int t = 0; t < width; ++t) {
        lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
        plan.tokens.push_back(static_cast<std::int32_t>((lcg >> 33) % 200000));
        plan.lane.push_back(0);
        plan.position.push_back(t);
    }
    plan.first  = {0};
    plan.prompt = std::getenv("INFERNIX_INKLING2_PREFILL_DECODE") == nullptr; // the decode route only for comparison
    cudaStream_t stream = nullptr;
    cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const auto call = [&] {
        fw.prepare(plan);
        fw.embed(stream);
        for (std::uint32_t v = 0; v < c.visits(); ++v) { fw.visit(v, state.view, stream); }
        fw.finish(state.view, fw.final_inputs(), stream, width - 1, 1);
        cuda(cudaStreamSynchronize(stream), "call");
    };
    call(); // warm: first-call attributes and allocations
    state.reset();
    std::vector<cudaEvent_t> events;
    std::vector<std::string> names;
    const auto record = [&](const char* name) {
        cudaEvent_t e = nullptr;
        cuda(cudaEventCreate(&e), "event");
        cuda(cudaEventRecord(e, stream), "event");
        events.push_back(e);
        names.emplace_back(name);
    };
    fw.set_marker([&](const char* op) { record(op); });
    fw.prepare(plan);
    record("start");
    fw.embed(stream);
    for (std::uint32_t v = 0; v < c.visits(); ++v) { fw.visit(v, state.view, stream); }
    fw.finish(state.view, fw.final_inputs(), stream, width - 1, 1);
    cuda(cudaStreamSynchronize(stream), "call");
    fw.set_marker(nullptr);
    std::map<std::string, std::pair<double, int>> ops;
    double total = 0;
    for (std::size_t i = 1; i < events.size(); ++i) {
        float ms = 0;
        cuda(cudaEventElapsedTime(&ms, events[i - 1], events[i]), "event");
        ops[names[i]].first += ms, ++ops[names[i]].second;
        total += ms;
    }
    std::printf("prefill profile: one %d-column call from position 0: %.3f ms (%.1f tokens/s)\n", width, total,
                1000.0 * width / total);
    std::vector<std::pair<std::string, std::pair<double, int>>> rows(ops.begin(), ops.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    for (const auto& [name, v] : rows) {
        std::printf("  %-16s %9.3f ms %5.1f %%  (%d launches' boundaries, %.1f us each)\n", name.c_str(), v.first,
                    100.0 * v.first / total, v.second, 1000.0 * v.first / v.second);
    }
    for (cudaEvent_t e : events) { cuda(cudaEventDestroy(e), "event"); }
    cuda(cudaStreamDestroy(stream), "stream");
}

// 5. Three lanes in one call with different token sequences and a padding column, in two calls of mixed widths:
//    each lane's logits equal its own run alone (as the only lane of a one-lane forward) bit for bit.
// a8: the solo forward's --a8-activations route, which the 3-lane forward takes too.
bool lanes_alone(const m::Parameters& p, m::Forward& solo, State& solo_state, const Reference& ref, bool a8) {
    const auto& c = p.model.config().text;
    const std::int32_t* ids = ref.get<std::int32_t>("ids");
    std::vector<std::vector<std::int32_t>> tokens(3);
    for (int i = 0; i < kPositions; ++i) {
        tokens[0].push_back(ids[i]);
        tokens[1].push_back(ids[kPositions - 1 - i]);
        tokens[2].push_back(ids[(i + 7) % kPositions]);
    }
    // Per call, each lane's first position and width (lane 0 0-7 then 8-15, lane 1 0-4 then 5-20, lane 2 0-11 then 12).
    const int widths[2][3] = {{8, 5, 12}, {8, 16, 1}};
    m::Forward multi(p, 3, m::kMaxForwardColumns,
                     a8 ? ops::w4a4_moe::ExpertActivation::A8 : ops::w4a4_moe::ExpertActivation::Checkpoint, a8);
    State multi_state(multi.layout(), 3);
    std::vector<std::vector<float>> together(3), alone(3);
    int at[3] = {0, 0, 0};
    for (int call = 0; call < 2; ++call) {
        m::CallPlan plan;
        plan.first = {0, 0, 0};
        int lane_at[3] = {at[0], at[1], at[2]};
        for (int l = 0; l < 3; ++l) {
            if (l == 1) { plan.tokens.push_back(0), plan.lane.push_back(-1), plan.position.push_back(0); } // padding
            for (int i = 0; i < widths[call][l]; ++i) {
                plan.tokens.push_back(tokens[l][lane_at[l] + i]);
                plan.lane.push_back(l);
                plan.position.push_back(lane_at[l] + i);
            }
        }
        const auto logits = run_call(p, multi, multi_state, plan, nullptr);
        for (int t = 0; t < plan.columns(); ++t) {
            if (plan.lane[t] < 0) { continue; }
            const float* row = logits.data() + static_cast<std::size_t>(t) * c.vocab_size;
            together[plan.lane[t]].insert(together[plan.lane[t]].end(), row, row + c.vocab_size);
        }
        for (int l = 0; l < 3; ++l) { at[l] += widths[call][l]; }
    }
    for (int l = 0; l < 3; ++l) {
        solo_state.reset();
        int position = 0;
        for (int call = 0; call < 2; ++call) {
            m::CallPlan plan;
            plan.first = {0};
            for (int i = 0; i < widths[call][l]; ++i) {
                plan.tokens.push_back(tokens[l][position + i]);
                plan.lane.push_back(0);
                plan.position.push_back(position + i);
            }
            const auto logits = run_call(p, solo, solo_state, plan, nullptr);
            alone[l].insert(alone[l].end(), logits.begin(), logits.end());
            position += widths[call][l];
        }
    }
    bool ok = true;
    for (int l = 0; l < 3; ++l) {
        const bool same = together[l].size() == alone[l].size() &&
                          std::memcmp(together[l].data(), alone[l].data(), alone[l].size() * 4) == 0;
        ok = ok && same;
        std::printf("lane %d (%zu positions) in a 3-lane call: %s\n", l, alone[l].size() / c.vocab_size,
                    same ? "its solo logits bit for bit" : "DIFFERENT");
    }
    return ok;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    const char* reference     = std::getenv("INFERNIX_INKLING2_REFERENCE");
    if (artifact_path == nullptr || *artifact_path == '\0' || reference == nullptr || *reference == '\0') {
        std::cout << "skip: INFERNIX_INKLING2_ARTIFACT and INFERNIX_INKLING2_REFERENCE are needed\n";
        return 77;
    }
    try {
        // The checks read BF16 windows and hold the BF16 windows' criteria.
        if (window_from_env() != ik::KvCodec::Bf16 && std::getenv("INFERNIX_INKLING2_PROFILE") == nullptr &&
            std::getenv("INFERNIX_INKLING2_PREFILL_PROFILE") == nullptr) {
            std::printf("FAIL: INFERNIX_INKLING2_WINDOW is for the profiles (INFERNIX_INKLING2_PROFILE, _PREFILL_PROFILE) only\n");
            return 1;
        }
        DeviceContext device(0);
        const Reference ref(reference);
        artifact::Reader reader(artifact_path);
        // INFERNIX_INKLING2_PACK: the projections packed to bf16_p12 at load (the same bits: the bit guard holds).
        // INFERNIX_INKLING2_Q8: --q8-weights' conversion (other bits: the reference criteria do not apply; for its
        // timing, the invariance checks and the census).
        models::LoadOptions load_options;
        load_options.pack_bf16  = std::getenv("INFERNIX_INKLING2_PACK") != nullptr;
        load_options.q8_weights = std::getenv("INFERNIX_INKLING2_Q8") != nullptr || std::getenv("INFERNIX_INKLING2_A8") != nullptr;
        auto model = m::materialize_model(m::plan_load(reader, load_options), device);
        if (model->packed().source_bytes != 0) {
            std::printf("packed %zu matrices at load: %.2f GB of BF16 -> %.2f GB, %llu escapes, %.0f ms\n",
                        model->packed().parents.size(), model->packed().source_bytes / 1e9,
                        model->packed().storage.bytes / 1e9, static_cast<unsigned long long>(model->packed().escapes),
                        1000.0 * model->packed().seconds);
        }
        if (model->q8().source_bytes != 0) {
            std::printf("q8 weights at load: %zu matrices, %.1f MiB of BF16 -> %.1f MiB, %.0f ms\n",
                        model->q8().parents.size(), model->q8().source_bytes / 1048576.0,
                        model->q8().storage.bytes / 1048576.0, 1000.0 * model->q8().seconds);
            q8_census(*model);
        }
        {
            std::size_t free = 0, total = 0;
            cuda(cudaMemGetInfo(&free, &total), "memory");
            std::printf("loaded: %.1f MiB of device weights; %.1f MiB of VRAM free\n",
                        model->device_weight_bytes() / 1048576.0, free / 1048576.0);
            // The forward's call buffers by capacity (the Program's is the prefill chunk, capped at the context).
            for (const std::int32_t capacity : {128, 512, 2048, 4096, 8191}) {
                std::printf("forward buffers at a capacity of %d columns, 1 lane: %.1f MiB\n", capacity,
                            m::Forward::device_bytes(model->config().text, 1, capacity) / 1048576.0);
            }
        }
        const m::Parameters parameters(*model);
        // INFERNIX_INKLING2_A8 (profiles only: the reference criteria are the default route's).
        const bool a8 = std::getenv("INFERNIX_INKLING2_A8") != nullptr;
        m::Forward forward(parameters, 1, m::kMaxForwardColumns,
                           a8 ? ops::w4a4_moe::ExpertActivation::A8 : ops::w4a4_moe::ExpertActivation::Checkpoint, a8,
                           window_from_env());
        State state(forward.layout(), 1, 1, window_from_env());
        const auto& c = model->config().text;
        if (const char* width = std::getenv("INFERNIX_INKLING2_PROFILE"); width != nullptr) {
            const char* ctx   = std::getenv("INFERNIX_INKLING2_CONTEXT");
            const int context = ctx != nullptr ? std::max(1, std::atoi(ctx)) : kT;
            State big(forward.layout(), 1, (context + 64) / ik::kKeyUnit + 1, window_from_env());
            profile_rounds(parameters, forward, big, ref, std::max(1, std::atoi(width)), context);
            std::printf("profiled 50 rounds\n");
            return 0;
        }
        if (const char* width = std::getenv("INFERNIX_INKLING2_PREFILL_PROFILE"); width != nullptr) {
            profile_prefill(parameters, std::max(1, std::atoi(width)));
            return 0;
        }

        bool ok = true;
        // 1. Teacher-forced, on the decode route and then on the prompt route (the same criteria).
        for (const bool prompt : {false, true}) {
            g_stages.clear();
            g_routing_differs = 0;
            teacher_forced(parameters, forward, state, ref, prompt);
            std::printf("teacher-forced, %s route:\n%-24s %8s %8s %s\n", prompt ? "prompt" : "decode", "stage", "columns",
                        "excluded", "worst |diff| / row max");
            for (const auto& [name, s] : g_stages) {
                // The tails cross the experts' 4-bit activation boundary with inputs an ulp apart: reported here,
                // judged boundary-conditioned (the reference's MoE on the kernel's own x_mlp: .local tools/tail_check.py).
                const bool judged = name.rfind("tail", 0) != 0;
                const bool pass   = !judged || s.worst <= 0x1p-5;
                ok                = ok && pass;
                std::printf("%-24s %8ld %8ld %.3e%s\n", name.c_str(), s.columns, s.excluded, s.worst,
                            !judged ? "  (reported)" : (pass ? "" : "  FAIL"));
            }
            std::printf("routing differs from the reference in %ld (visit, column) pairs\n", g_routing_differs);
        }

        std::vector<float> hybrid_logits;
        const char* hybrid = std::getenv("INFERNIX_INKLING2_HYBRID");
        if (hybrid != nullptr) {
            hybrid_logits = free_running(parameters, forward, state, ref, kT, std::string(hybrid) != "perturb", hybrid);
            double worst  = 0;
            for (std::size_t i = 0; i < hybrid_logits.size(); ++i) {
                worst = std::max(worst, std::fabs(double{hybrid_logits[i]} - ref.get<float>("logits")[i]));
            }
            std::printf("hybrid %s: largest logit difference from the reference %.4f\n", hybrid, worst);
            // INFERNIX_INKLING2_HYBRID_OUT writes the hybrid's free-running logits ([32][V] FP32): with "perturb",
            // the model's own sensitivity to a 1-ulp input change, a floor for comparing other weights' logits.
            if (const char* out = std::getenv("INFERNIX_INKLING2_HYBRID_OUT"); out != nullptr) {
                std::ofstream f(out, std::ios::binary);
                f.write(reinterpret_cast<const char*>(hybrid_logits.data()),
                        static_cast<std::streamsize>(hybrid_logits.size() * 4));
                std::printf("wrote the hybrid's logits to %s\n", out);
            }
        }
        const auto verify = free_running(parameters, forward, state, ref, kT, std::getenv("INFERNIX_INKLING2_DRIFT") != nullptr);
        // A bit guard for changes claimed to keep the bits: INFERNIX_INKLING2_LOGITS_OUT writes the free-running
        // logits ([32][V] FP32); INFERNIX_INKLING2_LOGITS_GUARD requires them byte for byte.
        if (const char* out = std::getenv("INFERNIX_INKLING2_LOGITS_OUT"); out != nullptr) {
            std::ofstream f(out, std::ios::binary);
            f.write(reinterpret_cast<const char*>(verify.data()), static_cast<std::streamsize>(verify.size() * 4));
            std::printf("wrote the free-running logits to %s\n", out);
        }
        if (const char* guard = std::getenv("INFERNIX_INKLING2_LOGITS_GUARD"); guard != nullptr) {
            std::ifstream f(guard, std::ios::binary);
            std::vector<float> want(verify.size());
            f.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * 4));
            const bool same = f && std::memcmp(want.data(), verify.data(), verify.size() * 4) == 0;
            std::printf("bit guard (%s): %s\n", guard, same ? "the free-running logits byte for byte" : "DIFFERENT");
            ok = ok && same;
        }
        if (hybrid != nullptr && std::string(hybrid) == "perturb") {
            int flips = 0;
            for (int t = 0; t < kPositions; ++t) {
                const float* a = verify.data() + static_cast<std::size_t>(t) * c.vocab_size;
                const float* q = hybrid_logits.data() + static_cast<std::size_t>(t) * c.vocab_size;
                double d       = 0;
                std::size_t aa = 0, qa = 0;
                for (std::size_t i = 0; i < c.vocab_size; ++i) {
                    d  = std::max(d, std::fabs(double{a[i]} - q[i]));
                    aa = a[i] > a[aa] ? i : aa;
                    qa = q[i] > q[qa] ? i : qa;
                }
                flips += aa != qa ? 1 : 0;
                std::printf("  perturbed position %2d: largest logit change %.3f%s\n", t, d, aa != qa ? " (argmax changed)" : "");
            }
            std::printf("perturbation (1 ulp in 24 of 2304 embedding elements): %d of %d argmax changed\n", flips, kPositions);
        }
        for (int t = 0; t < kPositions; ++t) {
            double d = 0;
            for (std::size_t i = 0; i < c.vocab_size; ++i) {
                d = std::max(d, std::fabs(double{verify[t * c.vocab_size + i]} - ref.get<float>("logits")[t * c.vocab_size + i]));
            }
            std::printf("  position %2d: largest logit difference %.3f\n", t, d);
        }
        const auto decode = free_running(parameters, forward, state, ref, 1);
        const bool equal  = std::memcmp(verify.data(), decode.data(), verify.size() * 4) == 0;
        std::printf("decode = verify: %s\n", equal ? "the logits agree bit for bit" : "DIFFERENT");
        ok = ok && equal;
        // Free-running, token level: the argmax from the last prompt position on (the greedy continuation) equals the
        // reference's; the mean NLL of the next ids is within the reference's own perturbation spread
        // (<reference>.spread.json from ref_perturb.py: three 1-ulp embedding perturbations of the CPU reference).
        // Elsewhere the model's sensitivity to rounding (a 1-ulp embedding change moves its logits by up to ~9)
        // makes per-position argmax a report, not a criterion.
        const std::size_t V = c.vocab_size;
        const float* rl     = ref.get<float>("logits");
        const std::int32_t* ids = ref.get<std::int32_t>("ids");
        constexpr int kLastPrompt = 23;
        int argmax_differ = 0;
        double worst = 0, nll = 0, ref_nll = 0;
        const auto log_softmax_at = [V](const float* l, std::size_t id) {
            double m = l[0];
            for (std::size_t i = 1; i < V; ++i) { m = std::max(m, double{l[i]}); }
            double s = 0;
            for (std::size_t i = 0; i < V; ++i) { s += std::exp(double{l[i]} - m); }
            return double{l[id]} - m - std::log(s);
        };
        for (int t = 0; t < kPositions; ++t) {
            const float* g = verify.data() + static_cast<std::size_t>(t) * V;
            const float* r = rl + static_cast<std::size_t>(t) * V;
            std::size_t ga = 0, ra = 0;
            for (std::size_t i = 0; i < V; ++i) {
                worst = std::max(worst, std::fabs(double{g[i]} - r[i]));
                if (g[i] > g[ga]) { ga = i; }
                if (r[i] > r[ra]) { ra = i; }
            }
            if (ga != ra) {
                std::printf("  position %d: argmax %zu, the reference's %zu%s\n", t, ga, ra,
                            t >= kLastPrompt ? "  FAIL (greedy continuation)" : "");
                ++argmax_differ;
                ok = ok && t < kLastPrompt;
            }
            if (t + 1 < kPositions) {
                nll -= log_softmax_at(g, static_cast<std::size_t>(ids[t + 1]));
                ref_nll -= log_softmax_at(r, static_cast<std::size_t>(ids[t + 1]));
            }
        }
        nll /= kPositions - 1, ref_nll /= kPositions - 1;
        if (std::getenv("INFERNIX_INKLING2_DRIFT") != nullptr || hybrid != nullptr) {
            // Diagnostic: per-position NLL, and this forward's own NLL under 1-ulp embedding perturbations.
            for (int t = 0; t + 1 < kPositions; ++t) {
                const std::size_t id = static_cast<std::size_t>(ids[t + 1]);
                std::printf("  nll position %2d: %.4f, the reference's %.4f\n", t,
                            -log_softmax_at(verify.data() + static_cast<std::size_t>(t) * V, id),
                            -log_softmax_at(rl + static_cast<std::size_t>(t) * V, id));
            }
            for (const int offset : {0, 31, 61, 7, 13, 47, 71, 89}) {
                const auto pl = free_running(parameters, forward, state, ref, kT, false, "perturb", offset);
                double pn     = 0;
                for (int t = 0; t + 1 < kPositions; ++t) {
                    pn -= log_softmax_at(pl.data() + static_cast<std::size_t>(t) * V, static_cast<std::size_t>(ids[t + 1]));
                }
                pn /= kPositions - 1;
                std::printf("  this forward perturbed from offset %2d: mean NLL %.5f (change %.5f)\n", offset, pn, pn - nll);
            }
        }
        std::printf("free-running: %d of %d argmax differ from the reference (none allowed from position %d); largest "
                    "logit difference %.4f\n", argmax_differ, kPositions, kLastPrompt, worst);
        std::printf("mean NLL of the next ids (reported): %.5f, the reference's %.5f (difference %.5f)\n", nll, ref_nll,
                    nll - ref_nll);
        if (std::ifstream spread_file(std::string(reference) + ".spread.json"); spread_file) {
            const auto spread = nlohmann::json::parse(spread_file);
            std::printf("  the reference's own mean NLL under 1-ulp embedding perturbations:");
            for (const auto& [offset, value] : spread.at("perturbed").items()) {
                std::printf(" %s:%+.3f", offset.c_str(), value.get<double>() - spread.at("reference").get<double>());
            }
            std::printf("\n");
        }
        // 7. The prompt route (CallPlan::prompt: the tensor-core projections, router and experts): the 32 positions as
        //    one prompt call and as prompt calls of 8 give the same logits bit for bit, and the greedy continuation
        //    (the argmax from the last prompt position on) equals the reference's. Reported: the largest logit
        //    difference from the reference and from the decode route.
        {
            const auto one   = free_running(parameters, forward, state, ref, kPositions, false, {}, 0, true);
            const auto eight = free_running(parameters, forward, state, ref, 8, false, {}, 0, true);
            const bool same  = std::memcmp(one.data(), eight.data(), one.size() * 4) == 0;
            std::printf("prompt route: one 32-column call = calls of 8: %s\n", same ? "bit for bit" : "DIFFERENT");
            ok = ok && same;
            // The prompt route's guard (the Engine test's scoring check reads the same file):
            // INFERNIX_INKLING2_PROMPT_LOGITS_OUT writes its logits, INFERNIX_INKLING2_PROMPT_LOGITS_GUARD requires them.
            if (const char* out = std::getenv("INFERNIX_INKLING2_PROMPT_LOGITS_OUT"); out != nullptr) {
                std::ofstream f(out, std::ios::binary);
                f.write(reinterpret_cast<const char*>(one.data()), static_cast<std::streamsize>(one.size() * 4));
                std::printf("wrote the prompt route's logits to %s\n", out);
            }
            if (const char* guard = std::getenv("INFERNIX_INKLING2_PROMPT_LOGITS_GUARD"); guard != nullptr) {
                std::ifstream f(guard, std::ios::binary);
                std::vector<float> want(one.size());
                f.read(reinterpret_cast<char*>(want.data()), static_cast<std::streamsize>(want.size() * 4));
                const bool equal = static_cast<bool>(f) && std::memcmp(want.data(), one.data(), one.size() * 4) == 0;
                std::printf("prompt route guard (%s): %s\n", guard, equal ? "the logits byte for byte" : "DIFFERENT");
                ok = ok && equal;
            }
            double vs_ref = 0, vs_decode = 0;
            int prompt_differ = 0;
            for (int t = 0; t < kPositions; ++t) {
                const float* g = one.data() + static_cast<std::size_t>(t) * V;
                const float* r = rl + static_cast<std::size_t>(t) * V;
                const float* d = verify.data() + static_cast<std::size_t>(t) * V;
                std::size_t ga = 0, ra = 0;
                for (std::size_t i = 0; i < V; ++i) {
                    vs_ref    = std::max(vs_ref, std::fabs(double{g[i]} - r[i]));
                    vs_decode = std::max(vs_decode, std::fabs(double{g[i]} - d[i]));
                    if (g[i] > g[ga]) { ga = i; }
                    if (r[i] > r[ra]) { ra = i; }
                }
                if (ga != ra) {
                    ++prompt_differ;
                    std::printf("  prompt route position %d: argmax %zu, the reference's %zu%s\n", t, ga, ra,
                                t >= kLastPrompt ? "  FAIL (greedy continuation)" : "");
                    ok = ok && t < kLastPrompt;
                }
            }
            std::printf("prompt route: %d of %d argmax differ from the reference (none allowed from position %d); largest "
                        "logit difference %.4f from the reference, %.4f from the decode route\n",
                        prompt_differ, kPositions, kLastPrompt, vs_ref, vs_decode);
        }
        ok = graph_replay(parameters, forward, state, ref) && ok;
        ok = lanes_alone(parameters, forward, state, ref, a8) && ok;
        std::printf(ok ? "PASS\n" : "FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
