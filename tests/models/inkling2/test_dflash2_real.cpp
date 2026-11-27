// Inkling 2's DFlash2 drafter (src/models/inkling2/execution/dflash2.h, --spec dflash2 --draft-model) on the real
// artifact and companion.
//
// 1. The drafter against an FP64 CPU oracle of its mathematics (z-lab's DFlash2DraftModel with incoai's Inkling
//    deviations, read from SGLang's Inkling DFlash2 integration: the block's inputs are the target's normed embeddings,
//    the selector's softcap comes after its correction, the candidates come from the real vocabulary), over the same
//    Q8 and BF16 weights the drafter reads (codes times scales, exact), two lanes of real context features (the first
//    700 and 2,300 tokens of two of the repository's documents, prefilled by an Engine, their features read from its
//    DFlash2 context calls: the second's ring wrapped) and the block at each context's end, anchored at the target's own next token. (On synthetic features the
//    oracle's own rounded and exact forms disagreed on half the argmaxes: the drafter is chaotic there, nothing judged.) Per draft position, along the
//    drafter's own path (each position's predecessor is the drafter's draft before it): judged, the oracle's selector
//    argmax equals the draft wherever the oracle's two best edges differ by more than kMargin, and the drafter's
//    confidence (its temperature-1 probability at the drafted rank) is within kConfidence of the oracle's. Reported:
//    how many positions were judged, the largest confidence difference. With the text head and with --lm-head-draft's
//    shortlist (the oracle reads the shortlist's own 8-bit rows).
// 2. Through the Engine: speculative tokens equal plain decoding's (verification keeps a draft
//    only when it equals the target's own draw), greedy, top-20 sampled (the default) and sampled over the whole
//    vocabulary (whose drafts are coupled with the draws), one request and rounds of several rows drafting together (C = 2, 4, 8), with and without --lm-head-draft,
//    graphs and eager, and a request resumed from the prefix cache (its rings restored). Reported: the acceptance, and
//    the sampled acceptance (top 20, whole vocabulary) at a fixed 7 drafts by position. Then the acceptance gates (acceptance_gates, each
//    failure counted: token equality cannot see the drafter): a request alone and beside three others draft alike, rings
//    the commits wrote serve as well as a prefill's, and DFlash2 accepts more than DSpark on a code and a maths prompt.
//
//   INFERNIX_INKLING2_ARTIFACT  an artifact converted with the inkling2_mini recipe
//   INFERNIX_INKLING2_DFLASH2   its DFlash2 companion (recipe inkling2_mini_dflash2)

#include "artifact/reader.h"
#include "core/device.h"
#include "infernix/engine.h"
#include "infernix/ops/inkling2_visit.h"
#include "models/inkling2/execution/dflash2.h"
#include "models/inkling2/execution/forward.h"
#include "models/inkling2/execution/parameters.h"
#include "models/inkling2/load.h"
#include "models/inkling2/program/test_taps.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace m  = infernix::models::inkling2;
namespace ik = infernix::ops::inkling2;
using infernix::DeviceContext;
using infernix::Tensor;
using infernix::Weight;

// The tolerances are the oracle's own sensitivity to its BF16 rounding points (the largest change of an edge and of a
// confidence between the oracle with and without them, over every position whose 16 candidates it keeps), doubled,
// with these floors: fixed before any drafter output is judged.
constexpr double kMarginFloor     = 0.02;
constexpr double kConfidenceFloor = 0.01;
constexpr int kH = 2304, kTaps = 4, kF = kTaps * kH, kD = 128, kHq = 32, kHk = 8, kI = 6912, kRank = 256;
constexpr int kW = 8, kK = 7, kC = 16, kWindow = 2048;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}
void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

double bf(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
std::uint16_t to_bf(double x) {
    const float f = static_cast<float>(x);
    std::uint32_t bits;
    std::memcpy(&bits, &f, 4);
    bits += 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}
// The oracle's BF16 rounding points (the drafter's stored intermediates); off, the oracle computes them exactly, which
// measures its own sensitivity to them.
bool g_round = true;
double rbf(double x) { return g_round ? bf(to_bf(x)) : x; }
double half(std::uint16_t h) {
    const std::uint32_t sign = (h >> 15) & 1U, exponent = (h >> 10) & 31U, mantissa = h & 1023U;
    double v = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -24)
                             : std::ldexp(1.0 + mantissa / 1024.0, static_cast<int>(exponent) - 15);
    return sign ? -v : v;
}

template <class T>
std::vector<T> host(const void* device, std::size_t count) {
    std::vector<T> out(count);
    cuda(cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return out;
}

void parallel(int n, const std::function<void(int)>& body) {
    const int threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            for (int i = t; i < n; i += threads) { body(i); }
        });
    }
    for (auto& thread : pool) { thread.join(); }
}

// A projection's rows on the host: Q8_G32_FP16 RowSplit (codes times their group's FP16 scale) or BF16.
struct Matrix {
    int rows = 0, k = 0, stride = 0;
    bool q8 = false;
    std::vector<std::int8_t> codes;
    std::vector<std::uint16_t> scales, words;
    explicit Matrix(const Weight& w) : rows(w.n), k(w.k) {
        q8 = w.qtype == infernix::QType::Q8_G32_FP16;
        if (q8) {
            stride = static_cast<int>(w.padded_shape[1]);
            codes  = host<std::int8_t>(w.qdata, static_cast<std::size_t>(rows) * stride);
            scales = host<std::uint16_t>(w.scales, static_cast<std::size_t>(rows) * (stride / 32));
        } else {
            require(w.qtype == infernix::QType::BF16, "a BF16 or Q8 matrix");
            words = host<std::uint16_t>(w.qdata, static_cast<std::size_t>(rows) * k);
        }
    }
    // y[r] = sum_k W[r][k] x[k] for rows [first, first + count).
    void apply(const std::vector<double>& x, std::vector<double>& y, int first = 0, int count = -1) const {
        if (count < 0) { count = rows; }
        y.assign(static_cast<std::size_t>(count), 0.0);
        for (int r = 0; r < count; ++r) {
            const std::size_t row = static_cast<std::size_t>(first + r);
            double sum = 0;
            if (q8) {
                for (int g = 0; g < k / 32; ++g) {
                    double group = 0;
                    for (int j = 0; j < 32; ++j) { group += codes[row * stride + g * 32 + j] * x[g * 32 + j]; }
                    sum += group * half(scales[row * (stride / 32) + g]);
                }
            } else {
                for (int j = 0; j < k; ++j) { sum += bf(words[row * k + j]) * x[j]; }
            }
            y[static_cast<std::size_t>(r)] = sum;
        }
    }
};

std::vector<double> vec(const Tensor& t) {
    std::size_t n = 1;
    for (int i = 0; i < 4; ++i) { n *= static_cast<std::size_t>(std::max(1, t.ne[i])); }
    std::vector<double> out(n);
    const auto words = host<std::uint16_t>(t.data, n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = bf(words[i]); }
    return out;
}

// plain RMSNorm over x (eps 1e-6) times w, BF16-rounded as the drafter stores it.
std::vector<double> rmsnorm(const std::vector<double>& x, const std::vector<double>& w, std::size_t at = 0,
                            std::size_t n = 0) {
    if (n == 0) { n = x.size(); }
    double s = 0;
    for (std::size_t i = 0; i < n; ++i) { s += x[at + i] * x[at + i]; }
    const double inv = 1.0 / std::sqrt(s / static_cast<double>(n) + 1e-6);
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = rbf(x[at + i] * inv * w[i]); }
    return out;
}

// Full-head split-half RoPE (theta) at `position` over each head of 128 in v (heads * 128).
void rope(std::vector<double>& v, int heads, double position, double theta) {
    for (int h = 0; h < heads; ++h) {
        double* x = v.data() + static_cast<std::size_t>(h) * kD;
        for (int j = 0; j < kD / 2; ++j) {
            const double angle = position * std::pow(theta, -2.0 * j / kD);
            const double a = x[j], b = x[j + kD / 2];
            x[j]          = a * std::cos(angle) - b * std::sin(angle);
            x[j + kD / 2] = b * std::cos(angle) + a * std::sin(angle);
        }
    }
}

// Head-wise RMSNorm times w (128), then RoPE.
std::vector<double> head_norm_rope(const std::vector<double>& x, int heads, const std::vector<double>& w, double position,
                                   double theta) {
    std::vector<double> out(x.size());
    for (int h = 0; h < heads; ++h) {
        const auto n = rmsnorm(x, w, static_cast<std::size_t>(h) * kD, kD);
        std::copy(n.begin(), n.end(), out.begin() + static_cast<std::ptrdiff_t>(h) * kD);
    }
    rope(out, heads, position, theta);
    return out;
}

struct LayerHost {
    std::vector<double> input_norm, post_norm, q_norm, k_norm, attention_base, mlp_base;
    Matrix qkv, ck, cv, o, gate_up, down, attention_delta, mlp_delta;
    explicit LayerHost(const m::DFlash2LayerOperands& l)
        : input_norm(vec(l.input_norm)), post_norm(vec(l.post_attention_norm)), q_norm(vec(l.query_norm)),
          k_norm(vec(l.key_norm)), attention_base(vec(l.attention_base)), mlp_base(vec(l.mlp_base)),
          qkv(l.query_key_value), ck(l.context_key), cv(l.context_value), o(l.output), gate_up(l.gate_up), down(l.down),
          attention_delta(l.attention_projection), mlp_delta(l.mlp_projection) {}
};

// The dynamic grouped conv over a block's columns x[i][h] with side `side`'s base [2 sides][2 taps][H] and the
// columns' deltas [i][2 sides][2 taps][G].
std::vector<std::vector<double>> conv(const std::vector<std::vector<double>>& x, const std::vector<double>& base,
                                      const std::vector<std::vector<double>>& delta, int side) {
    constexpr int G = kH / 16;
    std::vector<std::vector<double>> out(x.size(), std::vector<double>(kH));
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (int h = 0; h < kH; ++h) {
            const int g = h / 16;
            double v    = (base[(side * 2 + 0) * kH + h] + delta[i][(side * 2 + 0) * G + g]) * x[i][h];
            if (i > 0) { v += (base[(side * 2 + 1) * kH + h] + delta[i][(side * 2 + 1) * G + g]) * x[i - 1][h]; }
            out[i][h] = v;
        }
    }
    return out;
}

struct OracleLane {
    std::vector<std::vector<double>> context_k[5], context_v[5]; // [position from F - window + 1][kv]
    int first = 0;                                              // the earliest stored position
};

// A real prompt's context: its target features at every position (BF16 [T][4 H], read from its prefill's DFlash2 context
// calls through testing::set_context_observer), its length T and the token the target drew after it (the block's anchor).
struct RealContext {
    int length          = 0;
    std::int32_t anchor = 0;
    std::vector<std::uint16_t> features;
};

int check_drafter(DeviceContext& device, const std::filesystem::path& artifact, const std::filesystem::path& companion,
                  bool shortlist, const std::vector<RealContext>& contexts) {
    const std::string label = shortlist ? "--lm-head-draft" : "text head";
    infernix::artifact::Reader reader(artifact);
    infernix::models::LoadOptions load;
    load.speculative   = infernix::SpeculativeBackend::DFlash2;
    load.draft_model   = companion;
    load.proposal_head = shortlist ? infernix::ProposalHead::Optimized : infernix::ProposalHead::Full;
    auto model         = m::materialize_model(m::plan_load(reader, load), device);
    const m::Parameters p(*model);
    const m::DFlash2Config& cfg = *model->config().dflash2;
    require(cfg.softcap_after_correction && cfg.softcap == 30.0F, "the companion's selector softcaps after its correction");
    m::Forward forward(p, 2, m::kMaxForwardColumns);
    constexpr int kLanes = 2;
    require(contexts.size() == kLanes, "two real contexts");
    m::DFlash2Drafter drafter(p, forward, kLanes, 4096);
    int lengths[kLanes], anchors[kLanes];
    std::vector<std::vector<std::uint16_t>> features(kLanes);
    for (int lane = 0; lane < kLanes; ++lane) {
        lengths[lane]  = contexts[static_cast<std::size_t>(lane)].length;
        anchors[lane]  = contexts[static_cast<std::size_t>(lane)].anchor;
        features[lane] = contexts[static_cast<std::size_t>(lane)].features;
    }

    // The real prompts' context features into the drafter's rings (lane 1's wrap them), the blocks at their ends.
    for (int lane = 0; lane < kLanes; ++lane) {
        const int T = lengths[lane];
        std::vector<std::int32_t> positions(static_cast<std::size_t>(T)), lanes(static_cast<std::size_t>(T), lane);
        for (int t = 0; t < T; ++t) { positions[static_cast<std::size_t>(t)] = t; }
        std::uint16_t* d_features = nullptr;
        std::int32_t *d_positions = nullptr, *d_lanes = nullptr;
        cuda(cudaMalloc(&d_features, features[lane].size() * 2), "features");
        cuda(cudaMalloc(&d_positions, positions.size() * 4), "positions");
        cuda(cudaMalloc(&d_lanes, lanes.size() * 4), "lanes");
        cuda(cudaMemcpy(d_features, features[lane].data(), features[lane].size() * 2, cudaMemcpyHostToDevice), "up");
        cuda(cudaMemcpy(d_positions, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice), "up");
        cuda(cudaMemcpy(d_lanes, lanes.data(), lanes.size() * 4, cudaMemcpyHostToDevice), "up");
        drafter.context(d_features, d_positions, d_lanes, T, device.stream);
        cuda(cudaStreamSynchronize(device.stream), "context");
        cudaFree(d_features), cudaFree(d_positions), cudaFree(d_lanes);
    }
    std::vector<ik::DSparkRound> rounds;
    for (int lane = 0; lane < kLanes; ++lane) { rounds.push_back({lengths[lane], anchors[lane], lane}); }
    drafter.stage(rounds);
    drafter.draft(device.stream, kLanes);
    drafter.download(device.stream, kLanes);
    cuda(cudaStreamSynchronize(device.stream), "draft");

    // The oracle's weights and inputs.
    const m::DFlash2Operands& d = *p.dflash2;
    const Matrix fc(d.feature_projection), selector(d.selector_hidden);
    const std::vector<double> context_norm = vec(d.context_norm), final_norm = vec(d.final_norm);
    std::vector<LayerHost> layers;
    for (const auto& l : d.layers) { layers.emplace_back(l); }
    const auto& shortlisted = model->draft_head();
    const Matrix head(shortlisted ? shortlisted->head : p.head);
    std::vector<std::int32_t> row_ids;
    if (shortlisted) { row_ids = host<std::int32_t>(shortlisted->token_ids, static_cast<std::size_t>(shortlisted->rows)); }
    const int V = static_cast<int>(model->config().text.vocab_size);
    const auto predecessor = host<std::uint16_t>(d.predecessor_codebook.data, static_cast<std::size_t>(V) * kRank);
    const auto successor   = host<std::uint16_t>(d.successor_codebook.data, static_cast<std::size_t>(V) * kRank);
    // The block's inputs: the target's normed embeddings (its own Op) of each anchor and the mask token.
    std::vector<std::vector<double>> embedded(kLanes * kW);
    {
        std::vector<std::int32_t> ids;
        for (int lane = 0; lane < kLanes; ++lane) {
            ids.push_back(anchors[lane]);
            for (int i = 1; i < kW; ++i) { ids.push_back(static_cast<std::int32_t>(cfg.mask_token_id)); }
        }
        std::int32_t* d_ids    = nullptr;
        std::uint16_t* d_rows  = nullptr;
        cuda(cudaMalloc(&d_ids, ids.size() * 4), "ids");
        cuda(cudaMalloc(&d_rows, ids.size() * kH * 2), "rows");
        cuda(cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "up");
        ik::embed(d_ids, static_cast<std::int32_t>(ids.size()), p.embedding, p.embedding_norm, d_rows, device.stream);
        cuda(cudaStreamSynchronize(device.stream), "embed");
        const auto rows = host<std::uint16_t>(d_rows, ids.size() * kH);
        for (std::size_t c = 0; c < ids.size(); ++c) {
            embedded[c].resize(kH);
            for (int h = 0; h < kH; ++h) { embedded[c][h] = bf(rows[c * kH + h]); }
        }
        cudaFree(d_ids), cudaFree(d_rows);
    }
    const double theta = cfg.rope_theta;
    // A draft position's 16 candidates (token ids, the head's rank order) and their edges, along the drafter's own path.
    struct Step {
        std::vector<int> tokens;
        std::vector<double> edges;
    };
    const auto oracle = [&](int lane) {
        std::vector<Step> steps;
        const int F = lengths[lane], from = std::max(0, F - (kWindow - 1));
        // Context K/V of positions [F - 2047, F): fc, norm, each layer's K (head norm, RoPE) and V.
        const int n = F - from;
        std::vector<std::vector<std::vector<double>>> ck(5, std::vector<std::vector<double>>(n)), cv = ck;
        parallel(n, [&](int t) {
            const int position = from + t;
            std::vector<double> f(kF), projected;
            for (int i = 0; i < kF; ++i) { f[i] = bf(features[lane][static_cast<std::size_t>(position) * kF + i]); }
            fc.apply(f, projected);
            for (double& v : projected) { v = rbf(v); }
            const auto c = rmsnorm(projected, context_norm);
            for (int l = 0; l < 5; ++l) {
                std::vector<double> k, v;
                layers[l].ck.apply(c, k);
                layers[l].cv.apply(c, v);
                ck[l][t] = head_norm_rope(k, kHk, layers[l].k_norm, position, theta);
                for (double& x : ck[l][t]) { x = rbf(x); }
                for (double& x : v) { x = rbf(x); }
                cv[l][t] = v;
            }
        });
        // The block through the layers.
        std::vector<std::vector<double>> x(kW);
        for (int i = 0; i < kW; ++i) { x[i] = embedded[static_cast<std::size_t>(lane * kW + i)]; }
        for (int l = 0; l < 5; ++l) {
            const LayerHost& L = layers[l];
            const auto branch = [&](const std::vector<double>& norm, const std::vector<double>& base, const Matrix& delta_w,
                                    std::vector<std::vector<double>>& prepared, std::vector<std::vector<double>>& delta) {
                std::vector<std::vector<double>> normed(kW);
                delta.assign(kW, {});
                for (int i = 0; i < kW; ++i) {
                    normed[i] = rmsnorm(x[i], norm);
                    delta_w.apply(normed[i], delta[i]);
                }
                prepared = conv(normed, base, delta, 0);
                for (auto& col : prepared) {
                    for (double& v : col) { v = rbf(v); }
                }
                for (auto& col : delta) {
                    for (double& v : col) { v = rbf(v); }
                }
            };
            std::vector<std::vector<double>> prepared, delta;
            branch(L.input_norm, L.attention_base, L.attention_delta, prepared, delta);
            std::vector<std::vector<double>> q(kW), k(kW), v(kW);
            for (int i = 0; i < kW; ++i) {
                std::vector<double> all;
                L.qkv.apply(prepared[i], all);
                for (double& a : all) { a = rbf(a); }
                std::vector<double> qi(all.begin(), all.begin() + kHq * kD), ki(all.begin() + kHq * kD,
                                                                                    all.begin() + (kHq + kHk) * kD);
                v[i].assign(all.begin() + (kHq + kHk) * kD, all.end());
                q[i] = head_norm_rope(qi, kHq, L.q_norm, F + i, theta);
                k[i] = head_norm_rope(ki, kHk, L.k_norm, F + i, theta);
                for (double& a : q[i]) { a = rbf(a); }
                for (double& a : k[i]) { a = rbf(a); }
            }
            std::vector<std::vector<double>> attention(kW, std::vector<double>(kHq * kD));
            parallel(kW * kHq, [&](int item) {
                const int i = item / kHq, h = item % kHq, g = h / (kHq / kHk);
                const int query = F + i;
                std::vector<double> scores;
                std::vector<const double*> values;
                const double scale = 1.0 / std::sqrt(static_cast<double>(kD));
                for (int t = 0; t < n; ++t) {
                    if (query - (from + t) >= kWindow) { continue; }
                    double s = 0;
                    for (int e = 0; e < kD; ++e) { s += q[i][h * kD + e] * ck[l][t][g * kD + e]; }
                    scores.push_back(s * scale);
                    values.push_back(cv[l][t].data() + g * kD);
                }
                for (int j = 0; j < kW; ++j) {
                    double s = 0;
                    for (int e = 0; e < kD; ++e) { s += q[i][h * kD + e] * k[j][g * kD + e]; }
                    scores.push_back(s * scale);
                    values.push_back(v[j].data() + g * kD);
                }
                const double top = *std::max_element(scores.begin(), scores.end());
                double sum = 0;
                for (double& s : scores) { s = std::exp(s - top), sum += s; }
                for (int e = 0; e < kD; ++e) {
                    double acc = 0;
                    for (std::size_t j = 0; j < scores.size(); ++j) { acc += scores[j] * values[j][e]; }
                    attention[i][h * kD + e] = rbf(acc / sum);
                }
            });
            std::vector<std::vector<double>> out(kW);
            for (int i = 0; i < kW; ++i) { L.o.apply(attention[i], out[i]); }
            const auto finished = conv(out, L.attention_base, delta, 1);
            for (int i = 0; i < kW; ++i) {
                for (int h = 0; h < kH; ++h) { x[i][h] = rbf(x[i][h] + finished[i][h]); }
            }
            branch(L.post_norm, L.mlp_base, L.mlp_delta, prepared, delta);
            std::vector<std::vector<double>> mlp(kW);
            parallel(kW, [&](int i) {
                std::vector<double> gu, product(kI);
                L.gate_up.apply(prepared[i], gu);
                for (int j = 0; j < kI; ++j) {
                    const double g = gu[j], u = gu[kI + j];
                    product[j] = rbf(g / (1.0 + std::exp(-g)) * u);
                }
                L.down.apply(product, mlp[i]);
            });
            const auto mlp_finished = conv(mlp, L.mlp_base, delta, 1);
            for (int i = 0; i < kW; ++i) {
                for (int h = 0; h < kH; ++h) { x[i][h] = rbf(x[i][h] + mlp_finished[i][h]); }
            }
        }
        // Each mask column's head candidates and the selector's edges along the drafter's path.
        const std::int32_t* drafts = drafter.host_tokens(lane);
        std::int32_t previous      = anchors[lane];
        for (int i = 0; i < kK; ++i) {
            const auto hidden = rmsnorm(x[i + 1], final_norm);
            // An 8-bit head reads 8-bit inputs (Forward::project_head's x8: per 32 values d = m / 127 and codes
            // rne(v 127 / m) in FP32, ops/ggml/ggml_a8.h), as the drafter passes them.
            std::vector<double> head_input = hidden;
            if (head.q8) {
                for (int s = 0; s < kH / 32; ++s) {
                    float largest = 0;
                    for (int j = 0; j < 32; ++j) { largest = std::max(largest, std::abs(static_cast<float>(hidden[s * 32 + j]))); }
                    const float d = largest / 127.0F, inv = largest > 0 ? 127.0F / largest : 0.0F;
                    for (int j = 0; j < 32; ++j) {
                        const float q = std::nearbyint(static_cast<float>(hidden[s * 32 + j]) * inv);
                        head_input[s * 32 + j] = static_cast<double>(std::clamp(q, -127.0F, 127.0F)) * d;
                    }
                }
            }
            std::vector<double> logits(static_cast<std::size_t>(head.rows));
            parallel(64, [&](int part) {
                const int rows = (head.rows + 63) / 64, first = part * rows, count = std::min(rows, head.rows - first);
                if (count <= 0) { return; }
                std::vector<double> y;
                head.apply(head_input, y, first, count);
                std::copy(y.begin(), y.end(), logits.begin() + first);
            });
            std::vector<int> order(logits.size());
            for (std::size_t r = 0; r < order.size(); ++r) { order[r] = static_cast<int>(r); }
            std::partial_sort(order.begin(), order.begin() + kC, order.end(), [&](int a, int b) {
                return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
            });
            std::vector<double> g;
            selector.apply(hidden, g);
            for (double& v : g) { v = rbf(v); }
            Step step;
            step.edges.resize(kC);
            step.tokens.resize(kC);
            for (int c = 0; c < kC; ++c) {
                const int row   = order[c];
                step.tokens[c]  = shortlisted ? row_ids[static_cast<std::size_t>(row)] : row;
                double e        = logits[row];
                for (int r = 0; r < kRank; ++r) {
                    e += bf(predecessor[static_cast<std::size_t>(previous) * kRank + r]) * g[r] *
                         bf(successor[static_cast<std::size_t>(step.tokens[c]) * kRank + r]);
                }
                step.edges[c] = cfg.softcap * std::tanh(e / cfg.softcap);
            }
            steps.push_back(std::move(step));
            previous = drafts[i];
        }
        return steps;
    };
    // A step's argmax rank, its margin over the next edge, and the temperature-1 probability of `token` (0 outside).
    struct Reading {
        int best = 0;
        double margin = 0, confidence = 0;
    };
    const auto read = [](const Step& s, int token) {
        Reading r;
        for (int c = 1; c < kC; ++c) { r.best = s.edges[c] > s.edges[r.best] ? c : r.best; }
        double second = -1e300, sum = 0;
        for (int c = 0; c < kC; ++c) {
            if (c != r.best) { second = std::max(second, s.edges[c]); }
            sum += std::exp(s.edges[c] - s.edges[r.best]);
        }
        r.margin = s.edges[r.best] - second;
        for (int c = 0; c < kC; ++c) {
            if (s.tokens[c] == token) { r.confidence = std::exp(s.edges[c] - s.edges[r.best]) / sum; }
        }
        return r;
    };
    std::vector<std::vector<Step>> rounded(kLanes), exact(kLanes);
    for (int lane = 0; lane < kLanes; ++lane) {
        g_round       = true;
        rounded[lane] = oracle(lane);
        g_round       = false;
        exact[lane]   = oracle(lane);
        g_round       = true;
    }
    // The tolerances, from the oracle alone: over the candidates both oracles keep (a near-tie in the head's lower ranks
    // reorders or swaps them), the largest edge change, and the drafted token's confidence change. A position whose
    // argmax token the roundings change is not judged.
    double edge_sensitivity = 0, confidence_sensitivity = 0;
    int unstable = 0, shared = 0;
    std::vector<std::vector<bool>> stable(kLanes, std::vector<bool>(kK, false));
    for (int lane = 0; lane < kLanes; ++lane) {
        for (int i = 0; i < kK; ++i) {
            const Step &a = rounded[lane][i], &b = exact[lane][i];
            const int token = drafter.host_tokens(lane)[i];
            const Reading ra = read(a, token), rb = read(b, token);
            for (int c = 0; c < kC; ++c) {
                const auto at = std::find(b.tokens.begin(), b.tokens.end(), a.tokens[c]);
                if (at == b.tokens.end()) { continue; }
                ++shared;
                edge_sensitivity = std::max(edge_sensitivity, std::abs(a.edges[c] - b.edges[at - b.tokens.begin()]));
            }
            if (a.tokens[ra.best] != b.tokens[rb.best]) {
                ++unstable;
                continue;
            }
            stable[lane][i]        = true;
            confidence_sensitivity = std::max(confidence_sensitivity, std::abs(ra.confidence - rb.confidence));
        }
    }
    std::printf("%s: the two oracles share %.1f of 16 candidates a position\n", label.c_str(),
                static_cast<double>(shared) / (kLanes * kK));
    const double margin_tolerance     = std::max(kMarginFloor, 2.0 * edge_sensitivity);
    const double confidence_tolerance = std::max(kConfidenceFloor, 2.0 * confidence_sensitivity);
    std::printf("%s: the oracle's rounding sensitivity: edges %.4f, confidence %.4f (%d of %d positions' argmax "
                "changes); judged margin > %.4f, confidence within %.4f\n",
                label.c_str(), edge_sensitivity, confidence_sensitivity, unstable, kLanes * kK, margin_tolerance,
                confidence_tolerance);
    int judged = 0, agree = 0, failures = 0;
    double worst_confidence = 0;
    for (int lane = 0; lane < kLanes; ++lane) {
        const std::int32_t* drafts = drafter.host_tokens(lane);
        const float* confidence    = drafter.host_probabilities(lane);
        for (int i = 0; i < kK; ++i) {
            if (!stable[lane][i]) { continue; }
            const Reading r     = read(rounded[lane][i], drafts[i]);
            const double diff   = std::abs(r.confidence - static_cast<double>(confidence[i]));
            worst_confidence    = std::max(worst_confidence, diff);
            if (diff > confidence_tolerance) {
                std::printf("FAIL %s lane %d draft %d: confidence %.4f, the oracle's %.4f\n", label.c_str(), lane, i,
                            confidence[i], r.confidence);
                ++failures;
            }
            if (r.margin > margin_tolerance) {
                ++judged;
                if (rounded[lane][i].tokens[r.best] == drafts[i]) {
                    ++agree;
                } else {
                    std::printf("FAIL %s lane %d draft %d: drafted %d, the oracle's argmax %d (margin %.3f)\n",
                                label.c_str(), lane, i, drafts[i], rounded[lane][i].tokens[r.best], r.margin);
                    ++failures;
                }
            }
        }
        std::printf("%s lane %d (context %d): drafts", label.c_str(), lane, lengths[lane]);
        for (int i = 0; i < kK; ++i) { std::printf(" %d (%.3f)", drafts[i], confidence[i]); }
        std::printf("\n");
    }
    std::printf("%s: %d of %d judged drafts equal the oracle's argmax; worst confidence difference %.4f\n", label.c_str(),
                agree, judged, worst_confidence);
    if (judged == 0) { // counted, so the later checks still run
        std::printf("FAIL %s: no draft position was judged (every margin within the tolerance)\n", label.c_str());
        ++failures;
    }
    return failures;
}

// ---- 2. the Engine
using Tokens = std::vector<infernix::TokenId>;

infernix::EngineOptions engine_options(const std::filesystem::path& artifact, std::uint32_t concurrency) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(concurrency * 4096);
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 16;
    options.context_cache.host_capacity_bytes = std::size_t{2} << 30U;
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level != infernix::DiagnosticLevel::Debug) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

infernix::RequestOptions sampled(std::size_t i, std::uint32_t tokens = 96) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 1.0F;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.top_p          = 1.0F;
    options.execution.sampling.seed           = 2000 + i;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

// Temperature 1 over the whole vocabulary (top_k 0, as the checkpoint's generation config asks; the engine's default
// is sampled(): the top 20).
infernix::RequestOptions whole(std::size_t i, std::uint32_t tokens = 96) {
    infernix::RequestOptions options = sampled(i, tokens);
    options.execution.sampling.top_k = 0;
    options.execution.sampling.seed  = 3000 + i;
    return options;
}

infernix::PromptInput chat(const std::string& text) {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    user.parts.push_back({.text = text});
    input.messages.push_back(std::move(user));
    input.options.reasoning_effort = infernix::ReasoningEffort::Low;
    return input;
}

const std::vector<std::string>& texts() {
    static const std::vector<std::string> t = {
        "Write a Python function that returns the n-th Fibonacci number, with a docstring and tests.",
        "Explain how a compiler turns source code into machine instructions.",
        "Solve step by step: a train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive?",
        "Describe the water cycle to a ten-year-old.",
        "List twelve uses for a paperclip.",
        "Write a haiku about autumn, then explain its imagery.",
        "Summarise the causes of the French Revolution in five bullet points.",
        "Translate 'The library opens at nine and closes at five' into French, German and Spanish.",
    };
    return t;
}

void require_same(const std::string& label, const Tokens& a, const Tokens& b) {
    std::size_t at = 0;
    while (at < a.size() && at < b.size() && a[at] == b[at]) { ++at; }
    require(!a.empty() && a == b, label + ": tokens differ at " + std::to_string(at) + " of " + std::to_string(a.size()) +
                                      "/" + std::to_string(b.size()));
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

void report(const std::string& label, const infernix::SpeculativeStats& s) {
    std::cout << label << ": " << s.rounds << " rounds, " << s.accepted_tokens << " of " << s.drafted_tokens
              << " drafts accepted (" << (s.rounds == 0 ? 0.0 : 1.0 + static_cast<double>(s.accepted_tokens) / s.rounds)
              << " tokens a round), " << s.fallback_steps << " plain steps\n";
}

double per_round(const infernix::SpeculativeStats& s) {
    return s.rounds == 0 ? 0.0 : static_cast<double>(s.accepted_tokens) / static_cast<double>(s.rounds);
}

infernix::EngineOptions drafting(const std::filesystem::path& artifact, const std::filesystem::path& companion,
                                 std::uint32_t concurrency, bool dflash2) {
    // A fixed length of 7: every round verifies the whole block, so the acceptance measures the drafts alone.
    infernix::EngineOptions options       = engine_options(artifact, concurrency);
    options.speculative.backend           = dflash2 ? infernix::SpeculativeBackend::DFlash2 : infernix::SpeculativeBackend::DSpark;
    options.speculative.draft_model       = dflash2 ? companion : std::filesystem::path{};
    options.speculative.draft_tokens_auto = false;
    options.speculative.draft_tokens      = 7;
    return options;
}

// Token equality cannot see the drafter (a draft never changes the output), so these gates read its acceptance (greedy,
// 7 drafts a round), each rule fixed before the first run:
// a. A request alone and beside three others drafts alike: its acceptance (accepted / drafted) within 0.05 (the rows'
//    drafts in one call, their commits gathered from uneven shares).
// b. Rings the commits wrote serve as well as rings a prefill wrote: a 320-token reply's accepted drafts a round are at
//    least 0.75 of the mean first round's of eight requests whose prompts hold that reply's first 40, 80, ... 320 tokens.
// c. DFlash2 accepts more drafts a round than DSpark on a code and a maths prompt (the card claims 4.8-5.0 tokens a
//    round against DSpark's 3.5 at temperature 1 with rejection sampling; greedy here).
int acceptance_gates(const std::filesystem::path& artifact, const std::filesystem::path& companion) {
    int failures     = 0;
    const auto gate  = [&](bool passed, const std::string& message) {
        if (!passed) {
            std::cout << "FAIL " << message << '\n';
            ++failures;
        }
    };
    {
        infernix::Engine engine(drafting(artifact, companion, 4, true));
        const auto alone = engine.generate(engine.prepare(chat(texts()[0])), greedy(256));
        std::vector<infernix::GenerationHandle> handles;
        for (std::size_t i = 0; i < 4; ++i) { handles.push_back(engine.submit(engine.prepare(chat(texts()[i])), greedy(256))); }
        std::vector<infernix::GenerationResult> results;
        for (auto& handle : handles) { results.push_back(handle.wait()); }
        require_same("gate a: request 0 beside three others", alone.generated_token_ids, results[0].generated_token_ids);
        const auto rate = [](const infernix::SpeculativeStats& s) {
            return s.drafted_tokens == 0 ? 0.0 : static_cast<double>(s.accepted_tokens) / s.drafted_tokens;
        };
        std::printf("gate a: acceptance alone %.3f (%llu rounds), beside three others %.3f (%llu rounds)\n",
                    rate(alone.speculative), static_cast<unsigned long long>(alone.speculative.rounds),
                    rate(results[0].speculative), static_cast<unsigned long long>(results[0].speculative.rounds));
        gate(std::abs(rate(alone.speculative) - rate(results[0].speculative)) <= 0.05,
             "gate a: a request drafts differently beside other rows");
    }
    {
        infernix::Engine engine(drafting(artifact, companion, 1, true));
        const Tokens prompt = engine.tokenize_text(
            "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n");
        const auto reply = engine.generate(engine.prepare_tokens(prompt), greedy(320));
        const Tokens& g  = reply.generated_token_ids;
        require(g.size() == 320, "gate b: the reply is 320 tokens");
        double first_rounds = 0;
        int samples         = 0;
        for (std::size_t t = 40; t <= 320; t += 40) {
            Tokens extended = prompt;
            extended.insert(extended.end(), g.begin(), g.begin() + static_cast<std::ptrdiff_t>(t));
            const auto next = engine.generate(engine.prepare_tokens(extended), greedy(8));
            first_rounds += per_round(next.speculative);
            ++samples;
        }
        const double prefilled = first_rounds / samples, committed = per_round(reply.speculative);
        std::printf("gate b: %.2f drafts accepted a round over a reply (commit-written rings), %.2f in first rounds "
                    "after a prefill\n", committed, prefilled);
        gate(committed >= 0.75 * prefilled, "gate b: the commits' rings serve worse than a prefill's");
    }
    for (const std::size_t text : {std::size_t{0}, std::size_t{2}}) {
        double rates[2] = {};
        for (const bool dflash2 : {false, true}) {
            infernix::Engine engine(drafting(artifact, companion, 1, dflash2));
            const auto result = engine.generate(engine.prepare(chat(texts()[text])), greedy(320));
            rates[dflash2 ? 1 : 0] = per_round(result.speculative);
            std::printf("gate c, prompt %zu: %s accepts %.2f drafts a round; by position", text,
                        dflash2 ? "DFlash2" : "DSpark", rates[dflash2 ? 1 : 0]);
            for (const auto n : result.speculative.accepted_per_position) {
                std::printf(" %llu", static_cast<unsigned long long>(n));
            }
            std::printf("\n");
        }
        gate(rates[1] > rates[0], "gate c, prompt " + std::to_string(text) + ": DFlash2 accepts no more drafts a round than DSpark");
    }
    return failures;
}

// Reported, not judged: the sampled acceptance at a fixed 7 drafts a round (every draft verified, so the length policy
// plays no part), by position, for the default top 20 (greedy drafts) and the whole vocabulary (coupled drafts) at
// temperature 1.
void fixed_length_sampled(const std::filesystem::path& artifact, const std::filesystem::path& companion) {
    infernix::Engine engine(drafting(artifact, companion, 1, true));
    for (const bool all : {false, true}) {
        infernix::SpeculativeStats total;
        std::vector<std::uint64_t> by_position(7, 0);
        for (std::size_t i = 0; i < 4; ++i) {
            const auto result = engine.generate(engine.prepare(chat(texts()[i])), all ? whole(i, 256) : sampled(i, 256));
            total.rounds += result.speculative.rounds;
            total.drafted_tokens += result.speculative.drafted_tokens;
            total.accepted_tokens += result.speculative.accepted_tokens;
            for (std::size_t p = 0; p < result.speculative.accepted_per_position.size() && p < by_position.size(); ++p) {
                by_position[p] += result.speculative.accepted_per_position[p];
            }
        }
        report(std::string("fixed 7, ") + (all ? "whole vocabulary (coupled drafts)" : "top 20 (greedy drafts)") +
                   " at temperature 1",
               total);
        std::printf("  rounds accepting draft i:");
        for (const auto n : by_position) { std::printf(" %llu", static_cast<unsigned long long>(n)); }
        std::printf("\n");
    }
}

int check_engine(const std::filesystem::path& artifact, const std::filesystem::path& companion) {
    // Plain decoding's tokens: one request at a time, no drafter.
    // Three samplings: greedy, top-20 at temperature 1, and the default (the whole vocabulary at temperature 1).
    enum Sampling { kGreedy, kTop20, kWhole };
    const auto request = [](Sampling s, std::size_t i) {
        return s == kGreedy ? greedy(160) : s == kTop20 ? sampled(i) : whole(i);
    };
    const char* names[] = {", greedy ", ", sampled ", ", whole vocabulary "};
    std::vector<Tokens> plain[3];
    {
        infernix::EngineOptions options = engine_options(artifact, 1);
        options.speculative.backend     = infernix::SpeculativeBackend::None;
        infernix::Engine engine(options);
        for (std::size_t i = 0; i < texts().size(); ++i) {
            for (const Sampling s : {kGreedy, kTop20, kWhole}) {
                plain[s].push_back(engine.generate(engine.prepare(chat(texts()[i])), request(s, i)).generated_token_ids);
            }
        }
    }
    for (const bool shortlist : {false, true}) {
        for (const std::uint32_t concurrency : {1U, 2U, 4U, 8U}) {
            for (const bool graphs : {true, false}) {
                if (!graphs && concurrency != 1) { continue; }
                infernix::EngineOptions options       = engine_options(artifact, concurrency);
                options.use_cuda_graph                = graphs;
                options.speculative.backend           = infernix::SpeculativeBackend::DFlash2;
                options.speculative.draft_model       = companion;
                options.speculative.draft_tokens_auto = true;
                options.speculative.draft_tokens      = 7;
                options.speculative.proposal_head =
                    shortlist ? infernix::ProposalHead::Optimized : infernix::ProposalHead::Full;
                infernix::Engine engine(options);
                const std::string label = std::string("dflash2 (") + (shortlist ? "--lm-head-draft, " : "") + "C = " +
                                          std::to_string(concurrency) + (graphs ? ", graphs" : ", eager") + ")";
                for (const Sampling s : {kGreedy, kTop20, kWhole}) {
                    std::vector<infernix::GenerationHandle> handles;
                    const std::size_t n = std::min<std::size_t>(texts().size(), std::max<std::uint32_t>(concurrency, 2));
                    for (std::size_t i = 0; i < n; ++i) {
                        handles.push_back(engine.submit(engine.prepare(chat(texts()[i])), request(s, i)));
                    }
                    infernix::SpeculativeStats total;
                    for (std::size_t i = 0; i < n; ++i) {
                        const auto result = handles[i].wait();
                        require_same(label + names[s] + std::to_string(i), plain[s][i], result.generated_token_ids);
                        require(result.speculative.backend == infernix::SpeculativeBackend::DFlash2,
                                label + ": the rounds are DFlash2's");
                        total.rounds += result.speculative.rounds;
                        total.drafted_tokens += result.speculative.drafted_tokens;
                        total.accepted_tokens += result.speculative.accepted_tokens;
                        total.fallback_steps += result.speculative.fallback_steps;
                    }
                    report(label + names[s] + "(every request)", total);
                }
                if (concurrency == 1 && graphs) {
                    // The same prompt again: resumed from the prefix cache (the drafter's rings restored with the state).
                    const auto again = engine.generate(engine.prepare(chat(texts()[0])), greedy(160));
                    require_same(label + ", resumed from the prefix cache", plain[kGreedy][0], again.generated_token_ids);
                    report(label + ", resumed", again.speculative);
                }
            }
        }
    }
    fixed_length_sampled(artifact, companion);
    return acceptance_gates(artifact, companion);
}

// Two real prompts' contexts for the oracle (700 tokens, and 2,300: past the ring's 2,048 slots): the openings of two of
// the repository's documents (prose without repetition, so the drafter's confidence varies): an Engine with the
// DFlash2 drafter prefills each (the observer copies every prompt call's features), and its first drawn token (greedy)
// is the block's anchor.
std::vector<RealContext> real_contexts(const std::filesystem::path& artifact, const std::filesystem::path& companion) {
    std::vector<RealContext> out;
    infernix::Engine engine(drafting(artifact, companion, 1, true));
    const std::pair<int, const char*> documents[] = {{700, "docs/maintainer/engine-architecture.md"}, {2300, "README.md"}};
    for (const auto& [T, name] : documents) {
        std::ifstream file(std::filesystem::path(INFERNIX_SOURCE_DIR) / name, std::ios::binary);
        require(file.good(), std::string("the document ") + name);
        const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        Tokens prompt = engine.tokenize_text(text);
        require(prompt.size() >= static_cast<std::size_t>(T), std::string("the document ") + name + " is long enough");
        prompt.resize(static_cast<std::size_t>(T));
        RealContext context;
        context.length = T;
        context.features.assign(static_cast<std::size_t>(T) * kF, 0);
        std::vector<bool> seen(static_cast<std::size_t>(T), false);
        m::testing::set_context_observer([&](const std::uint16_t* features, const std::int32_t* positions,
                                             std::int32_t count, void* stream) {
            cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "observer sync");
            const auto rows  = host<std::uint16_t>(features, static_cast<std::size_t>(count) * kF);
            const auto where = host<std::int32_t>(positions, static_cast<std::size_t>(count));
            for (std::int32_t c = 0; c < count; ++c) {
                const int position = where[static_cast<std::size_t>(c)];
                if (position < 0 || position >= T) { continue; }
                std::copy_n(rows.begin() + static_cast<std::ptrdiff_t>(c) * kF, kF,
                            context.features.begin() + static_cast<std::ptrdiff_t>(position) * kF);
                seen[static_cast<std::size_t>(position)] = true;
            }
        });
        const auto result = engine.generate(engine.prepare_tokens(prompt), greedy(1));
        m::testing::set_context_observer({});
        require(result.generated_token_ids.size() == 1, "the prefill draws a token");
        context.anchor = result.generated_token_ids[0];
        for (int t = std::max(0, T - (kWindow - 1)); t < T; ++t) {
            require(seen[static_cast<std::size_t>(t)], "a real context's features reach every position of its window");
        }
        std::printf("real context of %d tokens: anchor %d\n", T, context.anchor);
        out.push_back(std::move(context));
    }
    return out;
}

} // namespace

int main() {
    const char* artifact  = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    const char* companion = std::getenv("INFERNIX_INKLING2_DFLASH2");
    if (artifact == nullptr || *artifact == '\0' || companion == nullptr || *companion == '\0') {
        std::cout << "skip: INFERNIX_INKLING2_ARTIFACT and INFERNIX_INKLING2_DFLASH2 are needed\n";
        return 77;
    }
    try {
        int failures = 0;
        if (std::getenv("INFERNIX_INKLING2_DFLASH2_ENGINE_ONLY") == nullptr) {
            const std::vector<RealContext> contexts = real_contexts(artifact, companion); // its Engine released first
            DeviceContext device(0);
            failures += check_drafter(device, artifact, companion, false, contexts);
            failures += check_drafter(device, artifact, companion, true, contexts);
        }
        if (std::getenv("INFERNIX_INKLING2_DFLASH2_ORACLE_ONLY") == nullptr) { failures += check_engine(artifact, companion); }
        std::cout << (failures == 0 ? "PASS" : "FAIL") << '\n';
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
