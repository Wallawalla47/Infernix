// The DSpark drafter's configuration against the geometry its kernels are built for (models/inkling2/config.h
// require_dspark_kernel_geometry; host only). Inkling 2 Mini's drafter, as its artifact stores it, passes; a drafter of
// any other shape or norm epsilon is refused at load, before its weights bind at their own shapes into buffers sized
// for Inkling 2 Mini's.

#include "artifact/schema.h"
#include "models/inkling2/config.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace infernix::models::inkling2;

// Inkling 2 Mini's drafter, as D:\Infernix-staging\Inkling-2-Mini-Infernix.infernix stores it (2026-11-26).
DSparkConfig mini() {
    DSparkConfig d;
    d.layers           = 3;
    d.heads            = 24;
    d.kv_heads         = 8;
    d.head_dim         = 128;
    d.intermediate     = 9216;
    d.layer_types      = {AttentionKind::Sliding, AttentionKind::Full, AttentionKind::Sliding};
    d.sliding_window   = 1024;
    d.rope_theta       = 1.0e6F;
    d.rms_norm_eps     = 1e-6F;
    d.block_size       = 8;
    d.markov_rank      = 256;
    d.target_layer_ids = {5, 23, 41, 48};
    return d;
}

int failures = 0;

void refused(const std::function<void(DSparkConfig&)>& change, const std::string& label) {
    DSparkConfig d = mini();
    change(d);
    try {
        require_dspark_kernel_geometry(d);
    } catch (const infernix::artifact::ArtifactError& error) {
        if (std::string(error.what()).find("Inkling 2 Mini's drafter") == std::string::npos) {
            std::cerr << "FAIL: " << label << ": refused without naming the drafter's geometry: " << error.what() << '\n';
            ++failures;
        }
        return;
    }
    std::cerr << "FAIL: " << label << ": accepted\n";
    ++failures;
}

} // namespace

int main() {
    try {
        require_dspark_kernel_geometry(mini());
    } catch (const std::exception& error) {
        std::cerr << "FAIL: Inkling 2 Mini's drafter refused: " << error.what() << '\n';
        return 1;
    }
    refused([](DSparkConfig& d) { d.heads = 32; }, "32 query heads");
    refused([](DSparkConfig& d) { d.kv_heads = 4; }, "4 KV heads");
    refused([](DSparkConfig& d) { d.head_dim = 64; }, "head dim 64");
    refused([](DSparkConfig& d) { d.intermediate = 8192; }, "MLP 8192");
    refused([](DSparkConfig& d) { d.sliding_window = 2048; }, "window 2048");
    refused([](DSparkConfig& d) { d.block_size = 16; }, "block 16");
    refused([](DSparkConfig& d) { d.markov_rank = 512; }, "Markov rank 512");
    refused([](DSparkConfig& d) { d.target_layer_ids = {5, 23, 48}; }, "3 features");
    refused([](DSparkConfig& d) { d.rms_norm_eps = 1e-5F; }, "rms_norm_eps 1e-5");
    refused([](DSparkConfig& d) { d.layer_types[1] = AttentionKind::Sliding; }, "three sliding layers");
    refused([](DSparkConfig& d) {
        d.layers = 4;
        d.layer_types.push_back(AttentionKind::Sliding);
    }, "four layers");
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
