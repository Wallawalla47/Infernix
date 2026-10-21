// The lossless route's correctly rounded functions (src/ops/lossless/cr_math.cuh) against
// lossless_cr_oracle.py's exact results: run as python lossless_cr_oracle.py <this executable>, which
// passes the vector file (records of function, input bits, expected bits). Every result must match.

#include "core/arena.h"
#include "core/device.h"
#include "ops/lossless/cr_math.cuh"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

using namespace infernix;
namespace ll = infernix::ops::lossless;

namespace {

__global__ void evaluate(const std::uint32_t* func, const std::uint32_t* input, std::uint32_t* out, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) { return; }
    const float x = __uint_as_float(input[i]);
    switch (func[i]) {
    case 0: out[i] = __float_as_uint(ll::cr_exp(x)); break;
    case 1: out[i] = __float_as_uint(ll::cr_sigmoid(x)); break;
    case 2: out[i] = ll::cr_sigmoid_bf16(x); break;
    case 3: out[i] = __float_as_uint(ll::cr_silu(x)); break;
    case 4: out[i] = ll::cr_silu_bf16(x); break;
    case 5: out[i] = __float_as_uint(ll::cr_softplus(x)); break;
    case 6: out[i] = __float_as_uint(ll::cr_rsqrt(x)); break;
    case 7:
    case 8: {
        float c, s;
        ll::cr_cos_sin(x, c, s);
        out[i] = __float_as_uint(func[i] == 7 ? c : s);
        break;
    }
    default: out[i] = 0xFFFFFFFFU; break;
    }
}

const char* kNames[] = {"exp", "sigmoid", "sigmoid_bf16", "silu", "silu_bf16", "softplus", "rsqrt", "cos", "sin"};

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_lossless_cr_math <vectors.bin> (run through lossless_cr_oracle.py)\n";
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), 4);
    std::vector<std::uint32_t> func(count), input(count), want(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t r[3];
        in.read(reinterpret_cast<char*>(r), 12);
        func[i] = r[0];
        input[i] = r[1];
        want[i] = r[2];
    }
    if (!in) {
        std::cerr << "FAIL could not read the vectors\n";
        return 1;
    }
    DeviceBuffer df(count * 4ULL), dx(count * 4ULL), dout(count * 4ULL);
    df.copy_from_host(func.data(), count * 4ULL);
    dx.copy_from_host(input.data(), count * 4ULL);
    evaluate<<<(count + 255) / 256, 256>>>(static_cast<const std::uint32_t*>(df.p), static_cast<const std::uint32_t*>(dx.p),
                                          static_cast<std::uint32_t*>(dout.p), static_cast<int>(count));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint32_t> got(count);
    dout.copy_to_host(got.data(), count * 4ULL);
    std::size_t wrong[9] = {}, total[9] = {};
    for (std::uint32_t i = 0; i < count; ++i) {
        ++total[func[i]];
        if (got[i] != want[i]) {
            if (wrong[func[i]] < 5) {
                std::printf("%s(0x%08x): got 0x%08x want 0x%08x\n", kNames[func[i]], input[i], got[i], want[i]);
            }
            ++wrong[func[i]];
        }
    }
    int failures = 0;
    for (int f = 0; f < 9; ++f) {
        std::printf("%s: %zu of %zu correctly rounded\n", kNames[f], total[f] - wrong[f], total[f]);
        failures += wrong[f] != 0;
    }
    if (failures != 0) {
        std::cerr << "FAIL " << failures << " functions are not correctly rounded everywhere\n";
        return 1;
    }
    return 0;
}
