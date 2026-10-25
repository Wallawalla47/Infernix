// The q4-w4a8 fast path of the Q4 LinearSwiGLU (LinearRoutes::Q4W4A8) against the FP64 oracle.

#include "ops/op_family.h"
#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

INFERNIX_OP_TEST(linear_swiglu_q4_w4a8) {
    using namespace infernix;
    using namespace infernix::test::linear_swiglu;

    try {
        int failures = 0;

        // Every call passes the flag. The route serves T >= 256 (a private threshold the route
        // sweep sets; the exact-route cases below pin it: move them with it). Below it, the flag
        // must leave the exact route's bits, at each of its schedules: the paired GEMV (1), the
        // small-T tiles (2), the materialized projection (128), the narrow-tail folded MMA (129) and
        // the folded MMA (255). The int8 criterion covers both sides of the threshold.
        constexpr std::array<std::int32_t, 9> kTokenCases{1, 2, 128, 129, 255, 256, 257, 1024, 4096};
        constexpr std::array<std::int32_t, 4> kGraphCases{255, 256, 257, 4096};
        constexpr std::array<std::int32_t, 5> kExactRouteCases{1, 2, 128, 129, 255};
        failures += run_profile("LinearSwiGLU Q4 W4A8",
                                {QType::Q4_G64_FP16, 34816, 5120, 17408, 1403U,
                                 ActivationCompute::A8Int, ops::LinearRoutes::Q4W4A8},
                                kTokenCases, kGraphCases, kExactRouteCases);

        // Inputs int8 represents exactly: the route must then match the oracle as tightly as the
        // A16 routes, which a quantization-sized criterion could not show (a wrong group scale or
        // code permutation on one group costs about 1/sqrt(80) of the norm).
        constexpr std::array<std::int32_t, 3> kRepresentableCases{256, 257, 1024};
        failures += run_profile("LinearSwiGLU Q4 W4A8 int8-representable",
                                {QType::Q4_G64_FP16, 34816, 5120, 17408, 1405U,
                                 ActivationCompute::A8Int, ops::LinearRoutes::Q4W4A8,
                                 ActivationPattern::Int8Representable},
                                kRepresentableCases, std::array<std::int32_t, 1>{257});

        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU Q4 W4A8 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU Q4 W4A8 test failed: " << error.what() << '\n';
        return 1;
    }
}
