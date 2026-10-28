// Runs another linear test with every Windows TMA launch reading the staged device copy of its descriptors
// (core/tma_descriptor_staging.cuh), the route a misaligned kernel parameter block falls back to:
//   infernix_ops_linear_test tma_staging <test> [arguments...]

#include "ops/op_family.h"
#include "core/tma_descriptor_staging.cuh"

INFERNIX_OP_TEST(tma_staging) {
    infernix::tma_descriptor_force_staging(true);
    return infernix::test::run_op_family(argc, argv);
}
