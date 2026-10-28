add_library(infernix_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
infernix_test_includes(infernix_linear_test_support)
infernix_op_oracle_options(infernix_linear_test_support)
target_link_libraries(infernix_linear_test_support PUBLIC infernix_ops)

infernix_add_family_op_test(infernix_linear_q4_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_q5_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_q6_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_q8_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_nvfp4_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES infernix_linear_test_support)

# The column-invariant NVFP4 A16 slice ring tile (from 17 columns, 32 at K=17408): bit for bit the
# sliced-K tiles at the 27B MLP shapes, repeatable over dirty shared memory, and the FP64 oracle.
infernix_add_family_op_test(infernix_nvfp4_a16_slice_ring_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16_slice_ring.cpp" "${CMAKE_CURRENT_LIST_DIR}/../dirty_shared_memory.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_nvfp4_a16_slice_ring_test PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_family_op_test(infernix_linear_nvfp4_a4_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES infernix_linear_test_support)

# Back-to-back TMA staging race probe for both the plain Linear and fused LinearSwiGLU routes:
# distinct weights fired without inter-launch sync, each checked against an isolated reference.
infernix_add_family_op_test(infernix_nvfp4_tma_staging_race_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_tma_staging_race.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_linear_fp8_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_fp8_a8_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES infernix_linear_test_support)

# Repeat determinism of fp8-a8-pair-block32 and fp8-a8-pair on recipe C's FP8 problems: thousands of
# identical calls on [2560,6144], every output bit compared on the device, eager and graph-replayed.
infernix_add_family_op_test(infernix_fp8_a8_pair_repeat_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8_pair_repeat.cu"
  LIBRARIES infernix_linear_test_support)

# MXFP8 (fp8_e4m3fn_g32_e8m0) with 16-bit activations: NVIDIA FP4-Local's Flash-Next shapes.
infernix_add_family_op_test(infernix_linear_mxfp8_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mxfp8_a16.cpp" "${CMAKE_CURRENT_LIST_DIR}/../dirty_shared_memory.cu"
  LIBRARIES infernix_linear_test_support)

# f16-accumulate-prefill (FP16 Tensor Cores, FP16 chunk sums, FP32 carry) on the FP8 and MXFP8
# problems: FP64 oracle beside the exact route's error, encoder exactness, worst-case overflow,
# determinism, column invariance, seams, workspace and the MXFP8 GDN input rows.
infernix_add_family_op_test(infernix_linear_f16_accumulate_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_f16_accumulate.cpp"
  LIBRARIES infernix_linear_test_support)

# FP4-Local's MXFP8 shared expert on its own stream beside the offloaded routed experts, as the
# Qwen4Exp decode overlaps them: every output equals the serialized run's bits.
infernix_add_family_op_test(infernix_mxfp8_moe_overlap_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mxfp8_overlap.cu"
  LIBRARIES infernix_linear_test_support)

# Recipe C's FP8 shared expert on its own stream beside the offloaded routed experts, as the Qwen4Exp
# decode overlaps them: every output equals the serialized run's bits.
infernix_add_family_op_test(infernix_fp8_moe_overlap_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_moe_overlap.cu"
  LIBRARIES infernix_linear_test_support)

# The TMA-staged FP8 A8 route on its own stream beside the offloaded routed experts: every output equals
# the serialized run's bits (the consumers' proxy fence; unfenced it fails here, the other TMA routes do not).
infernix_add_family_op_test(infernix_tma_moe_overlap_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tma_moe_overlap.cu"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_bf16_a16_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES infernix_linear_test_support)

# FP32 vocabulary-head logits through ops::linear: Q6, Q8, row-FP8 and BF16 heads on every route, against
# FP64 and the same call's BF16 output.
infernix_add_family_op_test(infernix_linear_fp32_head_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_fp32_head.cpp"
  LIBRARIES infernix_linear_test_support)

# Windows TMA launches read their descriptors in place from the kernel parameters; these run every TMA
# family's staged instantiations, the fallback where the parameter block is misaligned, against the same
# oracles, the staging race probe and the overlap probe.
infernix_add_family_op_test(infernix_tma_staging_test FAMILY linear NO_TEST
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tma_staging.cu")
if(WIN32)
  foreach(tma_test infernix_linear_fp8_a8_test infernix_linear_fp8_a16_test infernix_fp8_a8_pair_repeat_test
          infernix_linear_mxfp8_a16_test infernix_linear_f16_accumulate_test infernix_linear_nvfp4_a4_test
          infernix_nvfp4_tma_staging_race_test infernix_tma_moe_overlap_test infernix_linear_bf16_a16_test)
    string(REGEX REPLACE "^infernix_(.*)_test$" "\\1" tma_member "${tma_test}")
    add_test(NAME ${tma_test}_staging COMMAND ${infernix_tma_staging_test_COMMAND} ${tma_member})
    set_tests_properties(${tma_test}_staging PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()

# FP32 projections (router and lm_head logits), BF16 and q8_g32_fp16 weights, against FP64.
infernix_add_family_op_test(infernix_projection_fp32_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_projection_fp32.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_resident_moe_test FAMILY linear
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_resident_moe.cpp"
  LIBRARIES infernix_linear_test_support)
