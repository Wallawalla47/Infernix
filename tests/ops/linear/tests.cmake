add_library(infernix_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
infernix_test_includes(infernix_linear_test_support)
infernix_op_oracle_options(infernix_linear_test_support)
target_link_libraries(infernix_linear_test_support PUBLIC infernix_ops)

infernix_add_op_test(infernix_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES infernix_linear_test_support)

# Back-to-back TMA staging race probe for both the plain Linear and fused LinearSwiGLU routes:
# distinct weights fired without inter-launch sync, each checked against an isolated reference.
infernix_add_op_test(infernix_nvfp4_tma_staging_race_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_tma_staging_race.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES infernix_linear_test_support)

# FP32 projections (router and lm_head logits), BF16 and q8_g32_fp16 weights, against FP64.
infernix_add_op_test(infernix_projection_fp32_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_projection_fp32.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_op_test(infernix_resident_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_resident_moe.cpp"
  LIBRARIES infernix_linear_test_support)
