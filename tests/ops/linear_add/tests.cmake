add_library(infernix_linear_add_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_add_test_common.cpp")
infernix_test_includes(infernix_linear_add_test_support)
infernix_op_oracle_options(infernix_linear_add_test_support)
target_link_libraries(infernix_linear_add_test_support PUBLIC infernix_ops)

infernix_add_op_test(infernix_linear_add_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES infernix_linear_add_test_support)

infernix_add_op_test(infernix_linear_add_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES infernix_linear_add_test_support)

infernix_add_op_test(infernix_linear_add_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES infernix_linear_add_test_support)

infernix_add_op_test(infernix_linear_add_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES infernix_linear_add_test_support)

infernix_add_op_test(infernix_linear_add_nvfp4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_linear_add_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_linear_add_residual_width_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_residual_width.cpp"
  LIBRARIES infernix_ops)

add_test(NAME infernix_linear_add_fp8_wide_test
  COMMAND infernix_linear_add_fp8_test --wide-only)
set_tests_properties(infernix_linear_add_fp8_wide_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
