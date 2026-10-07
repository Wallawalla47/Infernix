add_library(infernix_linear_swiglu_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_swiglu_test_common.cpp")
infernix_test_includes(infernix_linear_swiglu_test_support)
infernix_op_oracle_options(infernix_linear_swiglu_test_support)
target_link_libraries(infernix_linear_swiglu_test_support PUBLIC infernix_ops)

infernix_add_op_test(infernix_linear_swiglu_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES infernix_linear_swiglu_test_support)

infernix_add_op_test(infernix_linear_swiglu_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES infernix_linear_swiglu_test_support)

infernix_add_op_test(infernix_linear_swiglu_nvfp4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4.cpp"
  LIBRARIES infernix_linear_swiglu_test_support)

infernix_add_op_test(infernix_linear_swiglu_fp8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8.cpp"
  LIBRARIES infernix_linear_swiglu_test_support)
