infernix_add_test(infernix_bench_support_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_infernix_bench_support.cpp"
          ${PROJECT_SOURCE_DIR}/bench/inference/infernix_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_engine infernix::json)

target_include_directories(infernix_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/inference)

infernix_add_op_test(infernix_bench_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_bench_fixtures.cu"
  LIBRARIES infernix_ops)
target_include_directories(infernix_bench_fixtures_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

add_executable(infernix_context_cost_measure_test
  "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost_measure.cpp"
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)

target_include_directories(infernix_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)

add_test(NAME infernix_context_cost_measure_test COMMAND infernix_context_cost_measure_test)
