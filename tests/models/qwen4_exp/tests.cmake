ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  LIBRARIES ninfer_qwen4_exp_expert_cache
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES ninfer_model_loading
  NEEDS_SOURCE_DIR)

# The forward pass on the real artifact (NINFER_QWEN4_ARTIFACT) against the FP64 reference.
ninfer_add_test(ninfer_qwen4_exp_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_forward_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The internal route trace through the public Engine on the real artifact (memory track R0): ids
# identical with the trace on, the trace consistent with the requests, and the graph executables'
# device memory. Skips without NINFER_QWEN4_ARTIFACT and a prompt list (arguments).
ninfer_add_test(ninfer_qwen4_exp_route_trace_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_trace_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_runtime ninfer::json)
set_tests_properties(ninfer_qwen4_exp_route_trace_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The startup RAM ledger (host only).
ninfer_add_test(ninfer_qwen4_exp_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES ninfer_model_loading)
