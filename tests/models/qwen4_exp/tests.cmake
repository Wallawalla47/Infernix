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
