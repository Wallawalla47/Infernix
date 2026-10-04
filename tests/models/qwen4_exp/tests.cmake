ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  LIBRARIES ninfer_qwen4_exp_expert_cache
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES ninfer_qwen4_exp_frontend
  NEEDS_SOURCE_DIR)
