ninfer_add_test(ninfer_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  LIBRARIES ninfer_qwen4_exp_expert_cache
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES ninfer_model_loading
  NEEDS_SOURCE_DIR)

# The n-gram volume reader on a synthetic volume (CPU and disk only).
ninfer_add_test(ninfer_qwen4_exp_ngram_volume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_volume.cpp"
  LIBRARIES ninfer_model_runtime)

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

# Engine-level VRAM pressure (memory RT13): the expert cache shrinks and grows under a fake source.
ninfer_add_test(ninfer_qwen4_exp_vram_pressure_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vram_pressure_real.cpp"
  LIBRARIES ninfer_engine ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_vram_pressure_real_test PROPERTIES SKIP_RETURN_CODE 77)

# The Vision encode window's placement and visual columns (host only).
ninfer_add_test(ninfer_qwen4_exp_vision_window_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_window.cpp"
  LIBRARIES ninfer_model_runtime)

# The prefix cache's prefill call planner: coverage, grids, exact-tap admission (host only).
ninfer_add_test(ninfer_qwen4_exp_call_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_call_plan.cpp"
  LIBRARIES ninfer_model_runtime)

# RoPE staging of a lane: prompt M-RoPE, later tokens, block starts, MTP sub-chunks (host only).
ninfer_add_test(ninfer_qwen4_exp_rope_positions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rope_positions.cpp"
  LIBRARIES ninfer_model_runtime)

# The startup RAM ledger (host only).
ninfer_add_test(ninfer_qwen4_exp_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES ninfer_model_loading)

# Prefix-cache state images and page records on synthetic Device pools (no model; skips without a GPU).
ninfer_add_test(ninfer_qwen4_exp_prefix_state_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_state_image.cpp"
  LIBRARIES ninfer_model_runtime)
set_tests_properties(ninfer_qwen4_exp_prefix_state_test PROPERTIES SKIP_RETURN_CODE 77)

# The prefix cache through the public Engine on the real artifact (NINFER_QWEN4_ARTIFACT): tap,
# Host-block and endpoint resumes against cold and lane-resident runs, plain and MTP.
ninfer_add_test(ninfer_qwen4_exp_prefix_cache_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_cache_real.cpp"
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen4_exp_prefix_cache_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_preemption_real.cpp"
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen4_exp_preemption_real_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_qwen4_exp_decide_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_decide_real.cpp"
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen4_exp_decide_real_test PROPERTIES SKIP_RETURN_CODE 77)
