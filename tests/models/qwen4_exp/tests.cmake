infernix_add_test(infernix_qwen4_exp_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_cache.cpp"
  LIBRARIES infernix_qwen4_exp_calibrated_order
  NEEDS_SOURCE_DIR)

# The SSD tier's host level (design §19.3.7, R10 host part): slots, read agent, admission. CPU and disk.
infernix_add_test(infernix_qwen4_exp_host_expert_tier_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_expert_tier.cpp"
  LIBRARIES infernix_model_runtime)

# The SSD tier's RAM-level controller (design §19.3.7 RT2), host only.
infernix_add_test(infernix_qwen4_exp_host_tier_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_host_tier.cpp"
  LIBRARIES infernix_expert_offload_cache)

infernix_add_test(infernix_qwen4_exp_ngram_hash_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_hash.cpp"
  LIBRARIES infernix_model_loading
  NEEDS_SOURCE_DIR)

# The routed experts' activation precision as the config resolves it, with --expert-a4-activations
# (host only, a synthetic directory).
infernix_add_test(infernix_qwen4_exp_expert_activations_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_activations.cpp"
  LIBRARIES infernix_model_loading)

# The n-gram volume reader on a synthetic volume (CPU and disk only).
infernix_add_test(infernix_qwen4_exp_ngram_volume_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_volume.cpp"
  LIBRARIES infernix_model_runtime)
# TEMPORARY with INFERNIX_TMP_NGRAM_SCATTER: the large call's scattered reading order, same oracle.
add_test(NAME infernix_qwen4_exp_ngram_volume_scatter_test COMMAND infernix_qwen4_exp_ngram_volume_test)
set_tests_properties(infernix_qwen4_exp_ngram_volume_scatter_test
  PROPERTIES ENVIRONMENT "INFERNIX_TMP_NGRAM_SCATTER=1")

# The forward pass on the real artifact (INFERNIX_QWEN4_ARTIFACT) against the FP64 reference.
infernix_add_test(infernix_qwen4_exp_forward_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_forward_real.cpp"
  LIBRARIES infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_forward_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Causal scoring through the Engine against the Forward's teacher-forced logits (a --dump-logits
# reference of the same ids). Skips without INFERNIX_QWEN4_ARTIFACT and its arguments.
infernix_add_test(infernix_qwen4_exp_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_score_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_score_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The lossless route on the real artifact (docs/maintainer/lossless-route.md): deterministic, width- and
# selection-invariant scores. Skips without INFERNIX_QWEN4_ARTIFACT.
infernix_add_test(infernix_qwen4_exp_lossless_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lossless_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_lossless_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

infernix_add_test(infernix_qwen4_exp_lossless_generation_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lossless_generation_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_lossless_generation_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The SSD tier's in-place records on the real artifact (memory track R6, RT3/RT4): streamed records
# read through DirectReadQueue equal the pinned load's bytes. Skips without INFERNIX_QWEN4_ARTIFACT.
infernix_add_test(infernix_qwen4_exp_expert_store_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_store_real.cpp"
  LIBRARIES infernix_model_loading)
set_tests_properties(infernix_qwen4_exp_expert_store_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The internal route trace through the public Engine on the real artifact (memory track R0): ids
# identical with the trace on, the trace consistent with the requests, and the graph executables'
# device memory. Skips without INFERNIX_QWEN4_ARTIFACT and a prompt list (arguments).
infernix_add_test(infernix_qwen4_exp_route_trace_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_trace_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime infernix::json)
set_tests_properties(infernix_qwen4_exp_route_trace_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Engine-level VRAM pressure (memory RT13): the expert cache shrinks and grows under a fake source.
infernix_add_test(infernix_qwen4_exp_vram_pressure_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vram_pressure_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_vram_pressure_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Engine-level recovery from a failed expert read: the round's requests fail, the Engine keeps
# serving, and a later request equals the run before the fault.
infernix_add_test(infernix_qwen4_exp_expert_fault_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_fault_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_expert_fault_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The Vision encode window's placement and visual columns (host only).
infernix_add_test(infernix_qwen4_exp_vision_window_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_window.cpp"
  LIBRARIES infernix_model_runtime)

# FP4-Local's nvfp4_rg16_v1 vision MLP and merger at binding, Linear planning and offload staging
# (host only).
infernix_add_test(infernix_qwen4_exp_vision_rg16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_rg16.cpp"
  LIBRARIES infernix_model_loading)

# The prefix cache's prefill call planner: coverage, grids, exact-tap admission (host only).
infernix_add_test(infernix_qwen4_exp_call_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_call_plan.cpp"
  LIBRARIES infernix_model_runtime)

# RoPE staging of a lane: prompt M-RoPE, later tokens, block starts, MTP sub-chunks (host only).
infernix_add_test(infernix_qwen4_exp_rope_positions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rope_positions.cpp"
  LIBRARIES infernix_model_runtime)

# The startup RAM ledger (host only).
infernix_add_test(infernix_qwen4_exp_memory_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_memory_plan.cpp"
  LIBRARIES infernix_model_loading)

# Prefix-cache state images and page records on synthetic Device pools (no model; skips without a GPU).
infernix_add_test(infernix_qwen4_exp_prefix_state_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_state_image.cpp"
  LIBRARIES infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_prefix_state_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The prefix cache through the public Engine on the real artifact (INFERNIX_QWEN4_ARTIFACT): tap,
# Host-block and endpoint resumes against cold and lane-resident runs, plain and MTP.
infernix_add_test(infernix_qwen4_exp_prefix_cache_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_cache_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_prefix_cache_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# A continuation resumed after speculative decoding against a cold run, MTP and DFlash2 (INFERNIX_QWEN4_DFLASH2),
# on the reproducible route (bit-identical) and the default route (a teacher-forced log-probability bound).
infernix_add_test(infernix_qwen4_exp_resume_cold_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_resume_cold_real.cpp"
  LIBRARIES infernix_engine
  ARGS mtp dflash2)
set_tests_properties(infernix_qwen4_exp_resume_cold_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# A long prompt's layer walk while another lane decodes, with SSD-only experts (issue #5): both
# requests complete with their solo ids and no call reports a silent host.
infernix_add_test(infernix_qwen4_exp_walk_interleave_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_walk_interleave_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_walk_interleave_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

infernix_add_test(infernix_qwen4_exp_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_preemption_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_preemption_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# MTP drafts (one-hot and sampled-mtp-drafts) and a DFlash2 companion drafter (--draft-model) against
# one plain-decoding reference: the sampler's output distribution, greedy tokens, concurrency and the
# drafter's ring through a prefix resume.
infernix_add_test(infernix_qwen4_exp_speculative_sampling_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_sampling_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_speculative_sampling_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

infernix_add_test(infernix_qwen4_exp_decide_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_decide_real.cpp"
  LIBRARIES infernix_engine)
set_tests_properties(infernix_qwen4_exp_decide_real_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Fast-path selection rules: where each path applies and which configurations qualify it.
infernix_add_test(infernix_qwen4_exp_fast_paths_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fast_paths.cpp"
  LIBRARIES infernix_model_runtime)
