set(infernix_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  kimi_delta_attention
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  state_codec
  prepare_ragged_prefix
  rows
  scatter
  scatter_bf16_batch
  target_logprobs
  top_logprobs
  token_constraint
  position
  ple)
set(infernix_op_family_add_bias elementwise)
set(infernix_op_family_gelu elementwise)
set(infernix_op_family_silu_mul elementwise)
set(infernix_op_family_residual_add elementwise)
set(infernix_op_family_sigmoid_mul elementwise)
set(infernix_op_family_rmsnorm elementwise)
set(infernix_op_family_rmsnorm_pack_tail elementwise)
set(infernix_op_family_gated_rmsnorm elementwise)
set(infernix_op_family_l2norm elementwise)
set(infernix_op_family_gated_delta_net recurrent)
set(infernix_op_family_kimi_delta_attention recurrent)
set(infernix_op_family_causal_conv1d_silu recurrent)
set(infernix_op_family_layer_norm elementwise)
set(infernix_op_family_embedding elementwise)
set(infernix_op_family_argmax sampling)
set(infernix_op_family_gdn_gating recurrent)
set(infernix_op_family_gdn_gating_proj recurrent)
set(infernix_op_family_rope elementwise)
set(infernix_op_family_vision_pos_embed elementwise)
set(infernix_op_family_sampling sampling)
set(infernix_op_family_scalar elementwise)
set(infernix_op_family_cast elementwise)
set(infernix_op_family_state_codec elementwise)
set(infernix_op_family_prepare_ragged_prefix elementwise)
set(infernix_op_family_rows elementwise)
set(infernix_op_family_scatter elementwise)
set(infernix_op_family_scatter_bf16_batch elementwise)
set(infernix_op_family_target_logprobs sampling)
set(infernix_op_family_top_logprobs sampling)
set(infernix_op_family_token_constraint sampling)
set(infernix_op_family_position elementwise)
set(infernix_op_family_ple elementwise)
foreach(op IN LISTS infernix_op_tests)
  infernix_add_family_op_test(infernix_${op}_test FAMILY ${infernix_op_family_${op}}
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES infernix_ops)
endforeach()

# GGUF expert banks through the layer route (frames, zero-copy, staged and forked misses) against the
# exact host expert, bit for bit.
infernix_add_family_op_test(infernix_offloaded_moe_ggml_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_ggml.cu"
  LIBRARIES infernix_ops)
# GGUF expert banks on the 8-bit activation route (--a8-activations) through the layer route, against the
# CPU route's scalar 8-bit expert, bit for bit; calls past the route against the exact host expert.
infernix_add_family_op_test(infernix_offloaded_moe_ggml_a8_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_ggml_a8.cu"
  LIBRARIES infernix_ops)
# NVFP4 W4A16 expert banks on the 8-bit activation route (--a8-activations, NarrowA16::kA8) at both expert
# geometries through the layer route (frames, zero-copy, staged, copied-ahead, landed, forked and CPU-served
# misses; eager and graph-captured) against an independent scalar oracle of nvfp4_a8.h, bit for bit, with
# column-count invariance; a call past the route against the exact host expert.
infernix_add_family_op_test(infernix_offloaded_moe_w4a8_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_w4a8.cu"
  LIBRARIES infernix_ops)
# The experts' 8-bit wide route (--a8-activations at prefill widths, WideA16::kA8) on GGUF banks and NVFP4 W4A16
# banks at both geometries through the layer route: narrow experts bit for bit against the narrow 8-bit
# arithmetic, wide experts against the FP64 chain over the codec, placement and column-count invariance.
infernix_add_family_op_test(infernix_offloaded_moe_a8_wide_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_a8_wide.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_offloaded_moe_a8_wide_test PROPERTIES TIMEOUT 900)
# Block-FP8 expert banks (Qwen's official FP8 experts) through the layer route and the CPU route
# against the exact host expert, bit for bit; that reference against a bignum sum.
infernix_add_family_op_test(infernix_offloaded_moe_fp8_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_fp8.cu"
  LIBRARIES infernix_ops)
# Qwen3.5-122B-A10B's expert geometry (H 3072, I 1024) through the layer route (frames, zero-copy,
# staged, forked and CPU-served misses; narrow and wide widths) against the exact host expert, bit for
# bit, and its combine against the FP64 formula.
infernix_add_family_op_test(infernix_offloaded_moe_qwen3_5_122b_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_qwen3_5_122b.cu"
  LIBRARIES infernix_ops)
# The expert-a4-residual fast paths (two and three NVFP4 terms) at that geometry, 24 to 16,384 columns:
# gate/up, down and end to end against the FP64 expert oracle, narrow entries against the exact host
# expert bit for bit, repeat and staging determinism; 4,096 columns and more on the 64-token warp tiles.
infernix_add_family_op_test(infernix_offloaded_moe_a4_residual_122b_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_a4_residual_122b.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_offloaded_moe_a4_residual_122b_test PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
# fp8_e4m3fn_row_f32 (Qwen3.5-122B-A10B): the static and per-token A8 encodings bit for bit, the attention,
# GDN and residual projections against FP64 under A16, static and per-token A8 and the column-invariant
# route, and the per-row FP8 shared expert's and head's column invariance.
infernix_add_family_op_test(infernix_fp8_row_f32_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_row_f32.cpp"
  LIBRARIES infernix_ops)
# The canonical W4A16 projection of an nvfp4_rg16_v1 matrix at every registered shape (Qwen3.5-122B-A10B's
# head; Qwen3.8-Flash-Next FP4-Local's vision MLP and merger) on the GEMV and integer tensor-core routes,
# FP32 and BF16 outputs, against the CPU engine's exact row sums bit for bit, and the vision shapes'
# BF16 outputs against an FP64 oracle over independently decoded weights.
infernix_add_family_op_test(infernix_rg16_linear_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rg16_linear.cu"
  LIBRARIES infernix_ops)
# GGUF block projections against an independent exact oracle (every ggml format, K, width).
infernix_add_family_op_test(infernix_ggml_linear_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_linear.cpp"
  LIBRARIES infernix_ops)
# The 8-bit activation route of GGUF projections (--a8-activations) against a binary64 oracle of the
# quantized activations, every narrow route and width.
infernix_add_family_op_test(infernix_ggml_linear_a8_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_linear_a8.cu"
  LIBRARIES infernix_ops)
# Its wide route (LinearRoutes::GgmlA8Wide) against the same binary64 oracle at the route's and the tiles' widths and
# at prefill widths on the IQ3_S import's shapes; width, column and repeat invariance.
infernix_add_family_op_test(infernix_ggml_linear_a8_wide_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ggml_linear_a8_wide.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_ggml_linear_a8_wide_test PROPERTIES TIMEOUT 900)
# The --lossless route's projections: every output the correctly rounded exact sum (tensor-core route
# and exact fallback), against a host wide-integer oracle.
infernix_add_family_op_test(infernix_lossless_linear_test FAMILY lossless
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lossless_linear.cpp"
  LIBRARIES infernix_ops)
# Its correctly rounded binary32/BF16 functions against an exact decimal oracle (every BF16 input and
# random binary32 ones); the Python driver writes the vectors and runs the executable. Both exact
# oracles keep their deterministic output in oracle-cache/, keyed by their sources and Python version,
# so only a tree's first run after an oracle change computes it.
infernix_add_family_op_test(infernix_lossless_cr_math_test FAMILY lossless NO_TEST
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lossless_cr_math.cu"
  LIBRARIES infernix_ops)
add_test(NAME infernix_lossless_cr_math_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/lossless_cr_oracle.py
          $<TARGET_FILE:infernix_ops_lossless_test> lossless_cr_math ${CMAKE_CURRENT_BINARY_DIR}/oracle-cache)
# Its elementwise and row operations (norms, gates, the GDN convolution) against an exact oracle that
# follows docs/maintainer/lossless-route.md §2 step by step.
infernix_add_family_op_test(infernix_lossless_ops_test FAMILY lossless NO_TEST
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lossless_ops.cu"
  LIBRARIES infernix_ops)
add_test(NAME infernix_lossless_ops_test
  COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_LIST_DIR}/lossless_ops_oracle.py
          $<TARGET_FILE:infernix_ops_lossless_test> lossless_ops ${CMAKE_CURRENT_BINARY_DIR}/oracle-cache)
# The design reference's KV storage: the production append's planes and window rows, byte for byte.
infernix_add_family_op_test(infernix_lossless_kv_encode_test FAMILY lossless
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_lossless_kv_encode.cu"
  LIBRARIES infernix_ops)
infernix_add_family_op_test(infernix_linear_topk_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES infernix_ops)

# Screened vocabulary heads at Qwen3.8-Flash-Next's shapes (q4 proposal head, BF16 and q8 LM heads with
# their 2-bit screens): screen scores against FP64, rescored rows against projection_fp32's bits, the
# selection's margin and cap, the same argmax as the full head, column invariance and graph replay.
infernix_add_family_op_test(infernix_screened_head_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_screened_head.cpp"
  LIBRARIES infernix_linear_test_support)
set_tests_properties(infernix_screened_head_test PROPERTIES TIMEOUT 1800)

infernix_add_family_op_test(infernix_candidate_selector_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_softmax_attention_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES infernix_ops
  ARGS --non-causal-only)
# The causal cases, one entry per KV format: together they run exactly the cases of a run without
# --kv-dtype, and `ctest -j` runs them beside the other tests, one attention entry at a time
# (infernix_attention_oracle below).
foreach(kv_dtype bf16 int8 fp8 nvfp4 k8v4)
  add_test(NAME infernix_softmax_attention_${kv_dtype}_test
    COMMAND ${infernix_softmax_attention_test_COMMAND} --kv-dtype ${kv_dtype})
  set_tests_properties(infernix_softmax_attention_${kv_dtype}_test PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Beyond the native 262,144 visible keys that --rope-yarn-factor opens (up to 1,048,576).
add_test(NAME infernix_softmax_attention_extended_test
  COMMAND ${infernix_softmax_attention_test_COMMAND} --extended)
set_tests_properties(infernix_softmax_attention_extended_test PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_family_op_test(infernix_sliding_window_attention_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_kv_cache_append_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_kv_cache_vq_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_vq.cu"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_vq_attention_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vq_attention.cu"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_rmsnorm_rope_test FAMILY elementwise
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_context_kv_materialize_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/ring_footprint.cu"
  LIBRARIES infernix_ops)
# A device read and synchronization per layer of every case: 14 s alone, past its 900 s timeout beside the suite's
# GPU-heavy tests (2026-10-28, -j 8), whose kernels each synchronization waits behind.
set_tests_properties(infernix_context_kv_materialize_test PROPERTIES RUN_SERIAL TRUE)

add_test(NAME infernix_kv_cache_append_nvfp4_test
  COMMAND ${infernix_kv_cache_append_test_COMMAND} --nvfp4-only)

add_test(NAME infernix_kv_cache_append_k8v4_test
  COMMAND ${infernix_kv_cache_append_test_COMMAND} --k8v4-only)

set_tests_properties(
  infernix_kv_cache_append_nvfp4_test
  infernix_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_family_op_test(infernix_prepare_masked_block_test FAMILY elementwise
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_sparse_moe_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_mtp_pack_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_mtp_round_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_speculative_round_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_speculative_tree_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_tree.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_gumbel_speculation_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gumbel_speculation.cpp"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_speculative_tree_test PROPERTIES TIMEOUT 600)
add_test(NAME infernix_speculative_tree_fp32_test COMMAND ${infernix_speculative_tree_test_COMMAND} --fp32-logits)
set_tests_properties(infernix_speculative_tree_fp32_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

infernix_add_family_op_test(infernix_attn_input_proj_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES infernix_ops)

# LinearPolicy::A16ColumnInvariant: a column's bits do not depend on the call's other columns.
infernix_add_family_op_test(infernix_column_invariant_projections_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_column_invariant_projections.cpp"
  LIBRARIES infernix_ops)

# The pipelined groupwise sliced-K kernel returns the release kernels' bits on every route and shape.
infernix_add_family_op_test(infernix_groupwise_sliced_k_pipeline_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_groupwise_sliced_k_pipeline.cpp"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_groupwise_sliced_k_pipeline_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1200)

infernix_add_family_op_test(infernix_attn_input_proj_fused_rmsnorm_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj_fused_rmsnorm.cpp"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_attn_input_proj_fused_rmsnorm_test PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_family_op_test(infernix_gdn_input_proj_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_dynamic_grouped_conv_prepare_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_linear_dynamic_grouped_conv_add_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_gdn_input_proj_conv_snapshot_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_gdn_input_proj_conv_record_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_gated_delta_net_replay_record_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_gated_delta_net_replay_record_conv_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record_conv.cpp"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_gdn_replay_fold_test FAMILY recurrent
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES infernix_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")

foreach(kv_dtype bf16 int8 fp8 nvfp4 k8v4)
  add_test(NAME infernix_softmax_attention_wide_${kv_dtype}_test
    COMMAND ${infernix_softmax_attention_test_COMMAND} --wide-only --kv-dtype ${kv_dtype})
  set_tests_properties(infernix_softmax_attention_wide_${kv_dtype}_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endforeach()

# Every softmax attention entry's oracle runs on all host threads (ops/host_parallel.h) and its
# kernels fill the GPU, so entries started together only oversubscribe both: the bf16 entry, 62 s
# alone, timed out at 900 s beside five others (2026-10-22). They take turns; other tests still run
# beside them.
set(infernix_attention_oracle_tests infernix_softmax_attention_test infernix_softmax_attention_extended_test)
foreach(kv_dtype bf16 int8 fp8 nvfp4 k8v4)
  list(APPEND infernix_attention_oracle_tests infernix_softmax_attention_${kv_dtype}_test
    infernix_softmax_attention_wide_${kv_dtype}_test)
endforeach()
set_tests_properties(${infernix_attention_oracle_tests} PROPERTIES RESOURCE_LOCK infernix_attention_oracle)
# Their turns are the run's critical path, so a tree without measured costs (CTestCostData.txt, which
# replaces COST once present) starts them first, with the longest single entry.
set(infernix_attention_first ${infernix_attention_oracle_tests} infernix_lossless_ops_test)
set_tests_properties(${infernix_attention_first} PROPERTIES COST 1000)

add_test(NAME infernix_sparse_moe_wide_test
  COMMAND ${infernix_sparse_moe_test_COMMAND} --wide-only)
set_tests_properties(infernix_sparse_moe_wide_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)

foreach(mode IN ITEMS ngram-only onehot-distribution mtp-onehot mtp-distribution ngram-negative-penalties wide-accept wide-distribution)
  string(REPLACE "-" "_" test_suffix "${mode}")
  add_test(NAME infernix_speculative_${test_suffix}_test
    COMMAND ${infernix_speculative_round_test_COMMAND} --${mode})
  set_tests_properties(infernix_speculative_${test_suffix}_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
endforeach()
set_tests_properties(infernix_speculative_wide_accept_test PROPERTIES TIMEOUT 600)
# The acceptance suites on FP32 verification logits (every vocabulary head writes FP32).
add_test(NAME infernix_speculative_round_fp32_test COMMAND ${infernix_speculative_round_test_COMMAND} --fp32-logits)
# Thousands of small synchronous launches: 119 s alone, past 900 s beside the suite's GPU-heavy tests (2026-10-27,
# -j 8), whose kernels it waits behind at every synchronization.
set_tests_properties(infernix_speculative_round_fp32_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)
foreach(mode IN ITEMS ngram-only mtp-onehot onehot-distribution mtp-distribution)
  string(REPLACE "-" "_" test_suffix "${mode}")
  add_test(NAME infernix_speculative_${test_suffix}_fp32_test
    COMMAND ${infernix_speculative_round_test_COMMAND} --fp32-logits --${mode})
  set_tests_properties(infernix_speculative_${test_suffix}_fp32_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
endforeach()
set_tests_properties(infernix_speculative_wide_distribution_test PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# The wide sweep, one entry per width and layer profile plus the batched cases: together they run
# exactly the cases of --wide-only, and `ctest -j` runs the width entries concurrently (at most about
# 1.3 GiB of device memory each). The batched entry's 64-round record folds hold about 10 GiB, so it
# runs alone rather than failing an allocation beside the others.
foreach(width 33 48 64)
  foreach(layers 48 30)
    add_test(NAME infernix_gdn_replay_fold_wide_w${width}_l${layers}_test
      COMMAND ${infernix_gdn_replay_fold_test_COMMAND} --wide-only --width ${width} --layers ${layers})
    set_tests_properties(infernix_gdn_replay_fold_wide_w${width}_l${layers}_test
      PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  endforeach()
endforeach()
add_test(NAME infernix_gdn_replay_fold_wide_batched_test
  COMMAND ${infernix_gdn_replay_fold_test_COMMAND} --wide-only --batched)
set_tests_properties(infernix_gdn_replay_fold_wide_batched_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)

# hyper_connection_mix against the FP64 closed formula: fused Q8 route (T <= 16) and composed route.
infernix_add_family_op_test(infernix_hyper_connection_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_hyper_connection.cpp"
  LIBRARIES infernix_linear_test_support)
set_tests_properties(infernix_hyper_connection_test PROPERTIES SKIP_RETURN_CODE 77)
# hyper_connection_stream_mean (DFlash2's target features) against the FP64 stream mean.
infernix_add_family_op_test(infernix_hyper_connection_stream_mean_test FAMILY elementwise
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_hyper_connection_stream_mean.cpp"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_hyper_connection_stream_mean_test PROPERTIES SKIP_RETURN_CODE 77)

# NVIDIA FP4-Local's PLE: NVFP4 n-gram rows decoded bit-exactly (one rounding of the exact product)
# and the gate with FP32 norm weights, against FP64.
infernix_add_family_op_test(infernix_ple_nvfp4_test FAMILY projection
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ple_nvfp4.cpp"
  LIBRARIES infernix_linear_test_support)
set_tests_properties(infernix_ple_nvfp4_test PROPERTIES SKIP_RETURN_CODE 77)

# Canonical W4A4 arithmetic and the CPU expert engine; host-only, needs no GPU.
infernix_add_family_op_test(infernix_offloaded_moe_cpu_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# Canonical W4A16 arithmetic and its CPU route (experts without activation scales); host-only.
infernix_add_family_op_test(infernix_offloaded_moe_a16_cpu_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_a16_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# The two-term A4 activations of W4A16 experts (expert-a4-residual-decode) and their CPU route against
# independent binary64 and FP64 oracles, on every ISA and team size; host-only.
infernix_add_family_op_test(infernix_offloaded_moe_a4r_cpu_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_a4r_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# The CPU route of GGUF (ggml) expert banks against the exact expert, on every ISA; host-only.
infernix_add_family_op_test(infernix_offloaded_moe_ggml_cpu_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_ggml_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)
# The CPU route's 8-bit activation functions (--a8-activations): every ISA bit-equal to scalar, scalar
# within its bound of a binary64 oracle; host-only.
infernix_add_family_op_test(infernix_offloaded_moe_ggml_a8_cpu_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_ggml_a8_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)
# The 8-bit activations of NVFP4 W4A16 experts on the CPU route (--a8-activations, nvfp4_a8.h) at both
# registered geometries: every ISA and width, and the worker team, bit-equal to the spec's scalar oracle; host-only.
infernix_add_family_op_test(infernix_offloaded_moe_w4a8_cpu_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_w4a8_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# The CPU worker team: any worker count gives expert_forward's bits.
infernix_add_family_op_test(infernix_offloaded_moe_team_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_team.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# The GPU narrow route against the golden hashes and the CPU engine, bit for bit.
infernix_add_family_op_test(infernix_offloaded_moe_cuda_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_cuda.cu"
  LIBRARIES infernix_ops)

# The layer route (routing, dispatch, staged or zero-copy misses) against the CPU engine.
# Speculative verification's state Ops against their committing forms, bit for bit.
infernix_add_family_op_test(infernix_speculative_state_ops_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_state_ops.cu"
  LIBRARIES infernix_ops)

# QSA index queries, pooled keys and tails, selection and attention against FP64 oracles. One
# entry per independent part (infernix_qsa_<part>_test), so `ctest -j` runs them concurrently;
# together they run every case of the unsharded executable.
infernix_add_family_op_test(infernix_qsa_test FAMILY attention NO_TEST
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa.cpp"
  LIBRARIES infernix_ops)
foreach(part index bf16 int8 fp8 nvfp4 k8v4 vq2 k4v2)
  add_test(NAME infernix_qsa_${part}_test COMMAND ${infernix_qsa_test_COMMAND} --part ${part})
  set_tests_properties(infernix_qsa_${part}_test PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# QSA block selection against the kernel it replaced (bit for bit), the exact top-k of its own
# scores and an FP64 oracle, also under CUDA Graph replay.
infernix_add_family_op_test(infernix_qsa_select_test FAMILY attention
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa_select.cu"
  LIBRARIES infernix_ops)

infernix_add_family_op_test(infernix_offloaded_moe_layer_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_layer.cu"
  LIBRARIES infernix_ops
  ARGS --no-idle)

# The fused route's CPU plan at Qwen3.5-122B-A10B's and
# Qwen3.8-Flash-Next's routing against a host oracle of the plan, byte for byte.
infernix_add_family_op_test(infernix_offloaded_moe_plan_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_plan.cu"
  LIBRARIES infernix_ops)

# The wide route (experts with more than eight columns) against its FP64 W4A4 oracle.
infernix_add_family_op_test(infernix_offloaded_moe_wide_test FAMILY offloaded_moe
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_wide.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_offloaded_moe_wide_test PROPERTIES TIMEOUT 900)
# The layer test's idle-service checks, which bound a call's latency to 10 ms beyond the GPU's own:
# beside other GPU tests the same call's time ranged 40-75 ms and the check failed five times
# (2026-10-22), so they run alone; the rest of the layer test runs beside other tests.
add_test(NAME infernix_offloaded_moe_layer_idle_test COMMAND ${infernix_offloaded_moe_layer_test_COMMAND} --idle-only)
set_tests_properties(infernix_offloaded_moe_layer_idle_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

# The family executables of every op test above (infernix_ops_<family>_test).
infernix_finish_op_families()
