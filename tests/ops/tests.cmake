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
  prepare_ragged_prefix
  rows
  scatter
  scatter_bf16_batch
  target_logprobs
  top_logprobs
  token_constraint
  position)
foreach(op IN LISTS infernix_op_tests)
  infernix_add_op_test(infernix_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES infernix_ops)
endforeach()

infernix_add_op_test(infernix_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES infernix_ops
  ARGS --non-causal-only)
# The causal cases, one entry per KV format: together they run exactly the cases of a run without
# --kv-dtype, and `ctest -j` can run them concurrently.
foreach(kv_dtype bf16 int8 fp8 nvfp4 k8v4)
  add_test(NAME infernix_softmax_attention_${kv_dtype}_test
    COMMAND infernix_softmax_attention_test --kv-dtype ${kv_dtype})
  set_tests_properties(infernix_softmax_attention_${kv_dtype}_test PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Beyond the native 262,144 visible keys that --rope-yarn-factor opens (up to 1,048,576).
add_test(NAME infernix_softmax_attention_extended_test
  COMMAND infernix_softmax_attention_test --extended)
set_tests_properties(infernix_softmax_attention_extended_test PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_op_test(infernix_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_kv_cache_vq_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_vq.cu"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_vq_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vq_attention.cu"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
  LIBRARIES infernix_ops)

add_test(NAME infernix_kv_cache_append_nvfp4_test
  COMMAND infernix_kv_cache_append_test --nvfp4-only)

add_test(NAME infernix_kv_cache_append_k8v4_test
  COMMAND infernix_kv_cache_append_test --k8v4-only)

set_tests_properties(
  infernix_kv_cache_append_nvfp4_test
  infernix_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_op_test(infernix_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_sparse_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_speculative_tree_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_tree.cpp"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_speculative_tree_test PROPERTIES TIMEOUT 600)

infernix_add_op_test(infernix_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_attn_input_proj_fused_rmsnorm_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj_fused_rmsnorm.cpp"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_attn_input_proj_fused_rmsnorm_test PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_op_test(infernix_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES infernix_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")

foreach(kv_dtype bf16 int8 fp8 nvfp4 k8v4)
  add_test(NAME infernix_softmax_attention_wide_${kv_dtype}_test
    COMMAND infernix_softmax_attention_test --wide-only --kv-dtype ${kv_dtype})
  set_tests_properties(infernix_softmax_attention_wide_${kv_dtype}_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endforeach()

add_test(NAME infernix_sparse_moe_wide_test
  COMMAND infernix_sparse_moe_test --wide-only)
set_tests_properties(infernix_sparse_moe_wide_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)

foreach(mode IN ITEMS ngram-only onehot-distribution mtp-onehot mtp-distribution ngram-negative-penalties wide-accept wide-distribution)
  string(REPLACE "-" "_" test_suffix "${mode}")
  add_test(NAME infernix_speculative_${test_suffix}_test
    COMMAND infernix_speculative_round_test --${mode})
  set_tests_properties(infernix_speculative_${test_suffix}_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 240)
endforeach()
set_tests_properties(infernix_speculative_wide_accept_test PROPERTIES TIMEOUT 600)
set_tests_properties(infernix_speculative_wide_distribution_test PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)

# The wide sweep, one entry per width and layer profile plus the batched cases: together they run
# exactly the cases of --wide-only, and `ctest -j` runs them concurrently.
foreach(width 33 48 64)
  foreach(layers 48 30)
    add_test(NAME infernix_gdn_replay_fold_wide_w${width}_l${layers}_test
      COMMAND infernix_gdn_replay_fold_test --wide-only --width ${width} --layers ${layers})
    set_tests_properties(infernix_gdn_replay_fold_wide_w${width}_l${layers}_test
      PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  endforeach()
endforeach()
add_test(NAME infernix_gdn_replay_fold_wide_batched_test
  COMMAND infernix_gdn_replay_fold_test --wide-only --batched)
set_tests_properties(infernix_gdn_replay_fold_wide_batched_test
  PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)

# hyper_connection_mix against the FP64 closed formula: fused Q8 route (T <= 16) and composed route.
infernix_add_op_test(infernix_hyper_connection_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_hyper_connection.cpp"
  LIBRARIES infernix_linear_test_support)
set_tests_properties(infernix_hyper_connection_test PROPERTIES SKIP_RETURN_CODE 77)

# Canonical W4A4 arithmetic and the CPU expert engine; host-only, needs no GPU.
infernix_add_op_test(infernix_offloaded_moe_cpu_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# Canonical W4A16 arithmetic and its CPU route (experts without activation scales); host-only.
infernix_add_op_test(infernix_offloaded_moe_a16_cpu_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_a16_cpu.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# The CPU worker team: any worker count gives expert_forward's bits.
infernix_add_op_test(infernix_offloaded_moe_team_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_team.cpp"
  LIBRARIES infernix_offloaded_moe_cpu)

# The GPU narrow route against the golden hashes and the CPU engine, bit for bit.
infernix_add_op_test(infernix_offloaded_moe_cuda_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_cuda.cu"
  LIBRARIES infernix_ops)

# The layer route (routing, dispatch, staged or zero-copy misses) against the CPU engine.
# Speculative verification's state Ops against their committing forms, bit for bit.
infernix_add_op_test(infernix_speculative_state_ops_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_state_ops.cu"
  LIBRARIES infernix_ops)

# QSA index queries, pooled keys and tails, selection and attention against FP64 oracles.
infernix_add_op_test(infernix_qsa_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa.cpp"
  LIBRARIES infernix_ops)

# QSA block selection against the kernel it replaced (bit for bit), the exact top-k of its own
# scores and an FP64 oracle, also under CUDA Graph replay.
infernix_add_op_test(infernix_qsa_select_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_qsa_select.cu"
  LIBRARIES infernix_ops)

infernix_add_op_test(infernix_offloaded_moe_layer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_layer.cu"
  LIBRARIES infernix_ops)

# The wide route (experts with more than eight columns) against its FP64 W4A4 oracle.
infernix_add_op_test(infernix_offloaded_moe_wide_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_offloaded_moe_wide.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_offloaded_moe_wide_test PROPERTIES TIMEOUT 900)
