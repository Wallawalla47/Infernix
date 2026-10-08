infernix_add_test(infernix_ngram_proposer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_proposer.cpp")

infernix_add_test(infernix_ngram_archive_test SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/test_ngram_archive.cpp"
  "${PROJECT_SOURCE_DIR}/src/models/qwen3_5/ngram.cpp")

infernix_add_test(infernix_qwen3_5_tree_width_controller_test SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/test_tree_width_controller.cpp"
  "${PROJECT_SOURCE_DIR}/src/models/qwen3_5/program/speculative/tree_width_controller.cpp")

infernix_add_test(infernix_ngram_graph_planning_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_graph_planning.cpp"
  LIBRARIES infernix_engine infernix_core infernix::json)
add_test(NAME infernix_ngram_graph_planning_real
  COMMAND infernix_ngram_graph_planning_test --real)
set_tests_properties(infernix_ngram_graph_planning_real PROPERTIES SKIP_RETURN_CODE 77)

foreach(check lifecycle archive thinking stop_chat concurrent)
  infernix_add_test(infernix_ngram_${check}_real
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_ngram_${check}_real.cpp"
    LIBRARIES infernix_engine)
  set_tests_properties(infernix_ngram_${check}_real PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

infernix_add_test(infernix_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES infernix_model_loading)

infernix_add_test(infernix_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES infernix_model_loading)

infernix_add_test(infernix_qwen3_5_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_engine infernix_core infernix::json)

# The reasoning-loop guard's repeated-passage measure (CPU only).
infernix_add_test(infernix_qwen3_5_reasoning_loop_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reasoning_loop.cpp"
  LIBRARIES infernix_model_runtime)

infernix_add_test(infernix_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES infernix_engine infernix_core)

infernix_add_test(infernix_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES infernix_engine infernix_core)

infernix_add_test(infernix_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES infernix_engine infernix_core)

infernix_add_test(infernix_qwen3_5_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_store.cpp"
  LIBRARIES infernix_engine infernix_core)

infernix_add_test(infernix_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES infernix_model_runtime infernix_model_loading infernix_core)

infernix_add_test(infernix_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES infernix_engine)

infernix_add_test(infernix_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES infernix_engine infernix::json)

infernix_add_test(infernix_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES infernix_engine infernix::json)

infernix_add_test(infernix_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES infernix_engine infernix::json)

infernix_add_test(infernix_qwen3_5_hybrid_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_hybrid_prefix_real.cpp"
  LIBRARIES infernix_engine)

set_tests_properties(
  infernix_qwen3_5_hybrid_prefix_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_test(infernix_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES infernix_engine)

infernix_add_test(infernix_qwen3_5_readout_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_readout_real.cpp"
  LIBRARIES infernix_engine)

set_tests_properties(
  infernix_qwen3_5_readout_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_test(infernix_qwen3_5_constraint_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_constraint_real.cpp"
  LIBRARIES infernix_engine)

set_tests_properties(
  infernix_qwen3_5_constraint_real_test
  PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_test(infernix_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES infernix_model_runtime infernix_engine)

infernix_add_test(infernix_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES infernix_engine)

# DFlash2 tree verification at K=7 over every batch-size route, on the INT8, K8V4, NVFP4, VQ2 and
# K4V2 caches.
foreach(kv IN ITEMS int8 k8v4 nvfp4 vq2 k4v2)
  add_test(NAME infernix_qwen3_5_dflash2_tree_${kv}_real_test
    COMMAND infernix_qwen3_5_dflash2_real_test 7 1 1 4 ${kv} 0 3 16,12,12,10)
  list(APPEND infernix_qwen3_5_tree_real_tests infernix_qwen3_5_dflash2_tree_${kv}_real_test)
endforeach()
add_test(NAME infernix_qwen3_5_dflash2_tree_auto_real_test
  COMMAND infernix_qwen3_5_dflash2_real_test 7 1 1 4 int8 0 3 auto)

# Greedy tree-decoded streams against the target's own greedy choice at every position.
infernix_add_test(infernix_qwen3_5_dflash2_tree_greedy_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_tree_greedy_real.cpp"
  LIBRARIES infernix_engine)

foreach(kv IN ITEMS k8v4 nvfp4 vq2 k4v2)
  add_test(NAME infernix_qwen3_5_dflash2_tree_greedy_${kv}_real_test
    COMMAND infernix_qwen3_5_dflash2_tree_greedy_real_test ${kv})
  list(APPEND infernix_qwen3_5_tree_real_tests infernix_qwen3_5_dflash2_tree_greedy_${kv}_real_test)
endforeach()
set_tests_properties(${infernix_qwen3_5_tree_real_tests}
  infernix_qwen3_5_dflash2_tree_auto_real_test
  infernix_qwen3_5_dflash2_tree_greedy_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real")
infernix_add_test(infernix_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES infernix_model_runtime infernix_model_loading infernix_core)

infernix_add_test(infernix_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES infernix_engine)

infernix_add_test(infernix_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES infernix_engine)

infernix_add_test(infernix_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../../test_tool_call_parser.cpp"
  LIBRARIES infernix_engine infernix::json)

infernix_add_test(infernix_qwen3_5_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_visual_scatter.cpp"
  LIBRARIES infernix_engine infernix_core)

add_test(NAME infernix_qwen3_5_agent_continuation_real_test
  COMMAND infernix_qwen3_5_prefix_real_test)
set_tests_properties(infernix_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "INFERNIX_PREFIX_REAL_SCENARIO=agent-continuation")

# A real-model test owns the single GPU while its artifact is resident.
set(infernix_qwen3_5_real_tests
  infernix_qwen3_5_loading_real_test
  infernix_qwen3_5_native_transactions_test
  infernix_qwen3_5_prefix_real_test
  infernix_qwen3_5_agent_continuation_real_test
  infernix_qwen3_5_preemption_real_test
  infernix_qwen3_5_grammar_real_test
  infernix_qwen3_5_tools_real_test
  infernix_qwen3_5_score_real_test
  infernix_qwen3_5_vision_workspace_test
  infernix_qwen3_5_dflash2_real_test
  infernix_qwen3_5_dflash_prefill_real_test
  infernix_qwen3_5_moe_real_test
  infernix_qwen3_5_dflash_real_test)
set_tests_properties(${infernix_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  infernix_qwen3_5_state_image_test
  infernix_qwen3_5_context_store_test
  infernix_qwen3_5_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

infernix_add_test(infernix_qwen3_5_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES infernix_model_runtime infernix_grammar infernix::json)

add_test(NAME infernix_qwen3_5_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "INFERNIX_TOOL_PROBE=$<TARGET_FILE:infernix_qwen3_5_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")

infernix_add_test(infernix_qwen3_5_text_qk_norm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_text_qk_norm_rope.cpp"
  LIBRARIES infernix_engine infernix_core infernix::json)

set_tests_properties(
  infernix_qwen3_5_text_qk_norm_rope_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
