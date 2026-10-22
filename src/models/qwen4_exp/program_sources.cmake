# Qwen4Exp (Qwen3.8-Flash-Next): its own model architecture (docs/maintainer/qwen3_8-flash-next-design.md
# §7). Config, binding and loading join infernix_model_loading; execution and the Program join
# infernix_model_runtime.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/ngram_hash.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward_lossless.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/dflash2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/ngram_volume.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/fast_paths.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/rope_positions.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefill_walk.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/dflash2_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/cpu_rates.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/constraint.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/score.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_window.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/call_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/state_image.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/prefix_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/prefix_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/persist.cpp"
)

# The calibrated expert order (the expert cache's cold-start seed, design §19.3.5 S4b) is host-only
# data over the shared expert cache (src/runtime/expert_offload/); its conformance test links it.
add_library(infernix_qwen4_exp_calibrated_order STATIC
  "${CMAKE_CURRENT_LIST_DIR}/program/calibrated_order.cpp")
infernix_internal_includes(infernix_qwen4_exp_calibrated_order)
target_link_libraries(infernix_qwen4_exp_calibrated_order PUBLIC infernix_expert_offload_cache)
target_link_libraries(infernix_model_runtime PUBLIC infernix_qwen4_exp_calibrated_order)
