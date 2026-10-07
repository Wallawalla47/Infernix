# Qwen4Exp (Qwen3.8-Flash-Next): its own model architecture (docs/maintainer/qwen3_8-flash-next-design.md
# §7). Config, binding and loading join infernix_model_loading; execution and the Program join
# infernix_model_runtime.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_store.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/memory_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/ngram_hash.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/expert_stream.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/ngram_volume.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_residency.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/rope_positions.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefill_walk.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/constraint.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision_window.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/route_trace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/host_expert_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vram_monitor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/call_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/state_image.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/prefix_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/prefix_program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/persist.cpp"
)

# The host expert cache (design §9) is a host-only library its conformance test links directly.
add_library(infernix_qwen4_exp_expert_cache STATIC
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/expert_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/expert_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/host_tier.cpp")
infernix_internal_includes(infernix_qwen4_exp_expert_cache)
target_link_libraries(infernix_model_runtime PUBLIC infernix_qwen4_exp_expert_cache)
