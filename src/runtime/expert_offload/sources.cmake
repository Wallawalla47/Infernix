# The shared expert-offload component (docs/maintainer/qwen3_5-122b-offload-design.md §7): the pieces
# Qwen3.8-Flash-Next's and Qwen3.5's Programs both instantiate. Bank loading, the expert store and the
# memory plan join infernix_model_loading; the residency, tiers, monitor, route trace and expert stream
# join infernix_model_runtime.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/expert_bank.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_store.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/memory_plan.cpp"
)

target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/expert_offload.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_residency.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_stream.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host_expert_tier.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/route_trace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vram_monitor.cpp"
)

# The host expert cache (Qwen3.8-Flash-Next design §9: LFRU, frame pool, saved state, the SSD tier's
# RAM level) is a host-only library its conformance tests link directly.
add_library(infernix_expert_offload_cache STATIC
  "${CMAKE_CURRENT_LIST_DIR}/expert_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/expert_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host_tier.cpp")
infernix_internal_includes(infernix_expert_offload_cache)
target_link_libraries(infernix_model_runtime PUBLIC infernix_expert_offload_cache)
