# Qwen4Exp (Qwen3.8-Flash-Next): its own model architecture (docs/maintainer/qwen3_8-flash-next-design.md
# §7). Config, binding and loading join ninfer_model_loading; execution and the Program join
# ninfer_model_runtime.
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/ngram_hash.cpp"
)

target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/ngram_volume.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
)

# The host expert cache (design §9) is a host-only library its conformance test links directly.
add_library(ninfer_qwen4_exp_expert_cache STATIC
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/expert_cache.cpp")
ninfer_internal_includes(ninfer_qwen4_exp_expert_cache)
target_link_libraries(ninfer_model_runtime PUBLIC ninfer_qwen4_exp_expert_cache)
