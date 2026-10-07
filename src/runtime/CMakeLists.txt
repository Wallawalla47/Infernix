add_library(infernix_runtime_support STATIC
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
  prefix_cache/block_hash.cpp
  prefix_cache/prefix_index.cpp
  prefix_cache/tap_planner.cpp
)
infernix_internal_includes(infernix_runtime_support)
target_link_libraries(infernix_runtime_support PUBLIC infernix_core PRIVATE infernix::json)

# Public Engine and common contracts; model execution remains in the model layer.
add_library(infernix_engine STATIC
  engine/model_instance.cpp
  engine/qwen4_exp_instance.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
infernix_internal_includes(infernix_engine)
target_include_directories(infernix_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(infernix_engine PRIVATE
  infernix_model_runtime
  infernix_artifact
  infernix_core
  infernix_runtime_support
  infernix_ops
  infernix_text
  infernix_media_decode)
add_library(infernix::engine ALIAS infernix_engine)
