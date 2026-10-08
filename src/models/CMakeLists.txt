# Immutable loading and mutable execution have separate compile owners.
add_library(infernix_model_loading STATIC registry.cpp)
infernix_internal_includes(infernix_model_loading)
target_link_libraries(infernix_model_loading PUBLIC infernix_artifact infernix_text infernix_ops)

add_library(infernix_model_runtime STATIC)
infernix_internal_includes(infernix_model_runtime)
target_link_libraries(infernix_model_runtime
  PUBLIC infernix_model_loading infernix_ops infernix_runtime_support
  PRIVATE infernix_media_decode infernix_jinja infernix_grammar infernix_xgrammar)
target_include_directories(infernix_model_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(infernix_model_runtime SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)

include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
include(qwen4_exp/program_sources.cmake)
