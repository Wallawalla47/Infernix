# Chat-template render cost over synthetic agent histories (CPU only): one Jinja render and one
# whole CompiledChatTemplate request, with a byte-exact output dump for before/after comparison.
add_executable(infernix_chat_template_render_bench
  "${CMAKE_CURRENT_LIST_DIR}/chat_template_render_bench.cpp")
infernix_internal_includes(infernix_chat_template_render_bench)
target_compile_definitions(infernix_chat_template_render_bench PRIVATE
  INFERNIX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(infernix_chat_template_render_bench PRIVATE
  infernix_model_runtime infernix_jinja infernix::json)
