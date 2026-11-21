# Inkling2Loop (Thinking Machines Inkling 2 Mini): its own model architecture
# (docs/maintainer/inkling2_mini-design.md). Config, binding and loading join infernix_model_loading.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
)
# Execution: the native operands and the forward composition.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
)
# Frontend: the prompt renderer and the output session (channels from the checkpoint's response_template).
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
)
