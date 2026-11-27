# Inkling2Loop (Thinking Machines Inkling 2 Mini): its own model architecture
# (docs/maintainer/inkling2_mini-design.md). Config, binding and loading join infernix_model_loading.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/draft_head.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/lora.cpp"
)
# Execution: the native operands and the forward composition.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/lora_parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/lora_forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/dspark.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/dflash2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/lossless.cpp"
)
# Frontend: the prompt renderer, the output session (channels from the checkpoint's response_template) and the
# constrained route's grammars.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/constraints.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/image.cpp"
)
# Program: lanes, state, the Forward's calls, sampling and commit behind the common EngineCore.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix_cache.cpp"
)
