# Inkling2Loop (Thinking Machines Inkling 2 Mini): its own model architecture
# (docs/maintainer/inkling2_mini-design.md). Config, binding and loading join infernix_model_loading.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
)
