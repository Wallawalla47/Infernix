# Hyper-connection residual streams (docs/maintainer/qwen3_8-flash-next-design.md §2).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/hyper_connection.cu"
  "${CMAKE_CURRENT_LIST_DIR}/hyper_connection_mix_fused.cu")
