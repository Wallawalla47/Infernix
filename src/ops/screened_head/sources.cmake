# Screened vocabulary heads: a 2-bit screen of every row, the near-best rows rescored exactly.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/screened_head.cu"
  "${CMAKE_CURRENT_LIST_DIR}/screened_head_bf16.cu")
