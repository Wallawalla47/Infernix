# GGUF (ggml) block formats: exact projections over stored GGUF blocks, their 8-bit activation routes
# (--a8-activations: narrow and wide calls), the token embedding and the n-gram rows
# (docs/maintainer/qwen3_8-flash-next-design.md §16.2.2, §12.5, §19.3.25, §19.3.31).
target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ggml_encode_producers.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear_a8_rows.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear_a8_tiled.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear_a8_wide.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_rows.cu")
