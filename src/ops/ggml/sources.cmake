# GGUF (ggml) block formats: exact projections over stored GGUF blocks, the token embedding and the
# n-gram rows (docs/maintainer/qwen3_8-flash-next-design.md §16.2.2, §12.5, §19.3.25).
target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_rows.cu")
