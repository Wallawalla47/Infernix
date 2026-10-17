# GGUF (ggml) block formats: exact projections over stored GGUF blocks, the token embedding and the
# n-gram rows (docs/maintainer/qwen3_8-flash-next-iq3s-plan.md §6-9).
target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/ggml_linear.cu"
  "${CMAKE_CURRENT_LIST_DIR}/ggml_rows.cu")
