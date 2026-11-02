target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_p12_pack.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_p12_linear.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_p12_n13952_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_p12_n16384_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16_p12_n2560_k6144.cu"
)
