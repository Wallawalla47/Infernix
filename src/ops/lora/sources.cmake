target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/lora_shrink.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_lora.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a16_mma_lora_wide.cu"
)
