target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a8_linear.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k25600.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13824_k2304.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2304_k6912.cu"
)

target_sources(infernix_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a8_linear_ring.cu"
)
