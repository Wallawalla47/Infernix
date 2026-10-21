target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/mxfp8_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mxfp8_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13952_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1280_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k640.cu"
)
