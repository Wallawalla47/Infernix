target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/f16acc_encode.cu"
)

# The split-register launches keep setmaxnreg only without RDC.
target_sources(infernix_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/f16acc_split_registers.cu"
)
