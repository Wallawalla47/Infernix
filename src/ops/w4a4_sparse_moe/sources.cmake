# Inkling 2's routed experts (four-over-six NVFP4 and MXFP8, device-resident; include/infernix/ops/w4a4_sparse_moe.h).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/w4a4_sparse_moe.cu" "${CMAKE_CURRENT_LIST_DIR}/wide.cu")
