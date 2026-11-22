# Inkling 2's visit chain (include/infernix/ops/inkling2_visit.h): embed, the mixers, message gates, short convs and the
# experts' activation rows of the decode and verify route.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/visit.cu" "${CMAKE_CURRENT_LIST_DIR}/attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/projections.cu" "${CMAKE_CURRENT_LIST_DIR}/projections_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/attention_prompt.cu" "${CMAKE_CURRENT_LIST_DIR}/dspark.cu"
  "${CMAKE_CURRENT_LIST_DIR}/vision.cu")
