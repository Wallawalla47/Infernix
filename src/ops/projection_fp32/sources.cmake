# FP32 projections of BF16 activations (router and LM-head logits).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/projection_fp32.cu")
