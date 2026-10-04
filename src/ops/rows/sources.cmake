# Row splits and column gathers of BF16 activations.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/rows.cu")
