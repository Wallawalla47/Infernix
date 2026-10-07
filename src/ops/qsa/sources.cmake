# QSA pooled-key indexer, block selection and block-sparse attention
# (docs/maintainer/qwen3_8-flash-next-design.md §8.10, §19.3.6 item 2).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/qsa.cu" "${CMAKE_CURRENT_LIST_DIR}/qsa_select.cu"
                   "${CMAKE_CURRENT_LIST_DIR}/qsa_prompt.cu")
