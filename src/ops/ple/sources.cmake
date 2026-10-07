# Per-layer n-gram embedding injection (docs/maintainer/qwen3_8-flash-next-design.md §12).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/ple.cu")
