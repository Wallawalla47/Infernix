# The reference route's projections: exactly decoded weights, exact binary32 products summed in
# double-binary32, one rounding
# (docs/maintainer/model-quality.md §6.2).
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/precise_linear.cu")
