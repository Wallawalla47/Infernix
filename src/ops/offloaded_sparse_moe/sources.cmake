# Host side of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md §10, §16.2).
# A plain C++ archive: the CPU expert engine needs no CUDA, and its canonical arithmetic must be
# compiled without floating-point contraction or fast-math on every compiler.
add_library(ninfer_offloaded_moe_cpu STATIC
  "${CMAKE_CURRENT_LIST_DIR}/cpu/w4a4_expert.cpp")
ninfer_internal_includes(ninfer_offloaded_moe_cpu)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ninfer_offloaded_moe_cpu PRIVATE -ffp-contract=off -fno-fast-math)
elseif(MSVC)
  target_compile_options(ninfer_offloaded_moe_cpu PRIVATE /fp:precise)
endif()
target_link_libraries(ninfer_ops PUBLIC ninfer_offloaded_moe_cpu)

# GPU narrow route: the same canonical arithmetic, compiled for the device.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cuda/narrow_expert.cu")
