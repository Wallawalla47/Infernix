# Host side of offloaded_sparse_moe (docs/maintainer/qwen3_8-flash-next-design.md §10, §16.2).
# A plain C++ archive: the CPU expert engine needs no CUDA, and its canonical arithmetic must be
# compiled without floating-point contraction or fast-math on every compiler.
add_library(infernix_offloaded_moe_cpu STATIC
  "${CMAKE_CURRENT_LIST_DIR}/cpu/w4a4_expert.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/cpu/expert_team.cpp")
infernix_internal_includes(infernix_offloaded_moe_cpu)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(infernix_offloaded_moe_cpu PRIVATE -ffp-contract=off -fno-fast-math)
elseif(MSVC)
  target_compile_options(infernix_offloaded_moe_cpu PRIVATE /fp:precise)
endif()
target_link_libraries(infernix_offloaded_moe_cpu PUBLIC Threads::Threads)
target_link_libraries(infernix_ops PUBLIC infernix_offloaded_moe_cpu)

# GPU narrow route: the same canonical arithmetic, compiled for the device.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cuda/narrow_expert.cu")
# A whole MoE layer: routing, device-side dispatch, the exact expert kernels and the combine.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cuda/moe_layer.cu")
# The wide route: experts with more than eight columns on block-scaled tensor cores.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cuda/wide_expert.cu")
# The host side of CPU-served misses: mapped channel buffers and the service thread.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cpu/miss_service.cpp")
# The host side of the SSD tier's fetch channel: mapped request/response words and the responder.
target_sources(infernix_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/cpu/fetch_channel.cpp")
