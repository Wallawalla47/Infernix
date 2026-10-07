# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(infernix_bench
  "${CMAKE_CURRENT_LIST_DIR}/infernix_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/infernix_bench_support.cpp")
infernix_internal_includes(infernix_bench)
target_include_directories(infernix_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(infernix_bench PRIVATE INFERNIX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(infernix_bench PRIVATE infernix_engine CUDA::cudart)
