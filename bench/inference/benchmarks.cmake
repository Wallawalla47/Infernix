# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(infernix_bench
  "${CMAKE_CURRENT_LIST_DIR}/infernix_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/infernix_bench_support.cpp")
infernix_internal_includes(infernix_bench)
target_include_directories(infernix_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(infernix_bench PRIVATE INFERNIX_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(infernix_bench PRIVATE infernix_engine infernix_core infernix::json CUDA::cudart)

# A model's token-frequency ranking (tools/freq_corpus), through the public Engine.
add_executable(infernix_token_ranking "${CMAKE_CURRENT_LIST_DIR}/infernix_token_ranking.cpp")
infernix_internal_includes(infernix_token_ranking)
target_link_libraries(infernix_token_ranking PRIVATE infernix_engine infernix_core infernix::json CUDA::cudart)
