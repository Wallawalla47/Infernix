# End-to-end decode quality: a greedy generator (public Engine API shared with NInfer) and a
# judge that scores any number of builds' outputs with one lossless-route reference.
add_executable(infernix_decode_quality_gen "${CMAKE_CURRENT_LIST_DIR}/decode_quality_gen.cpp")
target_link_libraries(infernix_decode_quality_gen PRIVATE infernix_engine)

add_executable(infernix_decode_quality_judge "${CMAKE_CURRENT_LIST_DIR}/decode_quality_judge.cpp")
target_link_libraries(infernix_decode_quality_judge PRIVATE infernix_engine)

# A judged token's large regret: decode state or route arithmetic (forced decode from split points).
add_executable(infernix_decode_quality_probe "${CMAKE_CURRENT_LIST_DIR}/decode_quality_probe.cpp")
target_link_libraries(infernix_decode_quality_probe PRIVATE infernix_engine)
