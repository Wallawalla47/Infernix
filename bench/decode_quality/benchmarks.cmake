# End-to-end decode quality: a greedy generator (public Engine API shared with upstream) and a
# judge that scores any number of builds' outputs with one 16-bit-activation reference.
add_executable(infernix_decode_quality_gen "${CMAKE_CURRENT_LIST_DIR}/decode_quality_gen.cpp")
target_link_libraries(infernix_decode_quality_gen PRIVATE infernix_engine)

add_executable(infernix_decode_quality_judge "${CMAKE_CURRENT_LIST_DIR}/decode_quality_judge.cpp")
target_link_libraries(infernix_decode_quality_judge PRIVATE infernix_engine)
