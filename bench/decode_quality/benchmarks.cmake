# End-to-end decode quality: a greedy generator (public Engine API shared with NInfer) and a
# judge that scores any number of builds' outputs with one lossless-route reference.
add_executable(infernix_decode_quality_gen "${CMAKE_CURRENT_LIST_DIR}/decode_quality_gen.cpp")
target_link_libraries(infernix_decode_quality_gen PRIVATE infernix_engine)

add_executable(infernix_decode_quality_judge "${CMAKE_CURRENT_LIST_DIR}/decode_quality_judge.cpp")
target_link_libraries(infernix_decode_quality_judge PRIVATE infernix_engine)

# Teacher-forced decode: the decode route's log-probability of the corpus's own tokens, paired
# position by position between builds (the fast-path acceptance gate's decode measure).
add_executable(infernix_decode_quality_forced "${CMAKE_CURRENT_LIST_DIR}/decode_quality_forced.cpp")
target_link_libraries(infernix_decode_quality_forced PRIVATE infernix_engine infernix::json)
