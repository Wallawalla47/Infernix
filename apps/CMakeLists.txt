add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)
ninfer_stage_runtime_dlls(ninfer)

# Build id is stamped at BUILD time (see cmake/GenerateBuildId.cmake) into a
# generated header, so it always reflects the tree actually being compiled -- not
# a stale configure-time value. The header is only rewritten when its content
# changes, so main.cpp recompiles only when the id actually changes.
set(NINFER_BUILD_ID_HEADER "${CMAKE_BINARY_DIR}/generated/ninfer_build_id.h")
add_custom_target(ninfer_build_id_stamp ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${NINFER_BUILD_ID_HEADER}
          -P ${CMAKE_SOURCE_DIR}/cmake/GenerateBuildId.cmake
  COMMENT "Stamping NINFER build id from git (build-time)"
  VERBATIM)

add_executable(ninfer-serve serve/main.cpp)
add_dependencies(ninfer-serve ninfer_build_id_stamp)
ninfer_internal_includes(ninfer-serve)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib
  ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_logging)
ninfer_stage_runtime_dlls(ninfer-serve)

add_executable(ninfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(ninfer-perplexity)
target_include_directories(ninfer-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
ninfer_stage_runtime_dlls(ninfer-perplexity)
