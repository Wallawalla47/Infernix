add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(ninfer)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input)
add_dependencies(ninfer ninfer_build_id_stamp)
target_sources(ninfer PRIVATE ${NINFER_BUILD_ID_HEADER})
target_include_directories(ninfer PRIVATE ${CMAKE_BINARY_DIR}/generated)
ninfer_stage_runtime_dlls(ninfer)

# Build id is stamped at BUILD time (see cmake/GenerateBuildId.cmake) into a
# generated header, so it always reflects the tree actually being compiled -- not
# a stale configure-time value. Every product binary records it in its startup
# log ("build <id>"); stamping one target stamps the shared header all of them
# use. The header is only rewritten when its content changes, so each main.cpp
# recompiles only when the id actually changes.
set(NINFER_BUILD_ID_HEADER "${CMAKE_BINARY_DIR}/generated/ninfer_build_id.h")
add_custom_target(ninfer_build_id_stamp ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${NINFER_BUILD_ID_HEADER}
          -P ${CMAKE_SOURCE_DIR}/cmake/GenerateBuildId.cmake
  BYPRODUCTS ${NINFER_BUILD_ID_HEADER}
  COMMENT "Stamping NINFER build id from git (build-time)"
  VERBATIM)

add_executable(ninfer-serve serve/main.cpp)
add_dependencies(ninfer-serve ninfer_build_id_stamp)
# List the generated header as a source so MSBuild treats it as a real input of this target and
# recompiles main.cpp whenever the stamped id changes (the add_dependencies edge alone orders the
# projects but does not make the header a tracked input).
target_sources(ninfer-serve PRIVATE ${NINFER_BUILD_ID_HEADER})
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
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity
  ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(ninfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer::json)
add_dependencies(ninfer-perplexity ninfer_build_id_stamp)
target_sources(ninfer-perplexity PRIVATE ${NINFER_BUILD_ID_HEADER})
ninfer_stage_runtime_dlls(ninfer-perplexity)
