# Build id is stamped at BUILD time (see cmake/GenerateBuildId.cmake) into a
# generated header, so it always reflects the tree actually being compiled -- not
# a stale configure-time value. Every product binary records it in its startup
# log ("build <id>"); stamping one target stamps the shared header all of them
# use. The header is only rewritten when its content changes, so each main.cpp
# recompiles only when the id actually changes.
set(INFERNIX_BUILD_ID_HEADER "${CMAKE_BINARY_DIR}/generated/infernix_build_id.h")
add_custom_target(infernix_build_id_stamp ALL
  COMMAND ${CMAKE_COMMAND}
          -DSRC_DIR=${CMAKE_SOURCE_DIR}
          -DOUT=${INFERNIX_BUILD_ID_HEADER}
          -P ${CMAKE_SOURCE_DIR}/cmake/GenerateBuildId.cmake
  BYPRODUCTS ${INFERNIX_BUILD_ID_HEADER}
  COMMENT "Stamping INFERNIX build id from git (build-time)"
  VERBATIM)

add_executable(infernix
  cli/main.cpp
  cli/options.cpp)
infernix_internal_includes(infernix)
target_link_libraries(infernix PRIVATE
  infernix_engine
  infernix_product_logging
  infernix_product_prompt_input)
add_dependencies(infernix infernix_build_id_stamp)
target_sources(infernix PRIVATE ${INFERNIX_BUILD_ID_HEADER})
target_include_directories(infernix PRIVATE ${CMAKE_BINARY_DIR}/generated)
infernix_stage_runtime_dlls(infernix)

add_executable(infernix-serve serve/main.cpp)
add_dependencies(infernix-serve infernix_build_id_stamp)
# List the generated header as a source so MSBuild treats it as a real input of this target and
# recompiles main.cpp whenever the stamped id changes (the add_dependencies edge alone orders the
# projects but does not make the header a tracked input).
target_sources(infernix-serve PRIVATE ${INFERNIX_BUILD_ID_HEADER})
infernix_internal_includes(infernix-serve)
target_include_directories(infernix-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib
  ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(infernix-serve PRIVATE infernix_serve infernix_product_logging)
infernix_stage_runtime_dlls(infernix-serve)

add_executable(infernix-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp
  perplexity/distribution.cpp)
infernix_internal_includes(infernix-perplexity)
target_include_directories(infernix-perplexity PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/perplexity
  ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(infernix-perplexity PRIVATE
  infernix_engine infernix_product_logging infernix::json)
add_dependencies(infernix-perplexity infernix_build_id_stamp)
target_sources(infernix-perplexity PRIVATE ${INFERNIX_BUILD_ID_HEADER})
infernix_stage_runtime_dlls(infernix-perplexity)
