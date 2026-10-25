# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under build/tests.
function(infernix_test_includes target)
  infernix_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

# ARGS are passed to the executable by the registered test of the same name.
function(infernix_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES;ARGS")
  add_executable(${name} ${arg_SOURCES})
  infernix_test_includes(${name})
  target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      INFERNIX_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      INFERNIX_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name} ${arg_ARGS})
endfunction()

# Apply these to the translation unit containing the oracle, including shared
# test support libraries. Executable options do not propagate into those libraries.
function(infernix_op_oracle_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(infernix_add_op_test name)
  infernix_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  infernix_op_oracle_options(${name})
endfunction()

# The op tests under tests/ops link into one executable per family, infernix_ops_<family>_test
# (tests/ops/op_family.h), instead of one each: a test's sources and libraries join its family, its
# CTest entry runs `infernix_ops_<family>_test <name>` (name: the stem of infernix_<name>_test), and
# ${<test>_COMMAND} holds that command for further entries of the same test. NO_TEST adds the test to
# its family without an entry of its own. infernix_finish_op_families() creates the executables.
function(infernix_add_family_op_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NO_TEST" "FAMILY" "SOURCES;LIBRARIES;ARGS")
  if(NOT arg_FAMILY)
    message(FATAL_ERROR "${name}: an op test needs its FAMILY")
  endif()
  string(REGEX REPLACE "^infernix_(.*)_test$" "\\1" member "${name}")
  set(target infernix_ops_${arg_FAMILY}_test)
  set_property(GLOBAL APPEND PROPERTY INFERNIX_OP_FAMILIES ${arg_FAMILY})
  set_property(GLOBAL APPEND PROPERTY INFERNIX_OP_FAMILY_${arg_FAMILY}_SOURCES ${arg_SOURCES})
  set_property(GLOBAL APPEND PROPERTY INFERNIX_OP_FAMILY_${arg_FAMILY}_LIBRARIES ${arg_LIBRARIES})
  if(NOT arg_NO_TEST)
    add_test(NAME ${name} COMMAND ${target} ${member} ${arg_ARGS})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  endif()
  set(${name}_COMMAND ${target} ${member} PARENT_SCOPE)
endfunction()

function(infernix_finish_op_families)
  get_property(families GLOBAL PROPERTY INFERNIX_OP_FAMILIES)
  list(REMOVE_DUPLICATES families)
  foreach(family IN LISTS families)
    get_property(sources GLOBAL PROPERTY INFERNIX_OP_FAMILY_${family}_SOURCES)
    get_property(libraries GLOBAL PROPERTY INFERNIX_OP_FAMILY_${family}_LIBRARIES)
    list(REMOVE_DUPLICATES sources)
    list(REMOVE_DUPLICATES libraries)
    set(target infernix_ops_${family}_test)
    add_executable(${target} "${PROJECT_SOURCE_DIR}/tests/ops/op_family.cpp" ${sources})
    infernix_test_includes(${target})
    target_link_libraries(${target} PRIVATE ${libraries})
    infernix_op_oracle_options(${target})
  endforeach()
endfunction()
