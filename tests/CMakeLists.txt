find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/InfernixTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(models/qwen4_exp/tests.cmake)
include(ops/tests.cmake)
# Machine probes of the Qwen3.8-Flash-Next memory track, built beside the tests.
include(${PROJECT_SOURCE_DIR}/tools/flash_next_probe/probes.cmake)

# Suite-wide scheduling rules, applied after every registration so new entries follow them:
# - an entry that loads a real artifact (name contains "_real") holds most of the device and its
#   pinned host memory, so it runs alone (RUN_SERIAL), labelled "gpu;real" (`ctest -LE real`
#   runs the rest) and skips (77) without its artifact;
# - every entry has a timeout, so a hung wait fails its entry instead of stalling the run.
get_property(infernix_registered_tests DIRECTORY PROPERTY TESTS)
foreach(test_name IN LISTS infernix_registered_tests)
  if(test_name MATCHES "_real")
    set_tests_properties(${test_name} PROPERTIES
      RUN_SERIAL TRUE LABELS "gpu;real" SKIP_RETURN_CODE 77)
  endif()
  get_test_property(${test_name} TIMEOUT test_timeout)
  if(NOT test_timeout)
    if(test_name MATCHES "_real")
      set_tests_properties(${test_name} PROPERTIES TIMEOUT 3600)
    else()
      set_tests_properties(${test_name} PROPERTIES TIMEOUT 900)
    endif()
  endif()
endforeach()

# All test executables share one per-config runtime directory; stage the shared
# FFMPEG/curl DLL closure there once so fully-linked tests resolve their imports
# (the Windows loader only searches the exe's own dir, system dirs and PATH).
infernix_stage_test_runtime_dlls(infernix_device_test)
