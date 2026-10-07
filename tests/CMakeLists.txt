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

# All test executables share one per-config runtime directory; stage the shared
# FFMPEG/curl DLL closure there once so fully-linked tests resolve their imports
# (the Windows loader only searches the exe's own dir, system dirs and PATH).
infernix_stage_test_runtime_dlls(infernix_device_test)
