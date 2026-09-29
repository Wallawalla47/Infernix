find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(ops/tests.cmake)

# All test executables share one per-config runtime directory; stage the shared
# FFMPEG/curl DLL closure there once so fully-linked tests resolve their imports
# (the Windows loader only searches the exe's own dir, system dirs and PATH).
ninfer_stage_test_runtime_dlls(ninfer_device_test)
