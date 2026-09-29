# GenerateBuildId.cmake -- write the NInfer build id to a C header at BUILD time.
#
# Run as:  cmake -DSRC_DIR=<src> -DOUT=<out header> -P cmake/GenerateBuildId.cmake
#
# The id is $NINFER_BUILD_ID if that env var is set (the one-off restamp
# build_*.bat scripts use it), otherwise `git describe --always --dirty --tags`
# from the source tree. Generating it at build time -- instead of configure time,
# as apps/CMakeLists.txt did -- means the baked id always reflects the tree that is
# actually being compiled, not a stale value from the last reconfigure. The header
# is only rewritten when its content changes, so main.cpp recompiles only when the
# id actually changes (no needless churn on every build).

if(DEFINED ENV{NINFER_BUILD_ID})
  set(_id "$ENV{NINFER_BUILD_ID}")
else()
  find_program(_git_executable git)
  if(_git_executable)
    execute_process(
      COMMAND "${_git_executable}" describe --always --dirty --tags
      WORKING_DIRECTORY "${SRC_DIR}"
      OUTPUT_VARIABLE _id
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  if(NOT _id)
    set(_id "unknown")
  endif()
endif()

string(REPLACE "\"" "\\\"" _id_escaped "${_id}")
# Single string (not a list) so file(WRITE) does not inject a ';' list separator.
set(_content "// Auto-generated at build time by cmake/GenerateBuildId.cmake -- do not edit.\n#define NINFER_BUILD_ID \"${_id_escaped}\"\n")

if(EXISTS "${OUT}")
  file(READ "${OUT}" _existing)
  if(_existing STREQUAL _content)
    return()
  endif()
endif()

get_filename_component(_out_dir "${OUT}" DIRECTORY)
file(MAKE_DIRECTORY "${_out_dir}")
file(WRITE "${OUT}" "${_content}")
message(STATUS "NINFER build id: ${_id}")
