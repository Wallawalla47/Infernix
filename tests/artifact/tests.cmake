infernix_add_test(infernix_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES infernix_artifact)

infernix_add_test(infernix_artifact_materialization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp" "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp"
  LIBRARIES infernix_artifact)

if(NOT WIN32)
target_link_options(infernix_artifact_materialization_test PRIVATE
  "LINKER:--wrap=cudaMalloc"
  "LINKER:--wrap=cudaMallocHost"
  "LINKER:--wrap=cudaFree"
  "LINKER:--wrap=cudaFreeHost"
  "LINKER:--wrap=cudaEventCreateWithFlags"
  "LINKER:--wrap=cudaEventRecord"
  "LINKER:--wrap=cudaMemcpyAsync"
  "LINKER:--wrap=cudaStreamSynchronize")
endif()

add_test(NAME infernix_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:infernix_artifact_materialization_test>)

set_tests_properties(
  infernix_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  infernix_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Flash-Next expert banks and block-scaled FP8 written by Python and accepted by the C++ reader.
add_test(NAME infernix_artifact_flash_next_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/flash_next_interop.py"
    $<TARGET_FILE:infernix_artifact_reader_test>)
