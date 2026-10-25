# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(infernix_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(infernix_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
# It links the public Engine, as a consumer does, for the out-of-line definitions the options
# own (PrefixCacheSaveControl); infernix::engine exports only the include/ root.
target_link_libraries(infernix_public_api_test PRIVATE infernix::engine)
add_test(NAME infernix_public_api_test COMMAND infernix_public_api_test)

infernix_add_test(infernix_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES infernix_core)

# Asynchronous unbuffered reads for the SSD expert tier (CPU and disk only).
infernix_add_test(infernix_direct_read_queue_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_direct_read_queue.cpp"
  LIBRARIES infernix_core)

# Unbuffered block reads through a ring (CPU and disk only).
infernix_add_test(infernix_read_only_file_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_read_only_file.cpp"
  LIBRARIES infernix_core)

set_tests_properties(infernix_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "INFERNIX_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME infernix_device_sync_${mode}_test COMMAND infernix_device_test ${flags})
  set_tests_properties(infernix_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT "INFERNIX_CUDA_SYNC=${mode}" SKIP_RETURN_CODE 77)
endforeach()
foreach(mode IN ITEMS invalid empty)
  add_test(NAME infernix_device_sync_${mode}_test COMMAND infernix_device_test --invalid-sync)
endforeach()
set_tests_properties(infernix_device_sync_invalid_test PROPERTIES
  ENVIRONMENT "INFERNIX_CUDA_SYNC=invalid")
set_tests_properties(infernix_device_sync_empty_test PROPERTIES
  ENVIRONMENT "INFERNIX_CUDA_SYNC=")

infernix_add_test(infernix_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_decode_graph_gated_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph_gated.cu"
  LIBRARIES infernix_core)

infernix_add_test(infernix_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_layout_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_layout.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_copy_batch_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_copy_batch.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_block_copy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_block_copy.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_host_context_arena_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_context_arena.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES infernix_core)

set_tests_properties(
  infernix_device_test
  infernix_decode_graph_test
  infernix_decode_graph_gated_test
  infernix_arena_test
  infernix_copy_batch_test
  infernix_block_copy_test
  infernix_kv_cache_test
  infernix_host_context_arena_test
  infernix_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

infernix_add_test(infernix_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_host_load_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_load.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_machine_calibration_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_machine_calibration.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_cpu_topology_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cpu_topology.cpp"
  LIBRARIES infernix_core)

infernix_add_test(infernix_jinja_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_jinja.cpp"
  LIBRARIES infernix_jinja infernix::json)

add_test(NAME infernix_cuda_stack_tool_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/cuda_stack/test_check_cuda_stack.py)
# Register spills and local arrays: no kernel of the server may use more per-thread stack or local
# memory than tools/cuda_stack/cuda_stack_allowlist.json records.
if(TARGET infernix-serve)
  add_test(NAME infernix_cuda_stack_check
    COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tools/cuda_stack/check_cuda_stack.py check
            $<TARGET_FILE:infernix-serve> --cuobjdump "${CUDAToolkit_BIN_DIR}/cuobjdump${CMAKE_EXECUTABLE_SUFFIX}")
endif()

add_test(NAME infernix_chat_templates_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_chat_templates.py
          $<TARGET_FILE:infernix_jinja_test>)
