infernix_add_test(infernix_scheduler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_scheduler.cpp"
  LIBRARIES infernix_runtime_support)

infernix_add_test(infernix_context_cost_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_context_cost.cpp"
  LIBRARIES infernix_runtime_support infernix::json)

infernix_add_test(infernix_resource_manager_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_resource_manager.cpp"
  LIBRARIES infernix_runtime_support)

infernix_add_test(infernix_prefix_cache_index_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_prefix_cache_index.cpp"
  LIBRARIES infernix_runtime_support)

infernix_add_test(infernix_hybrid_cache_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_hybrid_cache_defaults.cpp"
  LIBRARIES infernix_engine infernix_core infernix::json)

infernix_add_test(infernix_kv_capacity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_kv_capacity.cpp"
  LIBRARIES infernix_runtime_support)

infernix_add_test(infernix_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_sampling_defaults.cpp"
  LIBRARIES infernix_engine infernix_core)

infernix_add_test(infernix_engine_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_engine_options.cpp"
  LIBRARIES infernix_engine infernix_core)

set_tests_properties(infernix_engine_options_test PROPERTIES SKIP_RETURN_CODE 77)
