infernix_add_test(infernix_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES infernix_media_decode)

infernix_add_test(infernix_prompt_input_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES infernix_product_prompt_input)

infernix_add_test(infernix_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES infernix_product_logging infernix_media_decode)

# Stable per-statistic console colouring (product/log_colour): family classification of the
# operational line prefixes and clause-aware colouring of the pretty stats format.
infernix_add_test(infernix_log_colour_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_log_colour.cpp")

infernix_add_test(infernix_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
  LIBRARIES infernix_core)

target_include_directories(infernix_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

infernix_add_test(infernix_cli_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp" ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES infernix_runtime_support infernix_product_logging)

target_include_directories(infernix_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)

infernix_add_test(infernix_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_decide_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decide_schema.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_serve_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_metrics_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_console_stats_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_console_stats.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

infernix_add_test(infernix_stop_control_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_stop_control.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_routes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_routes.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES infernix_serve)
