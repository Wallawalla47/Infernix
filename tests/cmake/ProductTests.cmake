infernix_add_test(infernix_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES infernix_media_decode)

infernix_add_test(infernix_prompt_input_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES infernix_product_prompt_input)

infernix_add_test(infernix_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES infernix_product_logging infernix_media_decode infernix_core)

# The abort path: the --log-file already holds the CUDA_CHECK record when the process dies.
if(TARGET infernix_pretty_logging_test)
  add_test(NAME infernix_cuda_failure_log_test
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_cuda_failure_log.py"
            $<TARGET_FILE:infernix_pretty_logging_test>)
  set_tests_properties(infernix_cuda_failure_log_test PROPERTIES TIMEOUT 60 LABELS "logging")
endif()

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

infernix_add_test(infernix_llama_kld_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_llama_kld.cpp"
          ${PROJECT_SOURCE_DIR}/apps/certify/llama_kld.cpp
  LIBRARIES infernix_core infernix::json)

target_include_directories(infernix_llama_kld_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/certify)

infernix_add_test(infernix_cli_options_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp" ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES infernix_runtime_support infernix_product_config infernix_product_logging)

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

infernix_add_test(infernix_tool_search_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tool_search.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_serve_options_test NEEDS_SOURCE_DIR
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

infernix_add_test(infernix_metrics_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_metrics_contract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_metrics_contract.cpp"
  LIBRARIES infernix_serve)

# infernix-serve --plan's JSON (docs/serving.md, "Sizing a start without loading").
infernix_add_test(infernix_plan_report_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_plan_report.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_prefix_miss_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prefix_miss.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_reasoning_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_reasoning_store.cpp"
  LIBRARIES infernix_serve)
infernix_add_test(infernix_console_stats_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_console_stats.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

infernix_add_test(infernix_launcher_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_launcher.cpp"
  LIBRARIES infernix_launcher)

# The launcher's Configure form has a control for every infernix-serve option (it reads the
# server's option table and the form's option list; nothing is built or run).
add_test(NAME infernix_launcher_option_coverage_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_launcher_option_coverage.py")
set_tests_properties(infernix_launcher_option_coverage_test PROPERTIES TIMEOUT 60 LABELS "launcher")

# A stand-in infernix-serve for the launcher's watchdog and web UI checks (docs/launcher.md).
add_executable(infernix_launcher_fake_serve "${CMAKE_CURRENT_LIST_DIR}/../launcher_fake_serve.cpp")
target_link_libraries(infernix_launcher_fake_serve PRIVATE infernix::json infernix::httplib Threads::Threads)
if(WIN32)
  target_compile_definitions(infernix_launcher_fake_serve PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_link_libraries(infernix_launcher_fake_serve PRIVATE ws2_32)
endif()

infernix_add_test(infernix_stop_control_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_stop_control.cpp"
  LIBRARIES infernix_serve)
# The whole server stopped (Ctrl+Break, SIGTERM) while a client streams: the stream ends with an error event,
# the prefix cache file is saved and the process exits 0 (INFERNIX_TEST_ARTIFACT: any Qwen3.5 artifact).
if(TARGET infernix-serve)
  add_test(NAME infernix_serve_shutdown_real_test
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_serve_shutdown_real.py"
            $<TARGET_FILE:infernix-serve>)
  set_tests_properties(infernix_serve_shutdown_real_test PROPERTIES
    SKIP_RETURN_CODE 77 RUN_SERIAL TRUE TIMEOUT 900 LABELS "gpu;real")
endif()

infernix_add_test(infernix_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES infernix_serve)

infernix_add_test(infernix_http_routes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_routes.cpp"
  LIBRARIES infernix_serve)

# Streaming requests refused before their stream starts (queue answers, a context overflow), over
# loopback HTTP against a refusing generation backend.
infernix_add_test(infernix_http_stream_refusals_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_stream_refusals.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

# --claude-code-usage-compat on a streamed Messages answer, over loopback HTTP through the real route:
# the option reaches the streaming encoder, and only message_start's input_tokens differs from the
# same server with the flag off.
infernix_add_test(infernix_http_anthropic_usage_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_anthropic_usage_compat.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

# A streamed reply whose only token is an incomplete UTF-8 character, over loopback HTTP on every
# streaming API: its U+FFFD (or no text) and the output limit's terminal events, and the stream ends.
infernix_add_test(infernix_http_stream_lone_token_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_stream_lone_token.cpp"
  LIBRARIES infernix_serve infernix_product_logging)

# YH-01: a client that stops reading a stream for 7 s (past cpp-httplib's 5 s default write timeout)
# keeps it: every delta, the finish and [DONE] arrive once it reads again (one 7 s pause).
infernix_add_test(infernix_http_stream_stall_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_stream_stall.cpp"
  LIBRARIES infernix_serve infernix_product_logging)
set_tests_properties(infernix_http_stream_stall_test PROPERTIES TIMEOUT 60)

infernix_add_test(infernix_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES infernix_serve)
