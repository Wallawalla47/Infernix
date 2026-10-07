add_library(infernix_serve STATIC
  http_transport.cpp
  request_validation.cpp
  openai_chat_request.cpp
  openai_chat_response.cpp
  openai_chat_http.cpp
  decide.cpp
  decide_http.cpp
  openai_common.cpp
  openai_responses_request.cpp
  openai_responses_state.cpp
  openai_responses_response.cpp
  anthropic_messages_request.cpp
  anthropic_messages_response.cpp
  anthropic_messages_http.cpp
  openai_responses_store.cpp
  translate.cpp
  generation_service.cpp
  request_events.cpp
  console_stats.cpp
  operational_log.cpp
  request_log.cpp
  metrics.cpp
  http_server.cpp
  openai_responses_http.cpp
  serve_options.cpp
  stop_control.cpp
)
infernix_internal_includes(infernix_serve)
target_link_libraries(infernix_serve
  PUBLIC infernix_engine Threads::Threads infernix::json infernix::httplib
  PRIVATE infernix_media_acquire infernix_product_logging infernix_cuda_runtime)
