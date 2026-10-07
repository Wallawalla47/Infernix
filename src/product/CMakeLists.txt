add_library(infernix_media_acquire STATIC
  media_acquire/acquire.cpp
)
infernix_internal_includes(infernix_media_acquire)
target_link_libraries(infernix_media_acquire PRIVATE ${INFERNIX_CURL_TARGET})
if(WIN32)
  target_link_libraries(infernix_media_acquire PRIVATE ws2_32)
endif()

add_library(infernix_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
infernix_internal_includes(infernix_product_prompt_input)
target_link_libraries(infernix_product_prompt_input PRIVATE infernix_media_acquire infernix::json)

add_library(infernix_product_logging STATIC
  logging/logging.cpp
  logging/pretty_format.cpp
  logging/engine_diagnostics.cpp
  logging/startup_log.cpp
)
infernix_internal_includes(infernix_product_logging)
target_link_libraries(infernix_product_logging PUBLIC spdlog::spdlog PRIVATE infernix_media_decode)
