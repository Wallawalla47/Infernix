add_library(infernix_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
infernix_internal_includes(infernix_text)
target_include_directories(infernix_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(infernix_text PUBLIC UTF8PROC_STATIC)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(infernix_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
infernix_internal_includes(infernix_jinja)
target_include_directories(infernix_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(infernix_jinja PRIVATE infernix_text infernix::json)
