add_library(infernix_jinja STATIC
  jinja/json.cpp
  jinja/lexer.cpp
  jinja/parser.cpp
  jinja/runtime.cpp
  jinja/string.cpp
  jinja/value.cpp
  jinja/unicode.cpp
)
target_include_directories(infernix_jinja
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
