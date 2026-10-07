add_library(infernix_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
  materializer.cpp
  views.cpp
)
infernix_internal_includes(infernix_artifact)
target_link_libraries(infernix_artifact PUBLIC infernix_core infernix::json)
