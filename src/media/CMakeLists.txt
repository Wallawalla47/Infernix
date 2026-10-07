# The prebuilt Windows vcpkg FFmpeg tree ships without the png decoder, so a native PNG
# path (png_decode.cpp, gated on INFERNIX_MEDIA_NATIVE_PNG) decodes PNG directly via zlib
# from the same tree. Cross-platform builds use system FFmpeg (which includes png) and
# skip the native path.
set(_infernix_media_sources decode/decode.cpp)
if(WIN32)
  list(APPEND _infernix_media_sources decode/png_decode.cpp)
endif()
add_library(infernix_media_decode STATIC
  ${_infernix_media_sources}
)
infernix_internal_includes(infernix_media_decode)
target_link_libraries(infernix_media_decode PRIVATE ${INFERNIX_FFMPEG_TARGET})
if(WIN32)
  add_library(infernix_zlib INTERFACE)
  set_target_properties(infernix_zlib PROPERTIES
    INTERFACE_LINK_LIBRARIES "${FFMPEG_VCPKG_TREE}/lib/z.lib")
  target_link_libraries(infernix_media_decode PRIVATE infernix_zlib)
  target_compile_definitions(infernix_media_decode PUBLIC INFERNIX_MEDIA_NATIVE_PNG)
endif()
