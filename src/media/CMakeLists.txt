# The prebuilt Windows vcpkg FFmpeg tree ships without the png decoder, so a native PNG
# path (png_decode.cpp, gated on NINFER_MEDIA_NATIVE_PNG) decodes PNG directly via zlib
# from the same tree. Cross-platform builds use system FFmpeg (which includes png) and
# skip the native path.
set(_ninfer_media_sources decode/decode.cpp)
if(WIN32)
  list(APPEND _ninfer_media_sources decode/png_decode.cpp)
endif()
add_library(ninfer_media_decode STATIC
  ${_ninfer_media_sources}
)
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE ${NINFER_FFMPEG_TARGET})
if(WIN32)
  add_library(ninfer_zlib INTERFACE)
  set_target_properties(ninfer_zlib PROPERTIES
    INTERFACE_LINK_LIBRARIES "${FFMPEG_VCPKG_TREE}/lib/z.lib")
  target_link_libraries(ninfer_media_decode PRIVATE ninfer_zlib)
  target_compile_definitions(ninfer_media_decode PUBLIC NINFER_MEDIA_NATIVE_PNG)
endif()
