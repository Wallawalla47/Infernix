# The prebuilt Windows vcpkg FFmpeg tree ships without the png decoder, so a native PNG
# path (png_decode.cpp, gated on INFERNIX_MEDIA_NATIVE_PNG) decodes PNG directly via zlib
# from the same tree. Cross-platform builds use system FFmpeg (which includes png) and
# skip the native path. JPEG images decode through libjpeg(-turbo) on every platform
# (jpeg_decode.cpp), so their pixels equal Pillow's.
set(_infernix_media_sources decode/decode.cpp decode/jpeg_decode.cpp)
if(WIN32)
  list(APPEND _infernix_media_sources decode/png_decode.cpp)
endif()
add_library(infernix_media_decode STATIC
  ${_infernix_media_sources}
)
infernix_internal_includes(infernix_media_decode)
target_link_libraries(infernix_media_decode PRIVATE ${INFERNIX_FFMPEG_TARGET} JPEG::JPEG)
if(WIN32)
  add_library(infernix_zlib INTERFACE)
  set_target_properties(infernix_zlib PROPERTIES
    INTERFACE_LINK_LIBRARIES "${FFMPEG_VCPKG_TREE}/lib/z.lib")
  target_link_libraries(infernix_media_decode PRIVATE infernix_zlib)
  target_compile_definitions(infernix_media_decode PUBLIC INFERNIX_MEDIA_NATIVE_PNG)
endif()

# PDF documents (pdf/pdf.cpp) through PDFium when the configure found it (cmake/Dependencies.cmake); without it
# the same interface refuses every document.
add_library(infernix_media_pdf STATIC pdf/pdf.cpp)
infernix_internal_includes(infernix_media_pdf)
if(INFERNIX_HAVE_PDFIUM)
  target_link_libraries(infernix_media_pdf PRIVATE pdfium)
  target_compile_definitions(infernix_media_pdf PRIVATE INFERNIX_HAVE_PDFIUM)
endif()
