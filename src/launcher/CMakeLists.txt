# infernix-launcher's service: supervisor, router, proxy, library, downloads and telemetry. It runs
# infernix-serve as a child process and links nothing of the Engine or CUDA.

set(INFERNIX_LAUNCHER_UI ${PROJECT_SOURCE_DIR}/apps/launcher/assets/index.html.gz)
set(INFERNIX_LAUNCHER_CATALOG ${PROJECT_SOURCE_DIR}/apps/launcher/catalog.json)
set(INFERNIX_LAUNCHER_ASSETS ${CMAKE_BINARY_DIR}/generated/launcher_assets.cpp)
set(_launcher_asset_inputs ${PROJECT_SOURCE_DIR}/cmake/EmbedLauncherAssets.cmake ${INFERNIX_LAUNCHER_CATALOG})
if(EXISTS ${INFERNIX_LAUNCHER_UI})
  list(APPEND _launcher_asset_inputs ${INFERNIX_LAUNCHER_UI})
endif()
add_custom_command(
  OUTPUT ${INFERNIX_LAUNCHER_ASSETS}
  COMMAND ${CMAKE_COMMAND}
          -DUI=${INFERNIX_LAUNCHER_UI}
          -DCATALOG=${INFERNIX_LAUNCHER_CATALOG}
          -DOUT=${INFERNIX_LAUNCHER_ASSETS}
          -P ${PROJECT_SOURCE_DIR}/cmake/EmbedLauncherAssets.cmake
  DEPENDS ${_launcher_asset_inputs}
  COMMENT "Embedding the launcher's web UI and catalog"
  VERBATIM)

add_library(infernix_launcher STATIC
  args.cpp
  artifact_scan.cpp
  chats.cpp
  downloads.cpp
  event_log.cpp
  events.cpp
  hf_hub.cpp
  launch_script.cpp
  launcher.cpp
  load_advice.cpp
  log_buffer.cpp
  model_defaults.cpp
  process.cpp
  proxy.cpp
  router.cpp
  routing.cpp
  serve_help.cpp
  sha256.cpp
  store.cpp
  supervisor.cpp
  telemetry.cpp
  updates.cpp
  usage.cpp
  util.cpp
  ${INFERNIX_LAUNCHER_ASSETS})
infernix_internal_includes(infernix_launcher)
target_link_libraries(infernix_launcher
  PUBLIC infernix::json infernix::httplib Threads::Threads
  PRIVATE ${INFERNIX_CURL_TARGET})
if(WIN32)
  target_compile_definitions(infernix_launcher PUBLIC WIN32_LEAN_AND_MEAN NOMINMAX)
  target_link_libraries(infernix_launcher PRIVATE ws2_32 pdh bcrypt psapi shell32 ole32 advapi32 uuid)
else()
  target_link_libraries(infernix_launcher PRIVATE ${CMAKE_DL_LIBS})
endif()
