# Machine probes of the memory track (design §19.3.7, measurements RM0a, the read half of RM0c,
# RM0e and RM0g). They measure the GPU, the driver and the artifact's drive and are run by hand, so
# they are built beside the tests but are not CTest tests.

add_executable(infernix_vram_probe
  "${CMAKE_CURRENT_LIST_DIR}/vram_probe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vram_probe_kernels.cu")
# nvml.h only: NVML itself is loaded at run time, so a machine without it still runs the probe.
target_include_directories(infernix_vram_probe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(infernix_vram_probe PRIVATE infernix_cuda_runtime CUDA::cuda_driver Threads::Threads)
if(WIN32)
  target_link_libraries(infernix_vram_probe PRIVATE dxgi gdi32 psapi)
else()
  target_link_libraries(infernix_vram_probe PRIVATE ${CMAKE_DL_LIBS})
endif()

add_executable(infernix_expert_read_probe "${CMAKE_CURRENT_LIST_DIR}/expert_read_probe.cpp")
infernix_internal_includes(infernix_expert_read_probe)
target_link_libraries(infernix_expert_read_probe PRIVATE infernix_artifact infernix_cuda_runtime Threads::Threads)
