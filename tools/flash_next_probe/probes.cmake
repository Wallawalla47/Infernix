# Machine probes of the memory track (design §19.3.7, measurements RM0a, the read half of RM0c,
# RM0e and RM0g). They measure the GPU, the driver and the artifact's drive and are run by hand, so
# they are built beside the tests but are not CTest tests.

add_executable(ninfer_vram_probe
  "${CMAKE_CURRENT_LIST_DIR}/vram_probe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vram_probe_kernels.cu")
# nvml.h only: NVML itself is loaded at run time, so a machine without it still runs the probe.
target_include_directories(ninfer_vram_probe PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(ninfer_vram_probe PRIVATE ninfer_cuda_runtime CUDA::cuda_driver Threads::Threads)
if(WIN32)
  target_link_libraries(ninfer_vram_probe PRIVATE dxgi gdi32 psapi)
else()
  target_link_libraries(ninfer_vram_probe PRIVATE ${CMAKE_DL_LIBS})
endif()

add_executable(ninfer_expert_read_probe "${CMAKE_CURRENT_LIST_DIR}/expert_read_probe.cpp")
ninfer_internal_includes(ninfer_expert_read_probe)
target_link_libraries(ninfer_expert_read_probe PRIVATE ninfer_artifact ninfer_cuda_runtime Threads::Threads)
