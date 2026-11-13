# Machine probes of the memory track (design §19.3.7, measurements RM0a, the read half of RM0c,
# RM0e and RM0g) and M0's host probe. They measure the GPU, the driver, the CPU and the artifact's
# drive and are run by hand, so they are built beside the tests but are not CTest tests.

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

# The M0 host probe (DRAM bandwidth, the CPU expert kernels' service rates; run_m0.sh) on the CPU expert engine's
# own library, so it builds whatever that library comes to need. Linux only: anonymous mmap with huge pages.
if(NOT WIN32)
  add_executable(infernix_flash_next_host_probe "${CMAKE_CURRENT_LIST_DIR}/host_probe.cpp")
  infernix_internal_includes(infernix_flash_next_host_probe)
  target_compile_options(infernix_flash_next_host_probe PRIVATE -ffp-contract=off -fno-fast-math)
  target_link_libraries(infernix_flash_next_host_probe PRIVATE infernix_offloaded_moe_cpu Threads::Threads)
endif()
