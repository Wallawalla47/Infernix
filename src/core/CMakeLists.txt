if(WIN32)
  set(NINFER_READ_ONLY_FILE_SOURCE read_only_file_win32.cpp)
else()
  set(NINFER_READ_ONLY_FILE_SOURCE read_only_file_posix.cpp)
endif()

add_library(ninfer_core STATIC
  ${NINFER_READ_ONLY_FILE_SOURCE}
  arena.cu
  evictable_weight_pool.cu
  vmm_arena.cu
  device.cu
  dtype.cpp
  host_memory.cpp
  host_worker_pool.cpp
  link_probe.cpp
  vram_budget.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cuda_runtime CUDA::nvtx3 CUDA::cuda_driver Threads::Threads)
if(WIN32)
  # DXGI: the OS video-memory budget and display outputs of the CUDA adapter (vram_budget.cpp).
  target_link_libraries(ninfer_core PRIVATE dxgi)
endif()
