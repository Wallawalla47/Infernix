# Qwen3.8-Flash-Next (Qwen4Exp) Program pieces that are already implemented: the host expert cache
# (docs/maintainer/qwen3_8-flash-next-design.md §9). Host-only; it joins ninfer_model_runtime when the
# Qwen4Exp Program lands.
add_library(ninfer_qwen4_exp_expert_cache STATIC
  "${CMAKE_CURRENT_LIST_DIR}/program/expert_cache/expert_cache.cpp")
ninfer_internal_includes(ninfer_qwen4_exp_expert_cache)
