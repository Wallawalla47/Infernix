#!/usr/bin/env bash
# Runs every Flash-Next M0 probe and records the outputs (docs/maintainer/qwen3_8-flash-next-design.md §19).
#
#   tools/flash_next_probe/run_m0.sh CHECKPOINT_DIR NVME_FILE [OUT_DIR]
#
# CHECKPOINT_DIR is the local nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot. NVME_FILE is a large file on
# the drive that will hold the n-gram volume (the checkpoint's largest shard will do). Outputs land
# in OUT_DIR (default profiles/m0/<host>-<date>). Set PYTHON to the Python 3.11 interpreter.
set -euo pipefail
checkpoint=${1:?checkpoint directory}
nvme_file=${2:?large file on the target NVMe}
out=${3:-profiles/m0/$(hostname)-$(date +%Y%m%d-%H%M)}
python=${PYTHON:-python3}
mkdir -p "$out" build

g++ -O3 -std=c++20 -pthread -ffp-contract=off -fno-fast-math -Isrc \
  tools/flash_next_probe/host_probe.cpp src/ops/offloaded_sparse_moe/cpu/w4a4_expert.cpp \
  src/ops/offloaded_sparse_moe/cpu/expert_team.cpp -o build/flash_next_host_probe
g++ -O2 -std=c++20 -pthread tools/flash_next_probe/nvme_probe.cpp -o build/flash_next_nvme_probe
nvcc -O3 -std=c++17 -arch=sm_120a tools/flash_next_probe/gpu_probe.cu -o build/flash_next_gpu_probe

{
  uname -a
  lscpu
  grep -E 'MemTotal|Hugepagesize|HugePages_Total' /proc/meminfo
  cat /sys/kernel/mm/transparent_hugepage/enabled
  ulimit -l
  nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.max,pcie.link.width.max,clocks.max.mem --format=csv
  sudo -n dmidecode -t memory 2>/dev/null | grep -E 'Size|Speed|Part' || echo "dmidecode unavailable (needs sudo)"
  cat /sys/module/nvme/parameters/poll_queues 2>/dev/null || true
} > "$out/inventory.txt" 2>&1

"$python" -m tools.flash_next.inspect_checkpoint "$checkpoint" --json "$out/source_facts.json" | tee "$out/source_facts.txt"
"$python" -m tools.flash_next.a4_reference --json "$out/a4_vs_modelopt.json" | tee "$out/a4_vs_modelopt.txt"
./build/flash_next_gpu_probe | tee "$out/gpu_probe.txt"
# The link trains down at idle; sample it while the probe's copies run.
( ./build/flash_next_gpu_probe copy >/dev/null & sleep 1;
  nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,clocks.mem --format=csv; wait ) \
  > "$out/pcie_under_load.txt"
./build/flash_next_host_probe | tee "$out/host_probe.txt"
./build/flash_next_nvme_probe "$nvme_file" | tee "$out/nvme_probe.txt"
echo "M0 outputs in $out"
