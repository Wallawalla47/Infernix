# CPU placement for the expert offload worker team

The expert offload worker team (the CPU threads that stage expert weights from RAM while the
GPU decodes) is sized by `cpu_expert_workers` (default **6**; `src/runtime/expert_offload/expert_offload.h`).
The team is **not** affected by `OMP_NUM_THREADS`.

## Switch

    INFERNIX_TMP_CPU_PLACEMENT=<mode>[,<workers>]

- `mode` ∈ `default` | `physical` | `physical-set`
- `workers` ∈ 1–40; when omitted (or 0) the configured `cpu_expert_workers` is used.

`physical` pins the team to distinct physical cores (two workers never share an SMT pair);
`physical-set` pins to a fixed core set. Parsed in `src/core/temporary_switches.cpp`; the
override in `src/runtime/expert_offload/placement.cpp`.

## Recommendation

On ≥32-core machines, pin the team to all physical cores:
`INFERNIX_TMP_CPU_PLACEMENT=physical,<nphysical>`.

Measured on EPYC 7532 (32C/64T) + RTX 5070 Ti, Qwen3.8-Flash-Next FP4-Local, 256-token decode
probe (median of 5 interleaved rounds; scheduling only, no output change):

| Placement   | Median tok/s | vs 6 workers |
|-------------|-------------:|-------------:|
| 6 (default) |          70.4 |            — |
| default,16  |          73.8 |          +8 % |
| physical,16 |          76.0 |         +12 % |
| physical,32 |          83.3 |         +19 % |

`physical` beats `default` at equal worker counts (SMT siblings contend for the staging
bandwidth), and the gain keeps growing up to the full physical core count.
