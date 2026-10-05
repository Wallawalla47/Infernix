#pragma once
#include "ops/linear/bf16/bf16_schedule.cuh"

namespace ninfer::ops::detail {

using Bf16A16MmaR32T32K256S3 =
    Bf16A16MmaSchedule<32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Bf16A16MmaR32T32K192S2 =
    Bf16A16MmaSchedule<32, 32, 192, 16, 8, 2, 2, Cache::cg, Cache::ca,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

using Bf16A16MmaR32T32K128S3 =
    Bf16A16MmaSchedule<32, 32, 128, 16, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Bf16A16MmaR64T32K64S3 =
    Bf16A16MmaSchedule<64, 32, 64, 32, 16, 3, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Bf16A16SlicedR32T16W4  = Bf16A16SlicedKMmaSchedule<32, 16, 4>;
using Bf16A16TmaR64T32K64S3  = Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>;
using Bf16A16TmaR64T64K64S3  = Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>;
using Bf16A16TmaR64T64K128S2 = Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>;
using Bf16A16TmaR64T128K64S2 = Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>;

// Vision tower routes. RowFast rasterization runs every row tile of one token tile before the
// next token tile, so a large-T call reads each activation tile from DRAM about once while the
// small tower weight stays resident in L2.
using Bf16A16TmaR64T128K64S2Rows =
    Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 1, Bf16MmaRaster::RowFast>;
// Tail-capable forms of the same tiles for the MLP's 4304 rows (fc1) or 4304 inputs (fc2).
using Bf16A16TmaTailR64T32S3 = Bf16A16TmaTailMmaSchedule<64, 32, 32, 16, 3>;
using Bf16A16TmaTailR64T64S3 = Bf16A16TmaTailMmaSchedule<64, 64, 32, 32, 3>;
using Bf16A16TmaTailR64T128S2Rows =
    Bf16A16TmaTailMmaSchedule<64, 128, 32, 32, 2, 1, Bf16MmaRaster::RowFast>;

} // namespace ninfer::ops::detail
