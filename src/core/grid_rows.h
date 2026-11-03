#pragma once

#include <algorithm>

namespace infernix {

// gridDim.y and gridDim.z hold at most 65,535 blocks. A kernel that gives each column (token) of a
// call a row of blocks launches grid_rows(columns) rows and loops
// `for (int t = blockIdx.y; t < columns; t += gridDim.y)`: a call of up to 65,535 columns runs one
// column a row as before, and a wider one (a prefill chunk past 65,535 tokens) wraps.
inline constexpr int kMaxGridRows = 65535;

inline unsigned grid_rows(int columns) { return static_cast<unsigned>(std::min(columns, kMaxGridRows)); }

} // namespace infernix
