#pragma once

// The staged SIMT launchers' choice among a kernel's four forms (Q4, Q5 and Q8 A16): Full when a slice's row blocks
// and token tiles are all whole, FullK when K is a whole number of the schedule's warp stages.

namespace infernix::ops::detail {

// Calls `launch.template operator()<Full, FullK>()` with the form the two run-time facts select.
template <class Launch>
void dispatch_simt_tiles(bool full, bool full_k, const Launch& launch) {
    if (full && full_k)
        launch.template operator()<true, true>();
    else if (full)
        launch.template operator()<true, false>();
    else if (full_k)
        launch.template operator()<false, true>();
    else
        launch.template operator()<false, false>();
}

} // namespace infernix::ops::detail
