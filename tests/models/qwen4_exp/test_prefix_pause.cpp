// What a pause of a Qwen4Exp lane leaves in the prefix cache (program/prefix/pause_capture.h; design §19.3.13),
// host only. ProgramImpl::pause_replay_tokens prices a pause with pause_captures_frontier, so the function must
// answer as the pause itself acts (ProgramImpl::prefix_finish): after the MTP flush (rule MR3) and the publication
// of every block the flushed frontier covers. The cases are stated from those rules, not from the function: a lane
// decoding at a block boundary has its last block unpublished and its last MTP cell pending until the pause writes
// them, and was once priced as a full replay there.

#include "models/qwen4_exp/program/prefix/pause_capture.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace q4 = infernix::models::qwen4_exp::prefix;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// A lane decoding at `frontier` as its commits leave it: the next input is known (history = frontier + 1), blocks
// are published up to min(mtp_cells, frontier - 1) with a drafter (the next round's step 0 rewrites cell
// frontier - 1) and up to the frontier without one.
q4::PauseLane decoding(std::uint32_t frontier, std::uint32_t deepest, bool mtp, std::uint32_t mtp_cells) {
    const std::uint32_t published = mtp ? (mtp_cells < frontier - 1U ? mtp_cells : frontier - 1U) : frontier;
    return {.frontier    = frontier,
            .deepest     = deepest,
            .history     = frontier + 1U,
            .path_blocks = published / 64U,
            .mtp_live    = mtp,
            .mtp_cells   = mtp_cells};
}

} // namespace

int main() {
    // Without a drafter every block below the frontier is on the path already.
    check(q4::pause_captures_frontier(decoding(1000, 512, false, 0)), "plain decode inside a block");
    check(q4::pause_captures_frontier(decoding(1024, 512, false, 0)), "plain decode at a block boundary");

    // MTP, the pending cell (mtp_cells = frontier - 1): the pause flushes it, so every cell is final.
    check(q4::flushed_mtp_cells(decoding(1000, 512, true, 999)) == 1000, "the pending cell is flushed");
    check(q4::pause_captures_frontier(decoding(1000, 512, true, 999)), "MTP decode inside a block");
    // At a block boundary the last block (cells 960..1023) is unpublished and its last cell pending before the
    // pause; the flush and the publication complete both. This is the case the prediction got wrong.
    {
        const q4::PauseLane lane = decoding(1024, 512, true, 1023);
        check(lane.path_blocks == 15, "the block ending at the frontier is not on the path before the pause");
        check(q4::pause_captures_frontier(lane), "MTP decode at a block boundary is captured");
    }
    // The settled state after a catch-up (mtp_cells = frontier): nothing to flush, the same answer.
    check(q4::pause_captures_frontier(decoding(1024, 512, true, 1024)), "MTP decode at a boundary, cells settled");

    // A pending cell whose token is unknown cannot be flushed (the lane knows no token past its frontier): at a
    // block boundary the last block stays unfinished and the pause captures nothing; inside a block the tail's
    // cell is rewritten by the resume, so the capture stands.
    {
        q4::PauseLane lane = decoding(1024, 512, true, 1023);
        lane.history       = 1024;
        check(q4::flushed_mtp_cells(lane) == 1023, "no flush without the next token");
        check(!q4::pause_captures_frontier(lane), "an unfinished MTP block at the frontier is not captured");
        lane          = decoding(1000, 512, true, 999);
        lane.history  = 1000;
        check(q4::pause_captures_frontier(lane), "a pending cell inside the tail block does not block the capture");
    }
    // Cells further behind than the pending one (a cancelled round left them): never flushed.
    check(!q4::pause_captures_frontier(decoding(1024, 512, true, 1000)), "MTP cells more than one behind");

    // A frontier at the lineage's deepest snapshot has nothing new to capture, however complete its blocks.
    check(!q4::pause_captures_frontier(decoding(1024, 1024, false, 0)), "the frontier is the deepest snapshot");
    check(q4::pause_captures_frontier(decoding(1025, 1024, false, 0)), "one token past the deepest snapshot");

    // A path short of the frontier's anchor that publication cannot complete (the history ends below it).
    {
        q4::PauseLane lane{.frontier = 256, .deepest = 0, .history = 200, .path_blocks = 2};
        check(!q4::pause_captures_frontier(lane), "blocks the lane cannot publish leave no anchor");
        lane.path_blocks = 4;
        check(q4::pause_captures_frontier(lane), "a path that reaches the anchor already");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("qwen4_exp prefix pause capture: ok\n");
    return 0;
}
