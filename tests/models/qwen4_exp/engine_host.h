#pragma once

// The process's EngineHost for the Qwen3.8-Flash-Next real tests (include/infernix/engine.h): every Engine a test
// starts on it takes the SSD expert tier's page-locked slots and the expert records the last Engine left there, so it
// neither page-locks that memory again nor reads those records (about 18 s of a 31 s start on Dense8). Each start
// still plans, sizes and fills its own tier, warms its own expert cache and captures its own graphs; the tests that
// compare a start with a fresh one (test_engine_host_real.cpp) construct without it. Never destroyed: its memory goes
// with the process, after the CUDA runtime's teardown could no longer free it.

#include "infernix/engine.h"

namespace infernix::test {

inline EngineHost& engine_host() {
    static EngineHost* const host = new EngineHost();
    return *host;
}

} // namespace infernix::test
