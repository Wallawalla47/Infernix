#pragma once
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/program/program.h"

namespace infernix::models::qwen3_5 {
struct RuntimeTypes {
    using Frontend          = qwen3_5::Frontend;
    using PreparedPrompt    = qwen3_5::PreparedPrompt;
    using NgramArchive      = qwen3_5::NgramArchive;
    using OutputSession     = qwen3_5::OutputSession;
    using PublishedOutput   = qwen3_5::PublishedOutput;
    using SequencePlanner   = qwen3_5::SequencePlanner;
    using SequencePlan      = qwen3_5::SequencePlan;
    using RequestBasePlan   = qwen3_5::RequestBasePlan;
    using SequenceHandle    = qwen3_5::SequenceHandle;
    using SourceCandidate   = qwen3_5::SourceCandidate;
    using ResumeState       = qwen3_5::ResumeState;
    using ExecutionUnit     = qwen3_5::ExecutionUnit;
    using ExecutionUnitKind = qwen3_5::ExecutionUnitKind;
    using ContextProgress   = qwen3_5::ContextProgress;
    using PendingBatch      = qwen3_5::PendingBatch;
    using PrefillProgress   = qwen3_5::PrefillProgress;
    using ReplayProgress    = qwen3_5::ReplayProgress;
    using CommitResult      = qwen3_5::CommitResult;
    using DiscardResult     = qwen3_5::DiscardResult;
    using FinishResult      = qwen3_5::FinishResult;
    using AbortResult       = qwen3_5::AbortResult;
    using Program           = qwen3_5::Program;
};
} // namespace infernix::models::qwen3_5
