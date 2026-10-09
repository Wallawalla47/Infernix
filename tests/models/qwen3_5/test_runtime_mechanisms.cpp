#include "core/layout.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/program/speculative/mtp_alignment.h"
#include "models/qwen3_5/program/round_buffers.h"
#include "models/qwen3_5/program/vision_control.h"

#include "models/qwen3_5/program/ledger_splits.h"
#include "runtime/contract/timing.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = infernix::models::qwen3_5;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void test_execution_timing_domains() {
    using namespace infernix::runtime;
    ExecutionTimingRecorder recorder(ExecutionTimingPhase::Paused);
    recorder.include(
        {.submit_host_ns = 10, .device_wait_ns = 20, .post_host_ns = 30, .gpu_elapsed_ns = 100});
    recorder.include(
        {.submit_host_ns = 4, .device_wait_ns = 5, .post_host_ns = 6, .gpu_elapsed_ns = 80});
    const auto timing = recorder.finish();
    expect(timing.gpu_elapsed_ns == 180 && timing.host_ns() == 50 && timing.elapsed_ns() == 75,
           "overlapping GPU intervals must accumulate separately from Host and wall phases");
    const auto repeated = recorder.finish();
    expect(repeated.gpu_elapsed_ns == timing.gpu_elapsed_ns &&
               repeated.elapsed_ns() == timing.elapsed_ns(),
           "reading a completed execution timing must not accumulate its work again");
}

q36::DecoderStateSpec decoder_spec(infernix::KvCacheStorage storage, bool mtp) {
    return q36::DecoderStateSpec{
        .full_attention_layers     = 2,
        .mtp_layers                = 1,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 256,
        .kv_storage                = storage,
        .enable_mtp                = mtp,
        .text_physical_page_groups = 5,
        .mtp_physical_page_groups  = mtp ? 4U : 0U,
    };
}

void test_decoder_layout() {
    infernix::LayoutBuilder bf16_builder;
    const q36::DecoderStateLayout bf16 = q36::plan_decoder_state(
        bf16_builder, decoder_spec(infernix::KvCacheStorage::BFloat16, false));
    (void)bf16_builder.finish(256);
    expect(bf16.text_kv.pages.planes.size() == 4, "BF16 Text KV has K/V planes per layer");
    expect(bf16.text_kv.pages.spec.page_group_count == 5 &&
               bf16.text_kv.execution_tables.spec.logical_page_capacity == 3 &&
               bf16.text_kv.execution_tables.spec.table_rows == 1,
           "Text KV separates five physical pages from three logical pages");
    expect(bf16.text_kv.pages.planes[0].geometry.dtype == infernix::DType::BF16 &&
               bf16.text_kv.pages.planes[1].geometry.dtype == infernix::DType::FP16 &&
               bf16.text_kv.pages.planes[2].geometry.dtype == infernix::DType::BF16 &&
               bf16.text_kv.pages.planes[3].geometry.dtype == infernix::DType::FP16,
           "BF16 KV has BF16 K and FP16 V without scale planes");
    expect(!bf16.mtp_kv.has_value(), "disabled MTP omits KV storage");
    expect(bf16.kv_payload_bytes() == bf16.text_kv.payload_bytes(), "BF16 KV payload accounting");

    infernix::LayoutBuilder int8_builder;
    const q36::DecoderStateLayout int8 = q36::plan_decoder_state(
        int8_builder, decoder_spec(infernix::KvCacheStorage::Int8Group64, true));
    (void)int8_builder.finish(256);
    expect(int8.text_kv.pages.planes.size() == 8 &&
               int8.text_kv.pages.planes[2].geometry.dtype == infernix::DType::FP16 &&
               int8.text_kv.pages.planes[3].geometry.dtype == infernix::DType::FP16,
           "INT8 Text KV has code and scale planes per layer");
    expect(int8.mtp_kv.has_value() && int8.mtp_kv->layers == 1 &&
               int8.mtp_kv->pages.planes.size() == 4 &&
               int8.mtp_kv->pages.spec.page_group_count == 4 &&
               int8.mtp_kv->execution_tables.spec.logical_page_capacity == 3,
           "enabled MTP has one paged KV layer");
    expect(int8.mtp_kv && int8.mtp_kv->pages.planes[2].geometry.dtype == infernix::DType::FP16 &&
               int8.mtp_kv->pages.planes[3].geometry.dtype == infernix::DType::FP16,
           "INT8 MTP KV has scale planes");
    expect(int8.kv_payload_bytes() == int8.text_kv.payload_bytes() + int8.mtp_kv->payload_bytes(),
           "INT8 Text/MTP KV payload accounting");

    q36::DecoderStateSpec fp8_spec = decoder_spec(infernix::KvCacheStorage::Fp8E4M3Row256, true);
    infernix::LayoutBuilder fp8_builder;
    const q36::DecoderStateLayout fp8 = q36::plan_decoder_state(fp8_builder, fp8_spec);
    (void)fp8_builder.finish(256);
    expect(fp8.text_kv.pages.planes.size() == 8 &&
               fp8.text_kv.pages.planes[0].geometry.dtype == infernix::DType::FP8_E4M3FN &&
               fp8.text_kv.pages.planes[2].geometry.dtype == infernix::DType::FP16 &&
               fp8.text_kv.pages.planes[2].geometry.leading_extent == 1,
           "FP8 Text KV has row-scaled code and scale planes per layer");
    expect(fp8.mtp_kv && fp8.mtp_kv->pages.planes.size() == 4 &&
               fp8.mtp_kv->pages.planes[0].geometry.dtype == infernix::DType::FP8_E4M3FN &&
               fp8.mtp_kv->pages.planes[2].geometry.leading_extent == 1,
           "FP8 MTP KV has row-scaled code and scale planes");
    expect(fp8.kv_payload_bytes() == fp8.text_kv.payload_bytes() + fp8.mtp_kv->payload_bytes(),
           "FP8 Text/MTP KV payload accounting");

    infernix::LayoutBuilder nvfp4_builder;
    const q36::DecoderStateLayout nvfp4 = q36::plan_decoder_state(
        nvfp4_builder, decoder_spec(infernix::KvCacheStorage::Nvfp4Group16, true));
    (void)nvfp4_builder.finish(256);
    expect(nvfp4.text_kv.pages.planes.size() == 8 &&
               nvfp4.text_kv.pages.planes[0].geometry.dtype == infernix::DType::U8 &&
               nvfp4.text_kv.pages.planes[0].geometry.leading_extent == 128 &&
               nvfp4.text_kv.pages.planes[2].geometry.dtype == infernix::DType::U8 &&
               nvfp4.text_kv.pages.planes[2].geometry.leading_extent == 16,
           "NVFP4 Text KV has packed E2M1 code and E4M3 scale planes");
    constexpr std::size_t nvfp4_vector_bytes = 128 + 16;
    constexpr std::size_t text_vectors       = 2ULL * 5ULL * 64ULL * 2ULL;
    constexpr std::size_t mtp_vectors        = 1ULL * 4ULL * 64ULL * 2ULL;
    expect(nvfp4.text_kv.payload_bytes() == 2ULL * nvfp4_vector_bytes * text_vectors &&
               nvfp4.mtp_kv &&
               nvfp4.mtp_kv->payload_bytes() == 2ULL * nvfp4_vector_bytes * mtp_vectors,
           "NVFP4 Text/MTP physical payload bytes");

    infernix::LayoutBuilder k8v4_builder;
    const q36::DecoderStateLayout k8v4 = q36::plan_decoder_state(
        k8v4_builder, decoder_spec(infernix::KvCacheStorage::Fp8KeyNvfp4Value, true));
    (void)k8v4_builder.finish(256);
    expect(k8v4.text_kv.pages.planes.size() == 8 &&
               k8v4.text_kv.pages.planes[0].geometry.dtype == infernix::DType::FP8_E4M3FN &&
               k8v4.text_kv.pages.planes[0].geometry.leading_extent == 256 &&
               k8v4.text_kv.pages.planes[1].geometry.dtype == infernix::DType::U8 &&
               k8v4.text_kv.pages.planes[1].geometry.leading_extent == 128 &&
               k8v4.text_kv.pages.planes[2].geometry.dtype == infernix::DType::FP16 &&
               k8v4.text_kv.pages.planes[2].geometry.leading_extent == 1 &&
               k8v4.text_kv.pages.planes[3].geometry.dtype == infernix::DType::U8 &&
               k8v4.text_kv.pages.planes[3].geometry.leading_extent == 16,
           "K8V4 Text KV preserves independent key/value code and scale geometry");
    constexpr std::size_t k8_vector_bytes = 256 + 2;
    constexpr std::size_t v4_vector_bytes = 128 + 16;
    expect(k8v4.text_kv.payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * text_vectors &&
               k8v4.mtp_kv &&
               k8v4.mtp_kv->payload_bytes() == (k8_vector_bytes + v4_vector_bytes) * mtp_vectors,
           "K8V4 Text/MTP asymmetric physical payload bytes");

    // Vector-quantized formats: compact code planes with one FP16 row scale. Their exact
    // recent-key window is sequence state (StateImage), not part of the page pool.
    for (const auto storage :
         {infernix::KvCacheStorage::Vq2, infernix::KvCacheStorage::Q4KeyVq2Value}) {
        infernix::LayoutBuilder vq_builder;
        const q36::DecoderStateSpec spec = decoder_spec(storage, true);
        const q36::DecoderStateLayout vq = q36::plan_decoder_state(vq_builder, spec);
        (void)vq_builder.finish(256);
        const int key_bytes = storage == infernix::KvCacheStorage::Vq2 ? 64 : 128;
        expect(vq.text_kv.pages.planes.size() == 8 &&
                   vq.text_kv.pages.planes[0].geometry.dtype == infernix::DType::U8 &&
                   vq.text_kv.pages.planes[0].geometry.leading_extent == key_bytes &&
                   vq.text_kv.pages.planes[1].geometry.leading_extent == 64 &&
                   vq.text_kv.pages.planes[2].geometry.dtype == infernix::DType::FP16 &&
                   vq.text_kv.pages.planes[2].geometry.leading_extent == 1 &&
                   vq.text_kv.pages.planes[3].geometry.leading_extent == 1,
               "vector-quantized Text KV has code planes and FP16 row-scale planes");
        expect(vq.text_kv.payload_bytes() ==
                       static_cast<std::size_t>(key_bytes + 2 + 64 + 2) * text_vectors &&
                   vq.mtp_kv &&
                   vq.mtp_kv->payload_bytes() ==
                       static_cast<std::size_t>(key_bytes + 2 + 64 + 2) * mtp_vectors,
               "vector-quantized Text/MTP physical payload bytes");
    }
}

void test_round_layout() {
    static_assert(offsetof(q36::DFlashDecodeIngress, anchors) % 16 == 0);
    static_assert(offsetof(q36::DFlashDecodeIngress, execution_frontiers) % 16 == 0);
    static_assert(offsetof(q36::DFlashDecodeIngress, sampling) % 16 == 0);
    infernix::LayoutBuilder builder;
    q36::RoundStateLayout round = q36::begin_round_state_layout(
        builder, q36::RoundStateSpec{.hidden       = 32,
                                     .output_rows  = 128,
                                     .draft_window = 5,
                                     .backend      = infernix::SpeculativeBackend::Mtp});
    (void)builder.add_tensor(infernix::DType::BF16, {32, 16}, 256, "exact prefill hidden");
    q36::complete_round_state_layout(builder, round);
    (void)builder.finish(256);
    expect(round.complete, "round layout completes");
    expect(round.logits.shape[0] == 128 && round.logits.shape[1] == 1, "round logits shape");
    expect(round.mtp.has_value() && round.mtp->draft_tokens.shape[0] == 5 &&
               round.mtp->target_input_ids.shape[0] == 6,
           "MTP prefill scratch shapes");
    expect(round.mtp.has_value() && round.mtp->position.shape[0] == 1,
           "MTP prefill scratch is explicit");
    expect(round.mtp_decode.has_value() && round.mtp_decode->alignment_ids.shape[0] == 6 &&
               round.mtp_decode->alignment_ids.shape[1] == 1,
           "MTP decode frame is explicit");

    infernix::LayoutBuilder speculative_builder;
    q36::RoundStateLayout dflash = q36::begin_round_state_layout(
        speculative_builder, q36::RoundStateSpec{.hidden       = 32,
                                                 .output_rows  = 128,
                                                 .draft_window = 15,
                                                 .backend = infernix::SpeculativeBackend::DFlash});
    q36::complete_round_state_layout(speculative_builder, dflash);
    (void)speculative_builder.finish(256);
    expect(dflash.logits.shape[1] == 1 && dflash.dflash_prefill.has_value() &&
               dflash.dflash_prefill->local_append_count.shape[0] == 1 &&
               dflash.dflash_decode.has_value() &&
               dflash.dflash_decode->draft_tokens.shape[0] == 15,
           "K=15 DFlash storage is backend-owned");
    expect(!dflash.mtp.has_value() && !dflash.mtp_decode.has_value(),
           "DFlash layout does not allocate MTP storage");

    infernix::LayoutBuilder scoring_builder;
    auto scoring = q36::begin_round_state_layout(
        scoring_builder, {.hidden = 32, .output_rows = 128, .causal_scoring = true});
    q36::complete_round_state_layout(scoring_builder, scoring);
    expect(scoring.rope_delta.region.bytes != 0 && scoring.text_kv_table_row.region.bytes != 0,
           "scoring keeps its Text prefill controls");
    expect(!scoring.ordinary && !scoring.mtp_decode && !scoring.dflash_decode &&
               scoring.token.region.bytes == 0 && scoring.logits.region.bytes == 0,
           "scoring does not reserve generation frames or sampled output");
    for (const std::uint32_t rows : {1U, 2U, 8U}) {
        infernix::LayoutBuilder copy_builder;
        auto copy_layout = q36::begin_round_state_layout(
            copy_builder, {.hidden         = 32,
                           .output_rows    = 128,
                           .batch_capacity = rows,
                           .draft_window   = 15,
                           .backend        = infernix::SpeculativeBackend::DFlash2});
        q36::complete_round_state_layout(copy_builder, copy_layout);
        const auto bytes = copy_builder.finish(256);
        std::vector<std::byte> storage(bytes + 255);
        auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        address      = (address + 255) & ~std::uintptr_t(255);
        q36::RoundState copy({reinterpret_cast<void*>(address), bytes}, copy_layout);
        const int capacity = static_cast<int>(rows);
        for (std::uint32_t k = 1; k <= 15; ++k) {
            const auto narrow = copy.dflash_decode->narrowed(k);
            expect(narrow.target_logits.ne[1] == static_cast<int>(k + 1) &&
                       narrow.target_logits.ne[2] == capacity &&
                       narrow.draft_tokens.ne[0] == static_cast<int>(k) &&
                       narrow.draft_tokens.ne[1] == capacity &&
                       narrow.candidate_ids.ne[1] == static_cast<int>(k) &&
                       narrow.candidate_ids.ne[2] == capacity &&
                       narrow.licensed_tokens.ne[0] == static_cast<int>(k + 1) &&
                       narrow.verify_ids.ne[1] == capacity,
                   "narrowed frame shapes cover every row at the round width");
            expect(narrow.target_logits.is_contiguous() && narrow.target_hidden.is_contiguous() &&
                       narrow.candidate_ids.is_contiguous() && narrow.proposal_q.is_contiguous() &&
                       narrow.draft_tokens.is_contiguous(),
                   "narrowed frame rows are dense at the round width");
            expect(narrow.target_logits.data == copy.dflash_decode->target_logits.data &&
                       narrow.draft_tokens.data == copy.dflash_decode->draft_tokens.data &&
                       narrow.append_positions.ne[0] == 16 &&
                       narrow.append_positions.ne[1] == capacity,
                   "narrowed frame aliases storage and keeps native catch-up capacity");
            expect(reinterpret_cast<std::uintptr_t>(narrow.anchors.data) % 16 == 0 &&
                       reinterpret_cast<std::uintptr_t>(narrow.execution_frontiers.data) % 16 == 0,
                   "narrowed frame controls retain vector alignment");
        }
        for (std::uint32_t k : {0, 16}) {
            bool rejected = false;
            try {
                (void)copy.dflash_decode->narrowed(k);
            } catch (const std::invalid_argument&) { rejected = true; }
            expect(rejected, "invalid narrowed frame width rejected");
        }
    }
    infernix::LayoutBuilder mtp_copy_builder;
    auto mtp_copy_layout = q36::begin_round_state_layout(
        mtp_copy_builder, {.hidden         = 32,
                           .output_rows    = 128,
                           .batch_capacity = 1,
                           .draft_window   = 15,
                           .backend        = infernix::SpeculativeBackend::Mtp});
    q36::complete_round_state_layout(mtp_copy_builder, mtp_copy_layout);
    const auto mtp_bytes = mtp_copy_builder.finish(256);
    std::vector<std::byte> mtp_storage(mtp_bytes + 255);
    const auto mtp_address =
        (reinterpret_cast<std::uintptr_t>(mtp_storage.data()) + 255) & ~std::uintptr_t(255);
    q36::RoundState mtp_copy({reinterpret_cast<void*>(mtp_address), mtp_bytes}, mtp_copy_layout);
    expect(mtp_copy.mtp_decode->ar_positions.ne[1] == 4 &&
               mtp_copy.mtp_decode->ar_rope_positions.ne[1] == 4 &&
               mtp_copy.mtp_decode->ar_valid_columns.ne[1] == 4,
           "MTP wide copy frame allocates at most four neural AR steps");
    for (std::uint32_t k = 1; k <= 15; ++k) {
        for (std::uint32_t next_k = 1; next_k <= 5; ++next_k) {
            const auto frame = mtp_copy.mtp_decode->narrowed(k, next_k);
            expect(frame.target_logits.ne[1] == static_cast<int>(k + 1) &&
                       frame.current_drafts.ne[0] == static_cast<int>(k) &&
                       frame.alignment_ids.ne[0] == static_cast<int>(k + 1) &&
                       frame.next_drafts.ne[1] == static_cast<int>(next_k) &&
                       frame.ar_positions.ne[1] == static_cast<int>(std::max(1U, next_k - 1U)),
                   "MTP copy verification and next proposal dimensions are independent");
            expect(frame.target_hidden.data == mtp_copy.mtp_decode->target_hidden.data &&
                       frame.target_logits.is_contiguous() &&
                       frame.alignment_hidden.is_contiguous(),
                   "MTP copy frame remains a dense prefix");
        }
    }
    for (const auto [k, next_k] :
         std::vector<std::pair<unsigned, unsigned>>{{0, 3}, {16, 3}, {15, 0}, {15, 6}}) {
        bool rejected = false;
        try {
            (void)mtp_copy.mtp_decode->narrowed(k, next_k);
        } catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "MTP invalid verify/proposal frame width rejected");
    }
    // Above one request the narrowed MTP frame stays a dense [k+1,C] view of the native storage,
    // and the step-major proposal tensors keep their row stride.
    infernix::LayoutBuilder mtp_batch_builder;
    auto mtp_batch_layout = q36::begin_round_state_layout(
        mtp_batch_builder, {.hidden         = 32,
                            .output_rows    = 128,
                            .batch_capacity = 3,
                            .draft_window   = 15,
                            .backend        = infernix::SpeculativeBackend::Mtp});
    q36::complete_round_state_layout(mtp_batch_builder, mtp_batch_layout);
    const auto mtp_batch_bytes = mtp_batch_builder.finish(256);
    std::vector<std::byte> mtp_batch_storage(mtp_batch_bytes + 255);
    const auto mtp_batch_address =
        (reinterpret_cast<std::uintptr_t>(mtp_batch_storage.data()) + 255) & ~std::uintptr_t(255);
    q36::RoundState mtp_batch({reinterpret_cast<void*>(mtp_batch_address), mtp_batch_bytes},
                              mtp_batch_layout);
    for (const auto [k, next_k] :
         std::vector<std::pair<unsigned, unsigned>>{{3, 3}, {7, 3}, {15, 3}, {5, 5}}) {
        const auto frame = mtp_batch.mtp_decode->narrowed(k, next_k);
        expect(frame.target_hidden.ne[1] == static_cast<int>(k + 1) &&
                   frame.target_hidden.ne[2] == 3 &&
                   frame.target_hidden.data == mtp_batch.mtp_decode->target_hidden.data &&
                   frame.target_hidden.is_contiguous() &&
                   frame.current_drafts.ne[0] == static_cast<int>(k) &&
                   frame.current_drafts.ne[1] == 3 &&
                   frame.verify_ids.ne[0] == static_cast<int>(k + 1) &&
                   frame.next_drafts.ne[1] == static_cast<int>(next_k) &&
                   frame.next_drafts.nb[1] == mtp_batch.mtp_decode->next_drafts.nb[1],
               "MTP frame narrowed above one request");
    }
}

void test_mtp_alignment() {
    const std::vector<std::int32_t> scatter{2, 4, 7};
    const q36::MtpAlignmentWindow first = q36::plan_mtp_alignment_window(8, 0, 4);
    expect(first.hidden_begin == 0 && first.position_begin == 0 &&
               first.shifted_embedding_begin == 1 && first.columns == 4 &&
               !first.final_column_uses_generated_token,
           "non-final MTP alignment window");
    const q36::MtpVisualOverlap first_visual = q36::shifted_visual_overlap(scatter, 8, first);
    expect(first_visual.source_begin == 0 &&
               first_visual.destination_columns == std::vector<std::int32_t>({1, 3}),
           "non-final shifted visual overlap");

    const q36::MtpAlignmentWindow final = q36::plan_mtp_alignment_window(8, 4, 4);
    expect(final.shifted_embedding_begin == 5 && final.final_column_uses_generated_token,
           "final MTP alignment window");
    const q36::MtpVisualOverlap final_visual = q36::shifted_visual_overlap(scatter, 8, final);
    expect(final_visual.source_begin == 2 &&
               final_visual.destination_columns == std::vector<std::int32_t>({2}),
           "final shifted visual overlap excludes generated-token column");
}

void test_vision_control() {
    q36::PreparedPromptData prompt;
    prompt.token_ids.resize(7);
    prompt.token_types           = {0, static_cast<std::uint8_t>(q36::PromptModality::Image),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0, static_cast<std::uint8_t>(q36::PromptModality::Video),
                                    0};
    prompt.prepare.media_items   = 2;
    prompt.prepare.raw_patches   = 12;
    prompt.prepare.vision_tokens = 3;
    prompt.vision_items          = {
        q36::VisionItem{.modality    = q36::PromptModality::Image,
                                 .grid        = {.temporal = 1, .height = 2, .width = 2},
                                 .patch_begin = 0,
                                 .patch_count = 4,
                                 .token_spans = {{.begin = 1, .count = 1}}},
        q36::VisionItem{.modality    = q36::PromptModality::Video,
                                 .grid        = {.temporal = 2, .height = 2, .width = 2},
                                 .patch_begin = 4,
                                 .patch_count = 8,
                                 .token_spans = {{.begin = 3, .count = 1}, {.begin = 5, .count = 1}}},
    };

    const q36::VisionControlPlan plan =
        q36::plan_vision_control(prompt, {.spatial_merge_size = 2, .position_grid_side = 48});
    const q36::VisionControl control = q36::build_vision_control(prompt, plan, 0);
    expect(control.items.size() == 2, "Vision per-item control count");
    expect(control.items[0].patch_begin == 0 && control.items[0].patch_count == 4 &&
               control.items[0].merged_count == 1 && control.items[0].segment_length == 4 &&
               control.items[0].segment_count == 1 &&
               control.items[0].scatter_indices == std::vector<std::int32_t>({1}) &&
               control.items[0].position_ids.size() == 8 &&
               control.items[0].position_table_indices.size() == 16 &&
               control.items[0].position_table_weights.size() == 16,
           "image item control offsets");
    expect(control.items[1].patch_begin == 4 && control.items[1].patch_count == 8 &&
               control.items[1].merged_count == 2 && control.items[1].segment_length == 4 &&
               control.items[1].segment_count == 2 &&
               control.items[1].scatter_indices == std::vector<std::int32_t>({3, 5}) &&
               control.items[1].position_ids.size() == 16 &&
               control.items[1].position_table_indices.size() == 32 &&
               control.items[1].position_table_weights.size() == 32,
           "video item control offsets");

    const q36::VisionControl suffix = q36::build_vision_control(prompt, plan, 1);
    expect(suffix.prepared_item_begin == 1 && suffix.items.size() == 1 &&
               suffix.items[0].patch_begin == control.items[1].patch_begin &&
               suffix.items[0].scatter_indices == control.items[1].scatter_indices &&
               suffix.items[0].position_ids == control.items[1].position_ids,
           "Vision suffix control contents");
}

// Replay splits a recovered ledger where its execution was split: at the prompt's rewrite
// frontiers and after a generated control span's split point.
void test_ledger_splits() {
    q36::PreparedPromptData prompt;
    prompt.token_ids                             = {10, 11, 12, 13, 14, 15};
    prompt.identity.rewrite_execution_frontiers = {2, 5};
    q36::detail::LedgerSplits splits;
    splits.assign(prompt);
    expect(splits.size() == 6 && splits.frontiers().size() == 2, "prompt splits not assigned");
    splits.append_generated(1);
    splits.append_generated(3, 2U);
    const std::array<std::uint32_t, 3> expected{2, 5, 9};
    expect(splits.size() == 10 &&
               std::equal(splits.frontiers().begin(), splits.frontiers().end(), expected.begin(),
                          expected.end()),
           "generated split was not placed after its span's prefix");
    bool rejected = false;
    try {
        splits.append_generated(2, 3U);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, "a split past its appended span was accepted");
    splits.truncate(7);
    expect(splits.size() == 7 && splits.frontiers().size() == 2 && splits.frontiers().back() == 5,
           "truncation kept a split past the retained ledger");
    rejected = false;
    try {
        splits.truncate(8);
    } catch (const std::out_of_range&) { rejected = true; }
    expect(rejected, "truncation extended the ledger");
}

} // namespace

int main() {
    test_execution_timing_domains();
    test_decoder_layout();
    test_round_layout();
    test_mtp_alignment();
    test_vision_control();
    test_ledger_splits();
    if (failures != 0) {
        std::cerr << failures << " Qwen3.6 runtime mechanism checks failed\n";
        return 1;
    }
    std::cout << "Qwen3.6 runtime mechanism checks passed\n";
    return 0;
}
