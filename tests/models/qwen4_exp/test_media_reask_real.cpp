// A new conversation that asks about the same image resumes past it (candidates QC-01): the Frontend places a
// structural prefix-cache tap where the prompt's last media item ends, so the second chat reuses the prompt up to
// there and its Vision pass encodes nothing (every item ends inside the reused prefix). Before, the first tap past
// the image was the user message's end, after the question, which the second chat does not share.
// Default route, prefix reuse on. The artifact must carry the vision tower. Skips (77) unless
// INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume).
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer infernix_qwen4_exp_media_reask_real_test

#include "engine_host.h"
#include "infernix/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> gradient_ppm(int size, int phase) {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n" + std::to_string(size) + ' ' + std::to_string(size) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < size * size; ++index) {
        ppm.push_back(static_cast<std::uint8_t>((index + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>(((index / size) * 3 + phase) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7 + phase * 5) & 0xff));
    }
    return ppm;
}

infernix::PromptInput ask(const std::string& question) {
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    infernix::MessagePart media;
    media.kind              = infernix::MessagePartKind::Media;
    media.media.kind        = infernix::MediaKind::Image;
    media.media.bytes       = gradient_ppm(1024, 17); // 1,024 merged tokens
    media.media.media_type  = "image/x-portable-pixmap";
    media.media.source_name = "screenshot.ppm";
    user.parts.push_back(std::move(media));
    user.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = question, .media = {}});
    infernix::PromptInput input;
    input.messages.push_back(std::move(user));
    input.options.enable_thinking = false;
    return input;
}

infernix::RequestOptions greedy() {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = 16;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    int failures      = 0;
    try {
        infernix::EngineOptions options;
        options.artifact_path = artifact;
        if (ngram != nullptr) { options.ngram_volume_path = ngram; }
        options.enable_vision   = true;
        options.max_context     = 8192;
        options.max_concurrency = 1;
        options.kv_cache        = infernix::KvCacheStorage::Int8Group64;
        infernix::Engine engine(options, infernix::test::engine_host());

        const infernix::GenerationResult first  = engine.generate(engine.prepare(ask("Describe the colours.")), greedy());
        const infernix::GenerationResult second = engine.generate(engine.prepare(ask("What shapes do you see?")), greedy());
        std::printf("        first: reused %u, vision %.3f s; second: reused %u, vision %.3f s\n",
                    first.reused_prompt_tokens, first.timings.vision_seconds, second.reused_prompt_tokens,
                    second.timings.vision_seconds);
        failures += check(first.timings.vision_seconds > 0.0, "the first chat encodes the image");
        failures += check(second.reused_prompt_tokens >= 1024, "the second chat resumes past the image (" +
                                                                   std::to_string(second.reused_prompt_tokens) +
                                                                   " prompt tokens reused)");
        failures += check(second.timings.vision_seconds == 0.0, "the second chat encodes no image");
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures == 0 ? "media re-ask checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
