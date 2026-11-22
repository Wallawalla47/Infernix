// Inkling 2's image input through the public Engine on the vision artifact (--components text,dspark,vision):
// 1. Answer: a red disc on a blue ground asked for the disc's colour answers "red" (greedy; reported and judged: a
//    tower that lost the picture answers at random), at 5 x 4 cells and at 48 x 27 (1,296 image tokens: the tower's
//    chunk loops, and at prefill chunks of 512 the image across three chunks, the same tokens).
// 2. Request invariance with images: two image requests and a text request, sampled with seeds, give their solo tokens
//    all at once in reverse order (C = 3), and the greedy image request its tokens at prefill chunks of 16 (the image
//    tokens split across chunk edges) and in one call, eager and with graphs.
// 3. DSpark (--spec dspark): the greedy image request's tokens equal plain decoding's.
// 4. --vision-offload on (the tower streamed from pinned host memory): the greedy small and large image requests and
//    a sampled one give the resident tower's tokens.
// 5. Without --vision the same artifact refuses an image request (InvalidMedia) and serves text.
//
//   INFERNIX_INKLING2_VISION_ARTIFACT  the artifact converted with the vision component

#include "infernix/engine.h"
#include "media/decode/decode.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Tokens = std::vector<infernix::TokenId>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

void require_same(const std::string& label, const Tokens& a, const Tokens& b) {
    std::size_t at = 0;
    while (at < a.size() && at < b.size() && a[at] == b[at]) { ++at; }
    require(!a.empty() && a == b, label + ": tokens differ at " + std::to_string(at) + " of " + std::to_string(a.size()) +
                                      "/" + std::to_string(b.size()));
    std::cout << label << ": " << a.size() << " tokens identical\n";
}

infernix::EngineOptions engine_options(const std::filesystem::path& artifact, std::uint32_t chunk, bool graphs,
                                       std::uint32_t concurrency, bool vision = true) {
    infernix::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 2048; // the large image's 1,328-token prompt
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(concurrency * 2048);
    options.max_concurrency      = concurrency;
    options.max_pending_requests = 16;
    // The prefix cache stays on (the Engine's default): requests sharing a prompt resume from it, which must not change
    // their tokens; a small Host tier keeps the start quick.
    options.context_cache.host_capacity_bytes = std::size_t{2} << 30U;
    options.prefill_chunk        = chunk;
    options.use_cuda_graph       = graphs;
    options.enable_vision        = vision;
    options.diagnostic_observer.callback = [](const infernix::Diagnostic& d) {
        if (d.level != infernix::DiagnosticLevel::Debug) { std::cerr << "[engine] " << d.message << '\n'; }
    };
    return options;
}

// Raw RGB in Infernix's container (media/decode/decode.h): `paint(x, y, rgb)` colours each pixel.
template <class Paint>
std::vector<std::uint8_t> picture(int width, int height, Paint paint) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) { paint(x, y, rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3); }
    }
    return infernix::media::decode::encode_raw_rgb(width, height, rgb);
}

std::vector<std::uint8_t> red_disc() { // 200 x 140: 5 x 4 cells
    return picture(200, 140, [](int x, int y, std::uint8_t* p) {
        const bool disc = (x - 120) * (x - 120) + (y - 65) * (y - 65) < 45 * 45;
        p[0] = disc ? 220 : 30, p[1] = disc ? 30 : 60, p[2] = disc ? 30 : 200;
    });
}

std::vector<std::uint8_t> large_red_disc() { // 1915 x 1079: 48 x 27 = 1,296 cells, past the tower's 4,096-token chunks
    return picture(1915, 1079, [](int x, int y, std::uint8_t* p) {
        const bool disc = (x - 1200) * (x - 1200) + (y - 500) * (y - 500) < 380 * 380;
        p[0] = disc ? 220 : 30, p[1] = disc ? 30 : 60, p[2] = disc ? 30 : 200;
    });
}

std::vector<std::uint8_t> stripes() { // 300 x 90: 8 x 3 cells
    return picture(300, 90, [](int x, int, std::uint8_t* p) {
        const bool green = (x / 30) % 2 == 0;
        p[0] = green ? 40 : 250, p[1] = green ? 180 : 250, p[2] = green ? 60 : 250;
    });
}

infernix::PromptInput ask(const std::vector<std::uint8_t>& image, const std::string& text) {
    infernix::PromptInput input;
    infernix::ChatMessage user;
    user.role = infernix::ChatRole::User;
    if (!image.empty()) {
        infernix::MessagePart part{.kind = infernix::MessagePartKind::Media};
        part.media.bytes = image;
        user.parts.push_back(std::move(part));
    }
    user.parts.push_back({.text = text});
    input.messages.push_back(std::move(user));
    input.options.reasoning_effort = infernix::ReasoningEffort::Low;
    return input;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    return options;
}

infernix::RequestOptions sampled(std::size_t i) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(32 + 8 * i);
    options.execution.sampling.temperature    = 1.0F;
    options.execution.sampling.top_k          = 20;
    options.execution.sampling.seed           = 2000 + i;
    options.stop.include_model_defaults       = false;
    options.output.raw                        = true;
    return options;
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_INKLING2_VISION_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: set INFERNIX_INKLING2_VISION_ARTIFACT\n";
        return 77;
    }
    try {
        const std::vector<infernix::PromptInput> prompts = {
            ask(red_disc(), "What colour is the circle in this image? Answer with one word."),
            ask(stripes(), "Describe this image in one sentence."),
            ask({}, "Name three rivers in Europe."),
        };
        const infernix::PromptInput large =
            ask(large_red_disc(), "What colour is the circle in this image? Answer with one word.");
        std::vector<Tokens> solo;
        Tokens disc_greedy, large_greedy;
        {
            infernix::Engine engine(engine_options(artifact, 0, true, 1));
            const auto answer = engine.generate(engine.prepare(prompts[0]), greedy(200));
            std::string lower = answer.content;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
            std::cout << "answer: \"" << answer.content << "\" (" << answer.generated_token_ids.size() << " tokens, "
                      << answer.prompt.prompt_tokens << " prompt tokens)\n";
            require(lower.find("red") != std::string::npos, "the red disc's colour is not named red");
            disc_greedy = answer.generated_token_ids;
            const auto big = engine.generate(engine.prepare(large), greedy(200));
            lower = big.content;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
            std::cout << "large image answer: \"" << big.content << "\" (" << big.prompt.prompt_tokens << " prompt tokens)\n";
            require(lower.find("red") != std::string::npos, "the large red disc's colour is not named red");
            large_greedy = big.generated_token_ids;
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                solo.push_back(engine.generate(engine.prepare(prompts[i]), sampled(i)).generated_token_ids);
            }
        }
        {
            infernix::Engine engine(engine_options(artifact, 0, true, 3));
            std::vector<infernix::GenerationHandle> handles;
            for (std::size_t i = prompts.size(); i-- > 0;) { handles.push_back(engine.submit(engine.prepare(prompts[i]), sampled(i))); }
            for (std::size_t j = 0; j < handles.size(); ++j) {
                const std::size_t i = prompts.size() - 1 - j;
                require_same("C = 3, request " + std::to_string(i), solo[i], handles[j].wait().generated_token_ids);
            }
        }
        {
            infernix::Engine narrow(engine_options(artifact, 512, true, 1));
            require_same("large image, chunks of 512", large_greedy,
                         narrow.generate(narrow.prepare(large), greedy(200)).generated_token_ids);
        }
        for (const bool graphs : {false, true}) {
            for (const std::uint32_t chunk : {16U, 4096U}) {
                infernix::Engine engine(engine_options(artifact, chunk, graphs, 1));
                require_same(std::string(graphs ? "graphs" : "eager") + ", chunks of " + std::to_string(chunk), disc_greedy,
                             engine.generate(engine.prepare(prompts[0]), greedy(200)).generated_token_ids);
            }
        }
        {
            infernix::EngineOptions options     = engine_options(artifact, 0, true, 1);
            options.speculative.backend           = infernix::SpeculativeBackend::DSpark;
            options.speculative.backend_auto      = false;
            options.speculative.draft_tokens_auto = true;
            options.speculative.draft_tokens      = 7;
            infernix::Engine engine(options);
            const auto drafted = engine.generate(engine.prepare(prompts[0]), greedy(200));
            require_same("DSpark", disc_greedy, drafted.generated_token_ids);
        }
        {
            infernix::EngineOptions options = engine_options(artifact, 0, true, 1);
            options.vision_offload          = infernix::VisionOffload::On;
            infernix::Engine engine(options);
            require_same("--vision-offload on", disc_greedy,
                         engine.generate(engine.prepare(prompts[0]), greedy(200)).generated_token_ids);
            require_same("--vision-offload on, large image", large_greedy,
                         engine.generate(engine.prepare(large), greedy(200)).generated_token_ids);
            require_same("--vision-offload on, sampled stripes", solo[1],
                         engine.generate(engine.prepare(prompts[1]), sampled(1)).generated_token_ids);
        }
        {
            infernix::Engine engine(engine_options(artifact, 0, true, 1, false));
            bool refused = false;
            try {
                (void)engine.prepare(prompts[0]);
            } catch (const infernix::RequestError& error) {
                refused = error.kind() == infernix::RequestErrorKind::InvalidMedia;
            }
            require(refused, "an Engine without --vision accepted an image");
            require(!engine.generate(engine.prepare(prompts[2]), greedy(16)).generated_token_ids.empty(),
                    "an Engine without --vision serves text");
            std::cout << "without --vision: images refused, text served\n";
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
