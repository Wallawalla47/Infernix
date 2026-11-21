// Inkling 2's load plan on a real artifact (CPU only, nothing materialized): the config parses, every parameter the
// loader declares binds with the converter's name, shape, format and Use policy, the tokenizer builds on the Inkling
// pipeline over exactly the model's vocabulary, and the plan's device bytes are the artifact's tensor bytes.
//
//   INFERNIX_INKLING2_ARTIFACT   an artifact converted with the inkling2_mini recipe (tools/convert/inkling2_loop.py)

#include "artifact/reader.h"
#include "models/inkling2/load.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    const char* path = std::getenv("INFERNIX_INKLING2_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::cout << "skip: INFERNIX_INKLING2_ARTIFACT is not set\n";
        return 77;
    }
    try {
        using namespace infernix;
        const artifact::Reader reader(path);
        require(models::inkling2::is_inkling2(reader), "the artifact is not Inkling2Loop");
        const auto plan    = models::inkling2::plan_load(reader, {});
        const auto& config = plan.config().text;
        require(config.num_hidden_layers == 24 && config.visits() == 48 && config.vocab_size == 200058,
                "unexpected Inkling 2 Mini dimensions");
        require(config.visit_layer(5) == 5 && config.visit_layer(6) == 6 && config.visit_layer(18) == 6 &&
                    config.visit_layer(41) == 17 && config.visit_layer(42) == 18 && config.visit_layer(47) == 23 &&
                    config.visit_pass(18) == 1 && config.visit_pass(41) == 2 && config.visit_pass(42) == 0,
                "the visit schedule is not the loop's");
        std::size_t nvfp4 = 0;
        for (const auto format : config.expert_formats) {
            nvfp4 += format == models::inkling2::ExpertFormat::Nvfp4FourOverSix ? 1 : 0;
        }
        require(nvfp4 == 20, "20 NVFP4 expert layers expected");
        require(plan.public_token_count() == config.vocab_size, "the tokenizer does not cover the vocabulary");
        const auto ids = plan.resources().tokenizer->encode("The capital of France is");
        require(!ids.empty(), "the tokenizer encodes nothing");
        std::cout << "plan: " << plan.device_bytes() / (1024.0 * 1024 * 1024) << " GiB device, " << ids.size()
                  << " tokens for the probe\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
