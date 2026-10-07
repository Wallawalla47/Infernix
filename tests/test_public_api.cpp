#include "infernix/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<infernix::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<infernix::PreparedPrompt>);
static_assert(std::is_move_constructible_v<infernix::Engine>);
static_assert(!std::is_copy_constructible_v<infernix::Engine>);

int main() {
    const infernix::EngineOptions options;
    return options.enable_vision ? 1 : 0;
}
