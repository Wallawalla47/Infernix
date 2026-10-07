#pragma once

#include "infernix/types.h"

#include <filesystem>
#include <string>

namespace infernix::product {

[[nodiscard]] PromptInput prompt_from_text(std::string text, std::optional<bool> enable_thinking);
[[nodiscard]] PromptInput prompt_from_messages(const std::filesystem::path& path,
                                               std::optional<bool> enable_thinking,
                                               bool vision_enabled);

} // namespace infernix::product
