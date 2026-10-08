#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::qwen3_5::frontend {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

// Parse Qwen's XML-like tool-call format. In tolerant mode, a complete function call is
// recovered even when the model adds malformed wrapper markup, a trailing suffix, or stops at
// its output budget before the closing tags; the strict parser keeps its all-or-nothing
// behavior.
[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract, bool tolerant = false);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time, constrained output retains completed calls across interruptions;
// free output restores malformed regions verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length, bool tolerant = false);

    [[nodiscard]] bool in_tool_region() const noexcept {
        return saw_tool_marker_ || !pending_tag_.empty();
    }
    [[nodiscard]] std::string feed(std::string_view text);
    void initialize_continuation(std::string_view prefix);
    // `open_reasoning` is the thinking of a turn that ended without closing it. For free tool
    // output in tolerant mode, when the turn ended on a stop token and no content was fed,
    // complete calls stranded at the end of that thinking become the structured turn.
    [[nodiscard]] Terminal finish(FinishReason reason             = FinishReason::StopToken,
                                  std::string_view open_reasoning = {});

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    std::string pending_tag_;
    std::size_t max_tool_name_length_        = 0;
    bool tolerant_                           = false;
    bool saw_tool_marker_                    = false;
    bool fed_content_                        = false;
    bool finished_                           = false;
    std::size_t continuation_withheld_bytes_ = 0;
};

} // namespace infernix::models::qwen3_5::frontend
