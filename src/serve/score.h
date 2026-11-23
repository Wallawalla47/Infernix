#pragma once

// POST /v1/score: teacher-forced log-probabilities of a given continuation after a given prompt,
// with the model's own top_k next-token distribution at every scored position
// (Engine::score_tokens; docs/serving.md, "Teacher-forced scoring"). Nothing is generated. It needs an Engine
// started for causal scoring (infernix-serve --causal-scoring); a generation server refuses it.
//
// The chat template is the caller's explicit choice and the response states it: a distribution
// measured on an untemplated prompt is not the one /v1/chat/completions samples from.

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace infernix::serve {

struct ScoreRequest {
    std::string model;
    // apply_chat_template: messages (or prompt as one user message) rendered exactly as
    // /v1/chat/completions renders them, generation prompt included. Otherwise prompt is
    // encoded as raw text with no template or special token added.
    bool apply_chat_template = false;
    std::string prompt;
    // The prompt's turns and template options (enable_thinking, chat_template_kwargs,
    // reasoning_effort) under apply_chat_template, parsed by the Chat Completions parser.
    GenerationRequest chat;
    // Encoded as raw text after the prompt; every one of its tokens is a scored position.
    std::string continuation;
    // The top_k most probable next tokens per scored position, in [0,kMaximumScoreTopTokens].
    std::uint32_t top_k = 0;
    // Every layer's hidden state at the prompt's last token (ScoreOptions::prompt_hidden_states).
    bool hidden_states = false;
};

struct ScoredAlternative {
    infernix::TokenId id = 0;
    std::string text;
    float logprob = 0.0F;
};

struct ScoredPosition {
    infernix::TokenId id = 0;
    std::string text;
    float logprob = 0.0F;
    std::vector<ScoredAlternative> top; // descending
};

struct ScoreOutcome {
    bool chat_template_applied = false;
    // The template's thinking switch as rendered; unset without a template.
    std::optional<bool> enable_thinking;
    std::uint32_t top_k            = 0;
    std::uint32_t prompt_tokens    = 0;
    std::vector<ScoredPosition> positions;
    // ScoreRequest::hidden_states: [layers][hidden_size] layer-major (ScoreResult::hidden_states); empty otherwise.
    std::vector<float> hidden_states;
    std::uint32_t hidden_state_layers = 0;
};

// Validates the whole body before anything runs: an unknown field, a missing choice or an
// out-of-range top_k is a 400, never a narrowed answer.
[[nodiscard]] ScoreRequest parse_score_request(const RequestJson& body);

[[nodiscard]] RequestJson make_score_response(const std::string& model, const ScoreOutcome& outcome);

} // namespace infernix::serve
