// Chat-template render cost over synthetic agent histories (CPU only).
//
// For each template and history size it reports the median wall time of one Jinja render of the
// request's full template context, and of one whole CompiledChatTemplate::render request (the
// prompt render plus the boundary probes the request performs). --dump writes every render's
// complete output (text, literal spans, input regions, boundary mappings; for requests also the
// message, cache and rewrite boundaries) so two builds can be compared byte for byte.

#include "models/qwen3_5/frontend/chat_template.h"
#include "text/jinja.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fi = infernix::models::qwen3_5::frontend;
using Json   = nlohmann::ordered_json;
using Clock  = std::chrono::steady_clock;

struct Rng {
    std::uint64_t state;

    std::uint64_t next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return state >> 33U;
    }

    std::size_t below(std::size_t bound) { return static_cast<std::size_t>(next() % bound); }

    bool chance(unsigned percent) { return below(100) < percent; }
};

const char* const kWords[] = {
    "the",      "parser", "returns",  "value",   "cache",   "token",  "boundary", "message",
    "frontend", "render", "template", "request", "context", "prefix", "state",    "lane",
    "kernel",   "column", "update",   "result",  "branch",  "commit", "schedule", "output",
    "naïve",    "größe",  "北京",     "测试",    "🌏",      "→",      "café",     "λ"};

std::string words(Rng& rng, std::size_t count) {
    std::string out;
    for (std::size_t i = 0; i < count; ++i) {
        if (i) out += (rng.chance(8) ? "\n" : " ");
        out += kWords[rng.below(std::size(kWords))];
    }
    return out;
}

std::string code_lines(Rng& rng, std::size_t lines, std::size_t first_line) {
    std::string out;
    for (std::size_t i = 0; i < lines; ++i) {
        char number[16];
        std::snprintf(number, sizeof(number), "%6zu\t", first_line + i);
        out += number;
        out += std::string(4 * rng.below(4), ' ');
        switch (rng.below(5)) {
        case 0:
            out += "if (" + words(rng, 2) + ") {";
            break;
        case 1:
            out += "const auto value = compute(\"" + words(rng, 2) + "\", " +
                   std::to_string(rng.below(1000)) + ");";
            break;
        case 2:
            out += "// " + words(rng, 6);
            break;
        case 3:
            out += "return {\"key\": " + std::to_string(rng.below(99)) + "};";
            break;
        default:
            out += "}";
            break;
        }
        out += '\n';
    }
    return out;
}

std::vector<std::string> tool_jsons() {
    struct Tool {
        const char* name;
        const char* description;
        Json properties;
        std::vector<std::string> required;
    };

    const Json string_type{{"type", "string"}};
    const std::vector<Tool> tools{
        {"read_file",
         "Reads a file from the local filesystem. Returns numbered lines.",
         {{"path", {{"type", "string"}, {"description", "The absolute path to the file."}}},
          {"offset", {{"type", "integer"}, {"description", "The line to start from."}}},
          {"limit", {{"type", "integer"}, {"description", "The number of lines to read."}}}},
         {"path"}},
        {"write_file",
         "Writes a file to the local filesystem, overwriting an existing file.",
         {{"path", string_type}, {"content", string_type}},
         {"path", "content"}},
        {"edit_file",
         "Performs exact string replacement in a file. old_string must be unique.",
         {{"path", string_type},
          {"old_string", string_type},
          {"new_string", string_type},
          {"replace_all", {{"type", "boolean"}, {"default", false}}}},
         {"path", "old_string", "new_string"}},
        {"bash",
         "Executes a shell command and returns its output. Quote paths with spaces.",
         {{"command", string_type},
          {"timeout", {{"type", "number"}, {"description", "Milliseconds, at most 600000."}}},
          {"run_in_background", {{"type", "boolean"}}}},
         {"command"}},
        {"grep",
         "Searches file contents with a regular expression (ripgrep syntax).",
         {{"pattern", string_type},
          {"path", string_type},
          {"glob", string_type},
          {"output_mode",
           {{"type", "string"}, {"enum", Json::array({"content", "files_with_matches", "count"})}}},
          {"-n", {{"type", "boolean"}}},
          {"head_limit", {{"type", "number"}}}},
         {"pattern"}},
        {"glob",
         "Fast file pattern matching. Returns matching paths by modification time.",
         {{"pattern", string_type}, {"path", string_type}},
         {"pattern"}},
        {"list_dir", "Lists a directory.", {{"path", string_type}}, {"path"}},
        {"todo_write",
         "Updates the task list shown to the user.",
         {{"todos",
           {{"type", "array"},
            {"items",
             {{"type", "object"},
              {"properties",
               {{"content", string_type},
                {"status",
                 {{"type", "string"},
                  {"enum", Json::array({"pending", "in_progress", "completed"})}}}}},
              {"required", Json::array({"content", "status"})}}}}}},
         {"todos"}},
        {"web_fetch",
         "Fetches a URL and answers a question about its content.",
         {{"url", {{"type", "string"}, {"format", "uri"}}}, {"prompt", string_type}},
         {"url", "prompt"}},
        {"task",
         "Launches a sub-agent for a multi-step task.",
         {{"description", string_type}, {"prompt", string_type}, {"subagent_type", string_type}},
         {"description", "prompt"}},
        {"notebook_edit",
         "Replaces the contents of one notebook cell.",
         {{"notebook_path", string_type},
          {"cell_id", string_type},
          {"new_source", string_type},
          {"edit_mode",
           {{"type", "string"}, {"enum", Json::array({"replace", "insert", "delete"})}}}},
         {"notebook_path", "new_source"}},
        {"ask_user",
         "Asks the user a multiple-choice question.",
         {{"question", string_type},
          {"options", {{"type", "array"}, {"items", string_type}, {"minItems", 2}}}},
         {"question", "options"}},
    };
    std::vector<std::string> result;
    for (const auto& tool : tools) {
        Json parameters{{"type", "object"}, {"properties", tool.properties}};
        parameters["required"] = tool.required;
        result.push_back(Json{{"type", "function"},
                              {"function",
                               {{"name", tool.name},
                                {"description", tool.description},
                                {"parameters", std::move(parameters)}}}}
                             .dump());
    }
    return result;
}

fi::ChatMessage text_message(infernix::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::ToolCall make_call(Rng& rng, std::size_t id) {
    fi::ToolCall call;
    call.id = "call_" + std::to_string(id);
    Json arguments;
    switch (rng.below(5)) {
    case 0:
        call.name = "read_file";
        arguments = {{"path", "/work/src/module_" + std::to_string(rng.below(90)) + ".cpp"},
                     {"offset", rng.below(400)},
                     {"limit", 200}};
        break;
    case 1:
        call.name = "edit_file";
        arguments = {{"path", "/work/src/module_" + std::to_string(rng.below(90)) + ".cpp"},
                     {"old_string", code_lines(rng, 1 + rng.below(4), 1)},
                     {"new_string", code_lines(rng, 1 + rng.below(5), 1)},
                     {"replace_all", rng.chance(10)}};
        break;
    case 2:
        call.name = "bash";
        arguments = {{"command", "cmake --build build -j && ctest -R " + words(rng, 1)},
                     {"timeout", 600000}};
        break;
    case 3:
        call.name = "grep";
        arguments = {{"pattern", "render\\(|" + words(rng, 1)},
                     {"glob", "*.cpp"},
                     {"output_mode", "content"},
                     {"-n", true}};
        break;
    default:
        call.name = "todo_write";
        arguments = {{"todos", Json::array({{{"content", words(rng, 5)}, {"status", "completed"}},
                                            {{"content", words(rng, 4)}, {"status", "in_progress"}},
                                            {{"content", words(rng, 6)}, {"status", "pending"}}})}};
        break;
    }
    call.arguments_json = arguments.dump();
    return call;
}

std::string tool_result(Rng& rng) {
    switch (rng.below(4)) {
    case 0:
        return code_lines(rng, 4 + rng.below(30), 1 + rng.below(300));
    case 1:
        return "The file was updated successfully.\n";
    case 2:
        return words(rng, 10 + rng.below(80)) + "\n";
    default:
        return "{\"exit_code\": 0, \"stdout\": \"" + words(rng, 12) + "\"}";
    }
}

// An agent conversation of exactly `count` messages: a leading System turn, the user's task, then
// assistant turns that call one or two tools (each followed by its results), with an occasional
// finished answer and a new user request. It ends with a tool result, as an agent loop does.
std::vector<fi::ChatMessage> agent_history(std::size_t count, std::uint64_t seed) {
    if (count < 4) throw std::invalid_argument("history needs at least four messages");
    Rng rng{seed};
    std::vector<fi::ChatMessage> messages;
    std::string system = "  You are a coding agent working in a local repository.\n\n";
    for (int i = 0; i < 40; ++i) system += "- " + words(rng, 12) + "\n";
    messages.push_back(text_message(infernix::ChatRole::System, system + "\n"));
    messages.push_back(
        text_message(infernix::ChatRole::User, "Fix the failing test: " + words(rng, 20)));
    std::size_t next_call  = 0;
    const auto calls_block = [&](std::size_t calls) {
        fi::ChatMessage assistant;
        assistant.role = infernix::ChatRole::Assistant;
        assistant.parts.push_back(
            fi::ChatPart::text_part(rng.chance(40) ? words(rng, 8 + rng.below(20)) : ""));
        assistant.reasoning_content = "\n" + words(rng, 20 + rng.below(60)) + "\n";
        for (std::size_t i = 0; i < calls; ++i)
            assistant.tool_calls.push_back(make_call(rng, next_call++));
        const auto ids = assistant.tool_calls;
        messages.push_back(std::move(assistant));
        for (const auto& call : ids) {
            auto result         = text_message(infernix::ChatRole::Tool, tool_result(rng));
            result.tool_call_id = call.id;
            messages.push_back(std::move(result));
        }
    };
    while (count - messages.size() > 3) {
        const auto remaining = count - messages.size();
        const auto kind      = rng.below(20);
        if (kind == 0 && remaining >= 6) {
            fi::ChatMessage answer =
                text_message(infernix::ChatRole::Assistant, "Done. " + words(rng, 30) + "\n");
            answer.reasoning_content = words(rng, 40);
            messages.push_back(std::move(answer));
            messages.push_back(text_message(infernix::ChatRole::User, "  Also " + words(rng, 15)));
        } else if (kind < 5 && remaining >= 5) {
            calls_block(2);
        } else {
            calls_block(1);
        }
    }
    calls_block(count - messages.size() - 1);
    if (messages.size() != count) throw std::logic_error("history size mismatch");
    return messages;
}

const nlohmann::ordered_json kSpecialTokens{{"eos_token", "<|im_end|>"},
                                            {"pad_token", "<|endoftext|>"}};

// The template context CompiledChatTemplate builds for a text request (same members and order).
Json request_context(const std::vector<fi::ChatMessage>& messages,
                     const std::vector<std::string>& tools,
                     std::vector<infernix::text::TemplateInputRegion>& regions) {
    Json context = Json::object();
    context.update(kSpecialTokens);
    context["continue_final_message"] = false;
    context["messages"]               = Json::array();
    context["add_generation_prompt"]  = true;
    std::uint32_t tag                 = 1;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& message               = messages[i];
        const auto pointer                = "/messages/" + std::to_string(i);
        static const char* const kRoles[] = {"system", "developer", "user", "assistant", "tool"};
        Json value{{"role", kRoles[static_cast<int>(message.role)]}};
        std::string content;
        for (const auto& part : message.parts) content += part.text;
        value["content"] = content;
        regions.push_back({pointer + "/content", tag++});
        if (!message.reasoning_content.empty()) {
            value["reasoning_content"] = message.reasoning_content;
            regions.push_back({pointer + "/reasoning_content", tag++});
        }
        if (!message.tool_calls.empty()) {
            value["tool_calls"] = Json::array();
            for (const auto& call : message.tool_calls) {
                value["tool_calls"].push_back(
                    {{"id", call.id},
                     {"type", "function"},
                     {"function",
                      {{"name", call.name}, {"arguments", Json::parse(call.arguments_json)}}}});
            }
        }
        if (!message.tool_call_id.empty()) value["tool_call_id"] = message.tool_call_id;
        context["messages"].push_back(std::move(value));
    }
    context["tools"] = Json::array();
    for (std::size_t i = 0; i < tools.size(); ++i) {
        context["tools"].push_back(Json::parse(tools[i]));
        regions.push_back({"/tools/" + std::to_string(i), tag++});
    }
    return context;
}

Json dump_output(const infernix::text::TemplateOutput& output) {
    Json result{{"text", output.text}};
    result["literal_spans"] = Json::array();
    for (const auto& span : output.literal_spans)
        result["literal_spans"].push_back({span.begin, span.end});
    result["regions"] = Json::array();
    for (const auto& region : output.regions) {
        result["regions"].push_back({region.tag, region.begin, region.end,
                                     region.source_offset ? Json(*region.source_offset) : Json()});
    }
    result["boundary_mappings"] = Json::array();
    for (const auto& mapping : output.boundary_mappings) {
        result["boundary_mappings"].push_back(
            {mapping.tag, mapping.source_begin, mapping.source_end, mapping.offset});
    }
    return result;
}

Json optional_json(const std::optional<std::size_t>& value) {
    return value ? Json(*value) : Json();
}

Json dump_chat(const fi::RenderedChat& chat) {
    Json result{{"text", chat.text}, {"continuation_content", chat.continuation_content}};
    result["literal_spans"] = Json::array();
    for (const auto& span : chat.literal_spans)
        result["literal_spans"].push_back({span.begin, span.end});
    result["starts_in_reasoning"] = chat.starts_in_reasoning;
    result["media_placeholders"]  = chat.media_placeholders.size();
    if (chat.rewrite_checkpoint) {
        result["rewrite_checkpoint"] = {static_cast<int>(chat.rewrite_checkpoint->kind),
                                        chat.rewrite_checkpoint->offset,
                                        chat.rewrite_checkpoint->recovery_offset};
    }
    result["rewrite_execution_boundaries"] = chat.rewrite_execution_boundaries;
    result["message_boundaries"]           = Json::array();
    for (const auto& value : chat.message_boundaries)
        result["message_boundaries"].push_back(optional_json(value));
    result["cache_boundaries"] = Json::array();
    for (const auto& value : chat.cache_boundaries)
        result["cache_boundaries"].push_back(optional_json(value));
    return result;
}

template <typename F>
Json guarded(F&& render) {
    try {
        return render();
    } catch (const std::exception& error) { return Json{{"error", error.what()}}; }
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::invalid_argument("cannot read " + path.string());
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

double median_ms(std::size_t repeats, const std::function<void()>& work) {
    work(); // warm-up
    std::vector<double> samples;
    for (std::size_t i = 0; i < repeats; ++i) {
        const auto begin = Clock::now();
        work();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

std::vector<std::size_t> parse_sizes(const std::string& text) {
    std::vector<std::size_t> sizes;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ',');) sizes.push_back(std::stoul(item));
    return sizes;
}

// Request variants for the byte-identity dump, beyond the timed agent request.
struct Variant {
    const char* name;
    std::function<void(std::vector<fi::ChatMessage>&, fi::ChatRenderOptions&)> apply;
};

const std::vector<Variant>& variants() {
    using infernix::ChatRole;
    static const std::vector<Variant> list{
        {"agent", [](auto&, auto&) {}},
        {"no-tools", [](auto&, auto& options) { options.tool_jsons.clear(); }},
        {"no-thinking", [](auto&, auto& options) { options.enable_thinking = false; }},
        {"drop-thinking", [](auto&, auto& options) { options.preserve_thinking = false; }},
        {"keep-thinking", [](auto&, auto& options) { options.preserve_thinking = true; }},
        {"effort-low",
         [](auto&, auto& options) { options.reasoning_effort = infernix::ReasoningEffort::Low; }},
        {"kwargs",
         [](auto&, auto& options) {
             options.chat_template_kwargs_json = R"({"add_vision_id":true,"extra":" <|im_end|> "})";
         }},
        {"user-turn",
         [](auto& messages, auto&) {
             messages.push_back(text_message(ChatRole::User, " next question \n"));
         }},
        {"late-system",
         [](auto& messages, auto&) {
             messages.push_back(text_message(ChatRole::System, "diagnostic"));
             messages.push_back(text_message(ChatRole::User, "continue"));
         }},
        {"continue",
         [](auto& messages, auto& options) {
             messages.push_back(
                 text_message(ChatRole::Assistant, "Partial answer\n  def f():\n    "));
             options.enable_thinking = false;
             options.continuation    = infernix::PromptContinuationMode::ContinueFinalAssistant;
         }},
        {"next-turn-text",
         [](auto& messages, auto& options) {
             fi::ChatMessage reply   = text_message(ChatRole::Assistant, "The answer.");
             reply.reasoning_content = "thinking";
             messages.push_back(std::move(reply));
             options.continuation = infernix::PromptContinuationMode::NextTurnHistory;
         }},
        {"next-turn-tools",
         [](auto& messages, auto& options) {
             fi::ChatMessage reply = text_message(ChatRole::Assistant, "");
             reply.tool_calls.push_back({"call_next", "bash", R"({"command":"ls"})"});
             messages.push_back(std::move(reply));
             options.continuation = infernix::PromptContinuationMode::NextTurnHistory;
         }},
    };
    return list;
}

std::vector<infernix::PromptCacheMarker> agent_markers(std::size_t count, std::size_t tools) {
    using infernix::PromptCacheMarker;
    using infernix::PromptCacheMarkerLocation;
    using infernix::SharedCandidateEvidence;
    std::vector<PromptCacheMarker> markers;
    markers.push_back({.after_message_count = static_cast<std::uint32_t>(count - 2)});
    markers.push_back({.after_message_count = static_cast<std::uint32_t>(count)});
    markers.push_back(
        {.after_message_count = 1, .evidence = SharedCandidateEvidence::DefaultAutomatic});
    if (tools) {
        markers.push_back({.evidence         = SharedCandidateEvidence::EngineStructural,
                           .location         = PromptCacheMarkerLocation::ToolBoundary,
                           .after_tool_count = static_cast<std::uint32_t>(tools)});
    }
    return markers;
}

int usage() {
    std::cerr << "usage: infernix_chat_template_render_bench [--template NAME=FILE]... "
                 "[--sizes 100,300,1000,1600] [--repeats 7] [--dump DIR] [--variant-size 40]\n"
                 "Without --template, the maintained qwen3_6 and qwen3_8 templates of the source "
                 "tree are rendered.\n";
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::pair<std::string, std::filesystem::path>> templates;
    std::vector<std::size_t> sizes{100, 300, 1000, 1600};
    std::size_t repeats      = 7;
    std::size_t variant_size = 40;
    std::optional<std::filesystem::path> dump;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) return usage();
        const std::string value = argv[++i];
        if (arg == "--template") {
            const auto equals = value.find('=');
            if (equals == std::string::npos) return usage();
            templates.emplace_back(value.substr(0, equals), value.substr(equals + 1));
        } else if (arg == "--sizes") {
            sizes = parse_sizes(value);
        } else if (arg == "--repeats") {
            repeats = std::stoul(value);
        } else if (arg == "--dump") {
            dump = value;
        } else if (arg == "--variant-size") {
            variant_size = std::stoul(value);
        } else {
            return usage();
        }
    }
    if (templates.empty()) {
        for (const char* name : {"qwen3_6", "qwen3_8"}) {
            templates.emplace_back(name, std::filesystem::path(INFERNIX_SOURCE_DIR) / "tools" /
                                             "chat_templates" / (std::string(name) + ".jinja"));
        }
    }
    if (dump) std::filesystem::create_directories(*dump);
    const auto tools = tool_jsons();
    const auto write = [&](const std::string& name, const Json& value) {
        std::ofstream file(*dump / name, std::ios::binary);
        file << value.dump(1, ' ', false, Json::error_handler_t::strict) << '\n';
    };
    try {
        std::cout << "template,messages,prompt_bytes,jinja_render_ms,request_ms,request_per_render,"
                     "context_copy_ms,context_bind_ms\n";
        const infernix::text::JinjaTemplate constant("-", "constant");
        for (const auto& [name, path] : templates) {
            const auto source = read_file(path);
            const auto chat   = fi::CompiledChatTemplate::resolve(source, name, kSpecialTokens);
            const infernix::text::JinjaTemplate jinja(source, name);
            std::vector<std::string> control;
            for (const auto& item : kSpecialTokens.items()) control.push_back(item.key());
            for (const auto size : sizes) {
                const auto messages = agent_history(size, 0x5eed + size);
                std::vector<infernix::text::TemplateInputRegion> regions;
                const auto context = request_context(messages, tools, regions);
                const infernix::text::TemplateRenderOptions render_options{
                    .timestamp = 0, .regions = regions, .control_variables = control};
                fi::ChatRenderOptions options;
                options.tool_jsons    = tools;
                options.cache_markers = agent_markers(size, tools.size());
                const auto rendered   = jinja.render(context, render_options);
                const auto request    = chat.render(messages, options);
                if (rendered.text != request.text)
                    throw std::logic_error(name + ": bench context differs from the request's");
                if (dump) {
                    const auto stem = name + "-" + std::to_string(size);
                    write(stem + ".render.json", dump_output(rendered));
                    write(stem + ".request.json", dump_chat(request));
                }
                if (repeats == 0) continue;
                const double render_ms =
                    median_ms(repeats, [&] { (void)jinja.render(context, render_options); });
                const double request_ms =
                    median_ms(repeats, [&] { (void)chat.render(messages, options); });
                const double copy_ms = median_ms(repeats, [&] {
                    Json copy = context;
                    if (copy.size() == 0) std::abort();
                });
                // A template that reads nothing: the per-render cost of binding the context.
                const double bind_ms =
                    median_ms(repeats, [&] { (void)constant.render(context, render_options); });
                char line[256];
                std::snprintf(line, sizeof(line), "%s,%zu,%zu,%.3f,%.3f,%.2f,%.3f,%.3f\n",
                              name.c_str(), size, rendered.text.size(), render_ms, request_ms,
                              request_ms / render_ms, copy_ms, bind_ms);
                std::cout << line << std::flush;
            }
            if (dump) {
                for (const auto& variant : variants()) {
                    auto messages = agent_history(variant_size, 0xfeed);
                    fi::ChatRenderOptions options;
                    options.tool_jsons = tools;
                    variant.apply(messages, options);
                    options.cache_markers =
                        agent_markers(messages.size(), options.tool_jsons.size());
                    write(name + "-variant-" + variant.name + ".request.json",
                          guarded([&] { return dump_chat(chat.render(messages, options)); }));
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
