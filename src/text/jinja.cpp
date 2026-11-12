#include "text/jinja.h"

#include "text/unicode.h"

#include <jinja/parser.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <unordered_map>

namespace infernix::text {
namespace {

// Interpreter error decoration must not change request cancellation/deadline errors.
struct Interrupted {
    std::exception_ptr error;
};

// Appends one JSON Pointer reference token: "~" and "/" are escaped as "~0" and "~1".
void append_pointer_component(std::string& pointer, std::string_view component) {
    pointer += '/';
    for (const char c : component) {
        if (c == '~')
            pointer += "~0";
        else if (c == '/')
            pointer += "~1";
        else
            pointer += c;
    }
}

using Tags = std::unordered_map<std::string, std::uint32_t>;

// `pointer` holds the input's JSON Pointer while it is converted; it is only maintained when the
// render tags input regions.
jinja::value convert(const nlohmann::ordered_json& input, std::string& pointer, const Tags& tags,
                     bool literal) {
    using namespace jinja;
    value result;
    if (input.is_null())
        result = mk_val<value_none>();
    else if (input.is_boolean())
        result = mk_val<value_bool>(input.get<bool>());
    else if (input.is_number_float())
        result = mk_val<value_float>(input.get<double>());
    else if (input.is_number_unsigned()) {
        const auto number = input.get<std::uint64_t>();
        if (number > INT64_MAX) throw std::invalid_argument("Template integer exceeds int64");
        result = mk_val<value_int>(static_cast<std::int64_t>(number));
    } else if (input.is_number_integer())
        result = mk_val<value_int>(input.get<std::int64_t>());
    else if (input.is_string())
        result = mk_val<value_string>(string(input.get_ref<const std::string&>(), 0, literal));
    else if (input.is_array()) {
        auto array        = mk_val<value_array>();
        const auto prefix = pointer.size();
        array->val_arr.reserve(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            if (!tags.empty()) append_pointer_component(pointer, std::to_string(i));
            array->push_back(convert(input[i], pointer, tags, literal));
            pointer.resize(prefix);
        }
        result = std::move(array);
    } else if (input.is_object()) {
        auto object       = mk_val<value_object>();
        const auto prefix = pointer.size();
        object->val_obj.reserve(input.size());
        for (const auto& item : input.items()) {
            if (!tags.empty()) append_pointer_component(pointer, item.key());
            object->insert(mk_val<value_string>(string(item.key(), 0, literal)),
                           convert(item.value(), pointer, tags, literal));
            pointer.resize(prefix);
        }
        result = std::move(object);
    } else
        throw std::invalid_argument("Unsupported template JSON value");
    if (!tags.empty()) {
        if (const auto it = tags.find(pointer); it != tags.end()) {
            result->origin = it->second;
            if (is_val<value_string>(result)) result->val_str.tag(it->second);
        }
    }
    return result;
}

} // namespace

struct JinjaTemplate::Impl {
    std::string source_name;
    std::shared_ptr<const std::string> source;
    jinja::program program;

    Impl(std::string text, std::string name) : source_name(std::move(name)) {
        try {
            unicode_internal::utf8_codepoints(text, "chat template");
            auto tokens   = jinja::lexer().tokenize(text);
            source        = std::make_shared<const std::string>(std::move(tokens.source));
            tokens.source = *source;
            program       = jinja::parse_from_tokens(tokens);
        } catch (const std::exception& error) {
            throw std::invalid_argument(source_name + ": " + error.what());
        }
    }
};

JinjaTemplate::JinjaTemplate(std::string source, std::string source_name)
    : impl_(std::make_shared<Impl>(std::move(source), std::move(source_name))) {}

TemplateOutput JinjaTemplate::render(const nlohmann::ordered_json& input,
                                     const TemplateRenderOptions& options) const {
    if (!input.is_object()) throw std::invalid_argument("Template context must be an object");
    if (options.checkpoint) options.checkpoint();
    try {
        jinja::context context(impl_->source);
        auto& state        = context.state();
        state.current_time = options.timestamp;
        std::size_t steps  = 0;
        if (options.checkpoint) {
            state.checkpoint = [&] {
                if ((++steps & 255) != 0) return;
                try {
                    options.checkpoint();
                } catch (...) { throw Interrupted{std::current_exception()}; }
            };
        }
        Tags tags;
        for (const auto& region : options.regions) tags.emplace(region.pointer, region.tag);
        std::string pointer;
        for (const auto& item : input.items()) {
            const bool literal =
                std::find(options.control_variables.begin(), options.control_variables.end(),
                          item.key()) == options.control_variables.end();
            pointer.clear();
            if (!tags.empty()) append_pointer_component(pointer, item.key());
            context.set_val(item.key(), convert(item.value(), pointer, tags, literal));
        }
        const jinja::string rendered = jinja::runtime(context).render(impl_->program);
        TemplateOutput result;
        result.text.reserve(rendered.byte_size());
        for (const auto& part : rendered.parts) {
            const auto begin = result.text.size();
            if (part.collapsed_source_end) {
                result.boundary_mappings.push_back(
                    {part.origin, *part.source_offset, *part.collapsed_source_end, begin});
                continue;
            }
            result.text += part.val;
            if (part.literal && !part.val.empty()) {
                if (!result.literal_spans.empty() && result.literal_spans.back().end == begin)
                    result.literal_spans.back().end = result.text.size();
                else
                    result.literal_spans.push_back({begin, result.text.size()});
            }
            if (part.origin)
                result.regions.push_back(
                    {part.origin, begin, result.text.size(), part.source_offset});
        }
        if (state.checkpoint) {
            try {
                options.checkpoint();
            } catch (...) { throw Interrupted{std::current_exception()}; }
        }
        return result;
    } catch (const Interrupted& interrupted) {
        std::rethrow_exception(interrupted.error);
    } catch (const std::exception& error) {
        throw std::invalid_argument(impl_->source_name + ": " + error.what());
    }
}

} // namespace infernix::text
