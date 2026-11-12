#include "text/jinja.h"

#include <atomic>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Json = nlohmann::ordered_json;
using infernix::text::JinjaTemplate;

int check(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << message << '\n';
    return 1;
}

int language_semantics() {
    struct Example {
        std::string source;
        Json input;
        std::string expected;
    };

    const Example cases[]{
        {"{{ missing is undefined }}|{{ value is none }}|{{ value|default('fallback') }}|{{ "
         "value|default('fallback', true) }}",
         {{"value", nullptr}},
         "True|True|None|fallback"},
        {"{{ missing }}|{{ [1, none, true, 'x'] }}|{{ 12 ~ none }}", Json::object(),
         "|[1, None, True, 'x']|12None"},
        {"{{ text|length }}|{{ text[-1] }}|{{ text[::-1] }}|{{ text[:-1:-1] }}|{{ text[99:] }}",
         {{"text", "北京🌏"}},
         "3|🌏|🌏京北||"},
        {"{% for c in text %}[{{ c }}]{% endfor %}", {{"text", "北京🌏"}}, "[北][京][🌏]"},
        {"{{ text|upper }}|{{ text|lower }}|{{ text|capitalize }}",
         {{"text", "straße ΟΣ İ"}},
         "STRASSE ΟΣ İ|straße ος i̇|Straße ος i̇"},
        {"{{ text|trim }}|{{ text.split()|tojson }}|{{ text.rsplit(none,1)|tojson }}",
         {{"text", "\u0085北京\u3000two\tend\u001c"}},
         "北京　two\tend|[\"北京\", \"two\", \"end\"]|[\"\u0085北京　two\", \"end\"]"},
        {"{{ text.replace('', '-', 2) }}|{{ text.replace('京', 'x', 1) }}",
         {{"text", "北京🌏"}},
         "-北-京🌏|北x🌏"},
        {"{{ value|tojson(ensure_ascii=true,sort_keys=true,separators=(',',':')) }}",
         {{"value", {{"z", "🌏"}, {"a", false}}}},
         "{\"a\":false,\"z\":\"\\ud83c\\udf0f\"}"},
        {"{% macro show(x) %}{{ x|default('local') }}{% endmacro %}{% set ns=namespace(n=0) %}{% "
         "for x in [1,2] %}{% set ns.n=ns.n+x %}{% endfor %}{{ ns.n }}|{{ show() }}",
         Json::object(), "3|local"},
        {"{% set x='outer' %}{% macro show(a='A',b='B') %}{{ x }}:{{ a }}:{{ b }}{% endmacro %}{% "
         "for x in ['inner'] %}{{ show(b='Z') }}{% endfor %}",
         Json::object(), "outer:A:Z"},
        {"{% set n=namespace(f=none) %}{% for x in ['captured'] %}{% macro show() %}{{ x }}{% "
         "endmacro %}{% set n.f=show %}{% endfor %}{{ n.f() }}",
         Json::object(), "captured"},
        {"{% macro wrap() %}[{{ caller('value') }}]{% endmacro %}{% call(x) wrap() %}{{ x }}{% "
         "endcall %}",
         Json::object(), "[value]"},
        {"{{ number }}", {{"number", 1.23456789}}, "1.23456789"},
        {"{% filter upper %}{{ text }}{% endfilter %}", {{"text", "straße"}}, "STRASSE"},
    };
    int failures = 0;
    for (const auto& item : cases) {
        const auto actual = JinjaTemplate(item.source, "language-test").render(item.input).text;
        if (actual != item.expected) {
            std::cerr << "Jinja semantic mismatch in " << item.source
                      << "\nexpected: " << Json(item.expected).dump()
                      << "\nactual: " << Json(actual).dump() << '\n';
            ++failures;
        }
    }
    for (const char* source : {"{{ none + 'x' }}", "{{ 'abc'[::0] }}", "{{ missing|tojson }}",
                               "{{ value|unknown_filter }}"}) {
        try {
            (void)JinjaTemplate(source, "error-test").render(Json::object());
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).starts_with("error-test:"),
                              "template error omitted its source name");
        }
    }
    return failures;
}

// Loop control, lazy loops, scopes and integer tests. Every expected text and every refusal is
// Python Jinja2 3.1.6's (plain Environment with jinja2.ext.loopcontrols), except where noted.
int loops_and_scopes() {
    struct Example {
        std::string source;
        Json input;
        std::string expected;
    };

    const Example cases[]{
        // Output written before a {% break %} or {% continue %} inside an if is kept.
        {"{% for i in [1] %}A{% if true %}B{% break %}{% endif %}C{% endfor %}", Json::object(),
         "AB"},
        {"{% for i in [1, 2] %}A{% if true %}B{% continue %}{% endif %}C{% endfor %}",
         Json::object(), "ABAB"},
        {"{% for i in [1, 2] %}A{% if true %}B{% if true %}C{% break %}D{% endif %}E{% endif %}F"
         "{% endfor %}G",
         Json::object(), "ABCG"},
        {"{% for i in [1, 2, 3] %}<{% if i == 2 %}x{% else %}{{ i }}{% continue %}{% endif %}>"
         "{% endfor %}",
         Json::object(), "<1<x><3"},
        {"{% for i in [1, 2] %}{% for j in [1, 2, 3] %}{{ i }}{{ j }}{% if j == 2 %}!{% break %}"
         "{% endif %}{% endfor %};{% endfor %}",
         Json::object(), "1112!;2122!;"},
        // An else block's loop control is the enclosing loop's.
        {"{% for i in [1, 2] %}[{% for j in [] %}{{ j }}{% else %}e{{ i }}{% break %}{% endfor %}]"
         "{% endfor %}",
         Json::object(), "[e1"},
        // Loop control inside a set or filter block leaves the loop without the block.
        {"{% for i in [1, 2] %}A{% set x %}B{% break %}{% endset %}C{% endfor %}D", Json::object(),
         "AD"},
        {"{% for i in [1, 2] %}A{% filter upper %}b{% break %}{% endfilter %}C{% endfor %}D",
         Json::object(), "AD"},
        // The else block runs when no iteration completed its body.
        {"{% for x in [1, 2] %}{% continue %}{% else %}E{% endfor %}", Json::object(), "E"},
        // Elements are taken one at a time: an element after the break is never unpacked, and the
        // loop reads its list live (length is the list's length when first asked for).
        {"{% for a, b in [[1, 2], 3] %}{{ a }}{{ b }}{% break %}{% endfor %}", Json::object(),
         "12"},
        {"{% set xs = [1, 2] %}{% for x in xs %}{{ x }}{% if x == 1 %}{% set _ = xs.append(3) %}"
         "{% endif %}{% endfor %}",
         Json::object(), "123"},
        {"{% set xs = [1, 2] %}{% for x in xs %}{{ x }}/{{ loop.length }}{% if x == 1 %}"
         "{% set _ = xs.append(3) %}{% endif %};{% endfor %}",
         Json::object(), "1/2;2/2;3/2;"},
        {"{% set xs = [1, 2] %}{% for x in xs %}{{ x }}{{ loop.last }}{% if loop.last %}"
         "{% set _ = xs.append(3) %}{% endif %};{% endfor %}",
         Json::object(), "1False;2True;"},
        // A filtered loop evaluates its condition when it reaches an element, or when a look-ahead
        // (length, last) needs it.
        {"{% set ns = namespace(n=0) %}{% for x in [1, 2, 3] if ns.n < 2 %}"
         "{% set ns.n = ns.n + 1 %}{{ x }}{% endfor %}",
         Json::object(), "12"},
        {"{% set ns = namespace(n=0) %}{% for x in [1, 2, 3] if ns.n < 2 %}{{ loop.length }}"
         "{% set ns.n = ns.n + 1 %}{{ x }}{% endfor %}",
         Json::object(), "313233"},
        {"{% set ns = namespace(n=0) %}{% for x in [1, 2, 3] if ns.n < 2 %}{{ loop.last }}"
         "{% set ns.n = ns.n + 1 %}{{ x }}{% endfor %}",
         Json::object(), "False1False2True3"},
        {"{% for x in [1, 2, 3, 4] if x is even %}{{ loop.index }}/{{ loop.length }}"
         "{{ loop.previtem }}{{ loop.nextitem }}{{ loop.last }};{% endfor %}",
         Json::object(), "1/24False;2/22True;"},
        {"{% for i in [1] %}{% for x in [1, 2] if loop.index == 1 %}{{ x }}{% endfor %}"
         "{% endfor %}",
         Json::object(), "12"},
        {"{% for c in 'abc' %}{{ c }}{{ loop.revindex0 }}{{ loop.nextitem }}{% endfor %}",
         Json::object(), "a2bb1cc0"},
        {"{% for k in {'a': 1, 'b': 2} %}{{ k }}{{ loop.previtem }}{{ loop.length }}{% endfor %}",
         Json::object(), "a2ba2"},
        // One loop object per loop, as Jinja2's LoopContext: a stored `loop` shows its last state.
        {"{% set n = namespace(l=none) %}{% for x in [1, 2] %}{% if loop.first %}"
         "{% set n.l = loop %}{% endif %}{% endfor %}{{ n.l.index }}",
         Json::object(), "2"},
        // A macro defined in a loop reads the enclosing scopes when it runs.
        {"{% set n = namespace(f=none) %}{% set y = 1 %}{% for x in ['c'] %}{% macro show() %}"
         "{{ y }}{% endmacro %}{% set n.f = show %}{% endfor %}{% set y = 2 %}{{ n.f() }}",
         Json::object(), "2"},
        // divisibleby: every integer is divisible by -1, including INT64_MIN.
        {"{{ -6 is divisibleby(3) }}{{ 7 is divisibleby(-2) }}{{ 0 is divisibleby(5) }}",
         Json::object(), "TrueFalseTrue"},
        {"{{ n is divisibleby(-1) }}{{ n is divisibleby(2) }}",
         {{"n", std::numeric_limits<std::int64_t>::min()}},
         "TrueTrue"},
    };
    int failures = 0;
    for (const auto& item : cases) {
        try {
            const auto actual = JinjaTemplate(item.source, "loop-test").render(item.input).text;
            if (actual == item.expected) continue;
            std::cerr << "Jinja loop/scope mismatch in " << item.source
                      << "\nexpected: " << Json(item.expected).dump()
                      << "\nactual: " << Json(actual).dump() << '\n';
        } catch (const std::exception& error) {
            std::cerr << "Jinja loop/scope case failed: " << item.source << '\n'
                      << error.what() << '\n';
        }
        ++failures;
    }
    // Jinja2 refuses these when the template compiles, reached or not.
    for (const char* source :
         {"A{% break %}B", "{% if false %}{% continue %}{% endif %}ok",
          "{% macro m() %}x{% break %}{% endmacro %}{% for i in [1] %}{{ m() }}{% endfor %}",
          "{% macro w() %}[{{ caller() }}]{% endmacro %}{% for i in [1, 2] %}{% call w() %}x"
          "{% break %}{% endcall %}{% endfor %}",
          "{% for loop in [1, 2] %}{{ loop }}{% endfor %}",
          "{% for a, loop in [[1, 2]] %}{{ a }}{% endfor %}"}) {
        try {
            (void)JinjaTemplate(source, "compile-test");
            std::cerr << "template compiled: " << source << '\n';
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).starts_with("compile-test:"),
                              "template compile error omitted its source name");
        }
    }
    // Template errors, not process faults: a zero divisor, a `loop` attribute assignment.
    const Json zero{{"zero", 0}};
    for (const char* source : {"{{ 1 is divisibleby(0) }}", "{{ 5 is divisibleby(zero) }}",
                               "{% for x in [1, 2] %}{% set loop.foo = 1 %}{% endfor %}"}) {
        try {
            (void)JinjaTemplate(source, "error-test").render(zero);
            std::cerr << "template rendered: " << source << '\n';
            ++failures;
        } catch (const std::invalid_argument& error) {
            failures += check(std::string(error.what()).starts_with("error-test:"),
                              "template error omitted its source name");
        }
    }
    return failures;
}

int origins_and_requests() {
    const JinjaTemplate compiled(
        "{% filter upper %}{{ text|trim }}{% endfilter %}|{{ tool|tojson }}", "shared-test");
    const Json context{{"text", " hello "}, {"tool", {{"name", "f"}}}};
    const std::vector<infernix::text::TemplateInputRegion> regions{{"/text", 7}, {"/tool", 9}};
    infernix::text::TemplateRenderOptions options{.timestamp = 0, .regions = regions};
    const auto result = compiled.render(context, options);
    int failures = check(result.text == "HELLO|{\"name\": \"f\"}" && result.regions.size() == 2 &&
                             result.regions[0].tag == 7 && result.regions[0].begin == 0 &&
                             result.regions[0].end == 5 && !result.regions[0].source_offset &&
                             result.regions[1].tag == 9 && result.regions[1].begin == 6 &&
                             result.regions[1].end == result.text.size(),
                         "transformed/JSON input regions were lost or incorrectly mapped");
    failures += check(compiled.render(context).regions.empty(),
                      "unannotated render retained request origins");
    const auto clipped =
        JinjaTemplate("{{ text[1:3] }}", "slice-test").render({{"text", "a北京c"}}, options);
    failures += check(clipped.regions.size() == 1 && clipped.regions[0].source_offset == 1 &&
                          clipped.regions[0].end == 6,
                      "Unicode slice lost byte origin mapping");
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 8; ++worker) {
        workers.push_back(std::async(std::launch::async, [&] {
            for (int i = 0; i < 100; ++i)
                if (compiled.render(context, options).text != result.text) return false;
            return true;
        }));
    }
    for (auto& worker : workers)
        failures += check(worker.get(), "shared template leaked mutable request state");

    struct Cancelled : std::runtime_error {
        Cancelled() : std::runtime_error("cancelled") {}
    };

    int calls          = 0;
    options.checkpoint = [&] {
        if (++calls > 2) throw Cancelled();
    };
    try {
        (void)JinjaTemplate("{% for i in range(1000000) %}{{ i }}{% endfor %}", "cancel-test")
            .render(Json::object(), options);
        failures += check(false, "template ignored cancellation");
    } catch (const Cancelled&) {}
    options.checkpoint = {};
    const JinjaTemplate clock("{{ strftime_now('%Y') }}", "clock-test");
    failures += check(clock.render(Json::object(), options).text ==
                          clock.render(Json::object(), options).text,
                      "time snapshot was not reused");
    return failures;
}

int trim_semantics() {
    const JinjaTemplate trims(
        "[{{ text.strip(chars) }}]|[{{ text.lstrip(chars) }}]|[{{ text.rstrip(chars) }}]",
        "trim-test");
    int failures = check(trims.render({{"text", "北🌏x🌏北"}, {"chars", "🌏北"}}).text ==
                             "[x]|[x🌏北]|[北🌏x]",
                         "selected Unicode trim changed left/right semantics");
    failures +=
        check(trims.render({{"text", "北🌏"}, {"chars", "🌏北"}}).text == "[]|[]|[]" &&
                  trims.render({{"text", "北🌏"}, {"chars", ""}}).text == "[北🌏]|[北🌏]|[北🌏]" &&
                  trims.render({{"text", ""}, {"chars", "🌏北"}}).text == "[]|[]|[]",
              "empty or fully trimmed Unicode input changed");
    failures +=
        check(JinjaTemplate("[{{ text.strip() }}]|[{{ text.lstrip() }}]|[{{ text.rstrip() }}]",
                            "whitespace-trim")
                      .render({{"text", "\u0085\u001c\u3000\t"}})
                      .text == "[]|[]|[]",
              "default Unicode whitespace trim changed");

    const std::vector<infernix::text::TemplateInputRegion> regions{{"/text", 7}};
    const auto clipped = JinjaTemplate("{{ ('　' ~ text ~ '　').strip() }}", "trim-origin")
                             .render({{"text", "　<|image_pad|>　"}}, {.regions = regions});
    failures +=
        check(clipped.text == "<|image_pad|>" && clipped.regions.size() == 1 &&
                  clipped.regions[0].tag == 7 && clipped.regions[0].begin == 0 &&
                  clipped.regions[0].end == clipped.text.size() &&
                  clipped.regions[0].source_offset == 3 && clipped.literal_spans.size() == 1 &&
                  clipped.literal_spans[0].begin == 0 &&
                  clipped.literal_spans[0].end == clipped.text.size(),
              "trim across string parts lost literal status or byte origin");
    failures += check(
        clipped.boundary_mappings.size() == 2 && clipped.boundary_mappings[0].tag == 7 &&
            clipped.boundary_mappings[0].source_begin == 0 &&
            clipped.boundary_mappings[0].source_end == 3 &&
            clipped.boundary_mappings[0].offset == 0 && clipped.boundary_mappings[1].tag == 7 &&
            clipped.boundary_mappings[1].source_begin == 16 &&
            clipped.boundary_mappings[1].source_end == 19 &&
            clipped.boundary_mappings[1].offset == 13,
        "trim did not map deleted Unicode source boundaries to the surviving content edges");

    const auto empty = JinjaTemplate("before{{ text|trim }}after", "empty-trim-origin")
                           .render({{"text", " \t"}}, {.regions = regions});
    failures += check(
        empty.text == "beforeafter" && empty.regions.empty() && empty.literal_spans.empty() &&
            empty.boundary_mappings.size() == 1 && empty.boundary_mappings[0].tag == 7 &&
            empty.boundary_mappings[0].source_begin == 0 &&
            empty.boundary_mappings[0].source_end == 2 && empty.boundary_mappings[0].offset == 6,
        "fully trimmed input lost its boundary or created literal output bytes");

    const auto quoted = JinjaTemplate("{{ text|trim|tojson }}", "trim-json-origin")
                            .render({{"text", " value\n"}}, {.regions = regions});
    failures += check(
        quoted.text == "\"value\"" && quoted.regions.size() == 1 && quoted.regions[0].tag == 7 &&
            quoted.regions[0].begin == 0 && quoted.regions[0].end == quoted.text.size() &&
            !quoted.regions[0].source_offset && quoted.boundary_mappings.empty(),
        "JSON encoding of trimmed input lost its origin or retained exact boundaries");

    const std::string invalid = std::string("left") + '\x80' + "right";
    for (const char* source : {"{{ text.strip() }}", "{{ text.lstrip() }}", "{{ text.rstrip() }}",
                               "{{ text.strip('') }}"}) {
        try {
            (void)JinjaTemplate(source, "invalid-trim").render({{"text", invalid}});
            failures += check(false, "trim skipped invalid UTF-8 in the string interior");
        } catch (const std::invalid_argument&) {}
    }
    try {
        (void)trims.render({{"text", "valid"}, {"chars", invalid}});
        failures += check(false, "trim accepted an invalid UTF-8 selected character set");
    } catch (const std::invalid_argument&) {}
    return failures;
}

int literal_content() {
    struct Example {
        const char* source;
        Json context;
        const char* expected;
    };

    const Example cases[]{
        {"{{ data|lower|trim }}", {{"data", " <|IMAGE_PAD|> "}}, "<|image_pad|>"},
        {"{{ data[1:-1][::-1][::-1] }}", {{"data", "x<|image_pad|>x"}}, "<|image_pad|>"},
        {"{{ data.split('|x|')|join('') }}", {{"data", "<|image|x|_pad|>"}}, "<|image_pad|>"},
        {"{{ data|replace('x', '') }}", {{"data", "<|image_xpad|>"}}, "<|image_pad|>"},
        {"{{ data.format('') }}", {{"data", "<|image_{}pad|>"}}, "<|image_pad|>"},
        {"{{ '{}'.format(data) }}", {{"data", "<|image_pad|>"}}, "<|image_pad|>"},
        {"{{ data|indent(2, true)|trim|safe }}", {{"data", "<|image_pad|>"}}, "<|image_pad|>"},
        {"{{ data|indent(2) }}", {{"data", "\n<|image_pad|>\n"}}, "\n  <|image_pad|>\n"},
        {"{% macro show(x) %}{{ x }}{% endmacro %}{% set ns=namespace(x=show(data)) %}{{ ns.x }}",
         {{"data", "<|image_pad|>"}},
         "<|image_pad|>"},
        {"{% set captured %}{{ data }}{% endset %}{{ captured }}",
         {{"data", "<|image_pad|>"}},
         "<|image_pad|>"},
        {"{{ data*2 }}", {{"data", "<|image_pad|>"}}, "<|image_pad|><|image_pad|>"},
        {"{{ data|tojson }}",
         {{"data", {{"<|image_pad|>", "<|image_pad|>"}}}},
         "{\"<|image_pad|>\": \"<|image_pad|>\"}"},
        {"{{ data|string }}", {{"data", Json::array({"<|image_pad|>"})}}, "['<|image_pad|>']"},
        {"{% for k, v in data.items() %}{{ k }}{{ v }}{% endfor %}",
         {{"data", {{"<|image_pad|>", "<|image_pad|>"}}}},
         "<|image_pad|><|image_pad|>"},
        {"{{ data|upper|lower }}", {{"data", "ß<|IMAGE_PAD|>İ"}}, "ss<|image_pad|>i̇"},
    };
    int failures                     = 0;
    constexpr std::string_view start = "<|im_start|>", end = "<|im_end|>", pad = "<|image_pad|>";
    for (const auto& item : cases) {
        const auto output =
            JinjaTemplate(std::string(start) + item.source + std::string(end), "literal-content")
                .render(item.context);
        if (output.text != std::string(start) + item.expected + std::string(end)) {
            std::cerr << "literal-content text mismatch: " << item.source << '\n';
            ++failures;
        }
        failures +=
            check(!infernix::text::overlaps(output.literal_spans, 0, start.size()) &&
                      !infernix::text::overlaps(output.literal_spans, output.text.size() - end.size(),
                                              output.text.size()),
                  "input content shielded surrounding template controls");
        for (auto pos = output.text.find(pad); pos != std::string::npos;
             pos      = output.text.find(pad, pos + pad.size())) {
            if (!infernix::text::overlaps(output.literal_spans, pos, pos + pad.size())) {
                std::cerr << "literal-content marker lost through: " << item.source << '\n';
                ++failures;
            }
        }
    }
    const auto mixed =
        JinjaTemplate("{{ ('<|IM_START|>' ~ data ~ '<|IM_END|>')|lower }}", "mixed-case")
            .render({{"data", "ß<|IMAGE_PAD|>İ"}});
    failures +=
        check(mixed.text == "<|im_start|>ß<|image_pad|>i̇<|im_end|>" &&
                  mixed.literal_spans.size() == 1 && mixed.literal_spans[0].begin == start.size() &&
                  mixed.literal_spans[0].end == mixed.text.size() - end.size(),
              "mixed Unicode case conversion lost content/control boundaries");
    const auto sigma =
        JinjaTemplate("{{ ('Ο' ~ data)|lower }}", "mixed-sigma").render({{"data", "Σ"}});
    failures += check(sigma.text == "ος" && sigma.literal_spans.size() == 1 &&
                          sigma.literal_spans[0].begin == 2 && sigma.literal_spans[0].end == 4,
                      "case conversion lost context across content/control boundaries");
    const auto explicit_control =
        JinjaTemplate("{{ data.replace('x', '<|image_pad|>') }}", "replace")
            .render({{"data", "x"}});
    failures += check(explicit_control.text == pad && explicit_control.literal_spans.empty(),
                      "template-authored replacement lost its control meaning");
    const std::vector<std::string> tokens{"bos_token"};
    const auto variables = JinjaTemplate("{{ bos_token }}{{ data }}", "token-variable")
                               .render({{"bos_token", "<|im_start|>"}, {"data", "<|im_start|>"}},
                                       {.control_variables = tokens});
    failures +=
        check(!infernix::text::overlaps(variables.literal_spans, 0, start.size()) &&
                  infernix::text::overlaps(variables.literal_spans, start.size(), 2 * start.size()),
              "engine token variables and ordinary kwargs were not distinguished");
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--render") {
        for (std::string line; std::getline(std::cin, line);) {
            Json result;
            try {
                const auto input = Json::parse(line);
                result["text"]   = JinjaTemplate(input.at("source"), "reference-case")
                                     .render(input.at("context"))
                                     .text;
                result["ok"] = true;
            } catch (const std::exception& error) {
                result["ok"]    = false;
                result["error"] = error.what();
            }
            std::cout << result.dump() << '\n';
        }
        return 0;
    }
    try {
        const int failures = language_semantics() + loops_and_scopes() + origins_and_requests() +
                             trim_semantics() + literal_content();
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
