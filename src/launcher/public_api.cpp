// The launcher's public API (Launcher members, launcher.h): the OpenAI, Anthropic and llama-server
// routes coding tools use, answered from the launcher's state or proxied to the loaded infernix-serve.

#include "launcher/launcher.h"

#include "launcher/proxy.h"
#include "launcher/route_support.h"
#include "launcher/routing.h"

#include <algorithm>
#include <ctime>
#include <exception>

namespace infernix::launcher {
namespace {

// The default output limit: --default-max-tokens, else infernix-serve's 8,192; at most the context.
std::uint64_t default_output_tokens(const Args& args, std::uint64_t context) {
    std::uint64_t max_output = 8192;
    if (const std::optional<std::string> value = flag_value(args, "--default-max-tokens")) {
        try {
            max_output = std::stoull(*value);
        } catch (const std::exception&) {}
    }
    return std::min(max_output, context);
}

std::string rfc3339(std::int64_t ms) {
    const std::time_t seconds = static_cast<std::time_t>(ms / 1000);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &seconds);
#else
    gmtime_r(&seconds, &utc);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

// Whether a public route's answer can be an event stream (the generation routes), not JSON only.
bool route_streams(std::string_view path) {
    for (const std::string_view json_only :
         {"/count_tokens", "/input_tokens", "/compact", "/decide", "/systemone", "/score"}) {
        if (path.ends_with(json_only)) { return false; }
    }
    return true;
}

std::string json_body_model(const std::string& body, bool& stream, bool& parsed) {
    parsed = false;
    stream = false;
    try {
        const Json value = Json::parse(body);
        parsed           = value.is_object();
        if (!parsed) { return {}; }
        stream = json_or<bool>(value, "stream", false);
        return json_or<std::string>(value, "model", "");
    } catch (const Json::exception&) { return {}; }
}

// A JSON object body with its `model` set. One nested past kMaxJsonNesting goes out as it came, for
// infernix-serve to refuse with its own error: rewriting it would recurse once per level.
std::string body_with_model(const std::string& body, const std::string& model) {
    if (json_text_nesting(body) > kMaxJsonNesting) { return body; }
    Json value     = Json::parse(body);
    value["model"] = model;
    return value.dump();
}

} // namespace

// One entry of /v1/models, readable by OpenAI clients, Anthropic clients (type, display_name,
// created_at, max_input_tokens) and the tools that read other servers' fields: llama.cpp and
// llama-swap (status.value, meta.n_ctx, architecture's modalities), vLLM (max_model_len), LM Studio
// (state, max_context_length), OpenRouter-style catalogues (name, description, context_length,
// supported_parameters).
Json Launcher::model_entry(const std::string& config_id, const ScannedModel& model, const LibraryScan& scan,
                           bool claude_code) const {
    const Args args                 = effective_args(config_id, model, scan);
    const std::uint64_t context     = advertised_context(config_id, model, args);
    const std::uint64_t max_output  = default_output_tokens(args, context);
    const InstanceSnapshot instance = supervisor_->snapshot();
    std::string state               = "not-loaded";
    std::string status              = "unloaded";
    std::string state_words         = "loads on first request";
    if (instance.model_key == config_id && instance.state == InstanceState::Ready) {
        state = status = state_words = "loaded";
    } else if (instance.model_key == config_id && instance.state == InstanceState::Starting) {
        state = status = state_words = "loading";
    }
    const std::string alias    = effective_alias(config_id, model, scan);
    const std::int64_t created = model.modified / 1000;
    const bool vision          = has_flag(args, "--vision");
    const std::string id = claude_code && !icontains(alias, "claude") && !icontains(alias, "anthropic")
                               ? alias + std::string(kClaudeCodeSuffix)
                               : alias;
    std::string description = model.family_label;
    if (model.quantization) { description += ", " + *model.quantization; }
    description += ", " + std::to_string((context + 500) / 1000) + "K context";
    if (vision) { description += ", vision"; }
    description += ", " + state_words;
    const Json input = vision ? Json::array({"text", "image"}) : Json::array({"text"});
    return {{"id", id},
            {"object", "model"},
            {"type", "model"},
            {"created", created},
            {"created_at", rfc3339(model.modified)},
            {"owned_by", "infernix"},
            {"name", alias},
            {"display_name", alias},
            {"description", description},
            {"context_length", context},
            {"context_window", context},
            {"max_model_len", context},
            {"max_context_length", context},
            {"max_input_tokens", context},
            {"max_tokens", max_output},
            {"state", state},
            {"status", {{"value", status}}},
            {"architecture",
             {{"modality", vision ? "text+image->text" : "text->text"},
              {"input_modalities", input},
              {"output_modalities", Json::array({"text"})}}},
            {"family", model.family_label},
            {"quantization", model.quantization ? Json(*model.quantization) : Json(nullptr)},
            {"capabilities", {{"vision", vision}, {"tools", true}, {"reasoning", true}}},
            // The request fields infernix-serve honours (docs/serving.md), in OpenRouter's names.
            {"supported_parameters",
             {"max_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "seed",
              "stop", "tools", "tool_choice", "parallel_tool_calls", "reasoning", "include_reasoning",
              "response_format", "structured_outputs"}},
            {"meta",
             {{"architecture", model.architecture},
              {"n_ctx", context},
              {"n_ctx_train", model.native_context},
              {"n_vocab", model.vocab_size},
              {"n_embd", model.hidden_size},
              {"size", model.bytes}}}};
}

void Launcher::handle_models(const Request& request, Response& response, ApiCall& call) {
    call.quiet           = true;
    const bool anthropic = request.has_header("anthropic-version");
    if (!client_authorized(request, response, anthropic, call)) { return; }
    const bool claude_code = is_claude_code_client(request);
    std::lock_guard lock(library_mutex_);
    if (request.matches.size() > 1 && request.matches[1].matched) {
        const std::string requested = request.matches[1].str();
        const std::optional<std::string> base = strip_claude_code(requested);
        for (const std::string& id : base ? std::vector{requested, *base} : std::vector{requested}) {
            for (const ScannedModel& model : library_.models) {
                for (const std::string& config_id : config_ids(model)) {
                    if (iequals(effective_alias(config_id, model, library_), id) || iequals(config_id, id)) {
                        // Asked by Claude Code's spelling, the entry carries it.
                        Json entry = model_entry(config_id, model, library_);
                        if (base) { entry["id"] = requested; }
                        send_json(response, entry);
                        return;
                    }
                }
            }
        }
        if (anthropic) {
            anthropic_error(response, 404, "not_found_error", "model: " + requested);
        } else {
            openai_error(response, 404, "invalid_request_error", "model_not_found",
                         "The model '" + requested + "' does not exist");
        }
        return;
    }
    Json data = Json::array();
    for (const ScannedModel& model : library_.models) {
        // Incomplete and unsupported artifacts (no Infernix model family) are not offered, nor
        // hidden configurations; each other configuration is a model of its own.
        if (!model.missing_files.empty() || model.family == "unknown") { continue; }
        for (const std::string& config_id : config_ids(model)) {
            const std::optional<StoredModelConfig> stored = store_.model_config(config_id);
            if (stored && !stored->listed) { continue; }
            data.push_back(model_entry(config_id, model, library_, claude_code));
        }
    }
    // Anthropic's list is paged (limit 1-1000, default 20; after_id or before_id); OpenAI's is whole.
    std::size_t begin = 0, end = data.size();
    if (anthropic) {
        std::size_t limit = 20;
        if (request.has_param("limit")) {
            try {
                limit = std::stoul(request.get_param_value("limit"));
            } catch (const std::exception&) {}
            limit = std::clamp<std::size_t>(limit, 1, 1000);
        }
        const auto index_of = [&data](const std::string& id) -> std::optional<std::size_t> {
            for (std::size_t i = 0; i < data.size(); ++i) {
                if (data[i]["id"] == id) { return i; }
            }
            return std::nullopt;
        };
        if (request.has_param("before_id")) {
            end   = index_of(request.get_param_value("before_id")).value_or(0);
            begin = end > limit ? end - limit : 0;
        } else {
            if (request.has_param("after_id")) {
                const std::optional<std::size_t> after = index_of(request.get_param_value("after_id"));
                begin = after ? *after + 1 : data.size();
            }
            end = std::min(data.size(), begin + limit);
        }
    }
    Json page = Json::array();
    for (std::size_t i = begin; i < end; ++i) { page.push_back(std::move(data[i])); }
    const bool more = request.has_param("before_id") ? begin > 0 : end < data.size();
    send_json(response, models_list_body(std::move(page), anthropic && more));
}

Json models_list_body(Json page, bool has_more) {
    Json body = {{"object", "list"}, {"data", page}, {"has_more", has_more}};
    body["first_id"] = page.empty() ? Json(nullptr) : page.front()["id"];
    body["last_id"]  = page.empty() ? Json(nullptr) : page.back()["id"];
    // Codex CLI's model-catalog refresh decodes a top-level `models` (infernix-serve writes both).
    body["models"] = std::move(page);
    return body;
}

// ---------------------------------------------------------------------------------------------
// HTTP: llama-server's router API (Settings.llamacpp_api)

Json Launcher::props_json(const std::string& config_id, const ScannedModel& model, const Args& args,
                          const std::string& alias) const {
    const std::uint64_t context     = advertised_context(config_id, model, args);
    const InstanceSnapshot instance = supervisor_->snapshot();
    const bool loaded               = instance.model_key == config_id && instance.state == InstanceState::Ready;
    int slots                       = 1; // infernix-serve's --max-concurrency default
    if (const std::optional<std::string> value = flag_value(args, "--max-concurrency")) {
        try {
            slots = std::stoi(*value);
        } catch (const std::exception&) {}
    }
    // Without --spec, a server that lists off among its --spec values runs --spec auto, which
    // speculates with the artifact's bundled drafter or --draft-model's; an older server decodes plainly.
    const std::optional<std::string> spec = flag_value(args, "--spec");
    const bool speculative =
        spec ? *spec != "off" && *spec != "none"
             : serve_options().lists_value("--spec", "off") && !has_flag(args, "--lossless") &&
                   (model.has_mtp || model.has_dflash || model.has_dflash2 || has_flag(args, "--draft-model"));
    return {{"default_generation_settings",
             {{"id", 0},
              {"id_task", -1},
              // --max-context bounds each request, as llama.cpp's per-slot n_ctx does.
              {"n_ctx", context},
              {"speculative", speculative},
              {"is_processing", loaded && router_->in_flight() > 0},
              {"params", {{"n_predict", -1}, {"max_tokens", default_output_tokens(args, context)}}}}},
            {"total_slots", slots},
            {"model_path", path_utf8(model.path)},
            {"model_alias", alias},
            {"chat_template_caps",
             {{"supports_tools", true},
              {"supports_tool_calls", true},
              {"supports_parallel_tool_calls", true},
              {"supports_system_role", true},
              {"supports_preserve_reasoning", has_flag(args, "--preserve-thinking")}}},
            {"modalities", {{"vision", has_flag(args, "--vision")}, {"audio", false}}},
            {"build_info", "infernix-launcher " + options_.version},
            {"is_sleeping", false}};
}

void Launcher::publish_model_event(const std::string& model, const std::string& event, Json data) {
    Json body = {{"model", model}, {"event", event}};
    if (!data.is_null()) { body["data"] = std::move(data); }
    model_events_.publish_text(EventHub::format_data(body));
}

void Launcher::handle_props(const Request& request, Response& response, ApiCall& call) {
    call.quiet = true;
    if (!store_.settings().llamacpp_api) {
        response.status = 404; // as an unknown path
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    // Router mode names the model (?model=); single-model mode describes the loaded one, here the
    // loaded configuration, else the default one. Nothing is loaded for the answer.
    const std::string name = request.get_param_value("model");
    std::optional<std::string> key;
    if (!name.empty()) {
        key = resolve_model(name);
        if (!key) {
            llamacpp_error(response, 400, "invalid_request_error", "model '" + name + "' not found");
            return;
        }
    } else {
        const InstanceSnapshot instance = supervisor_->snapshot();
        if (!instance.model_key.empty() &&
            (instance.state == InstanceState::Ready || instance.state == InstanceState::Starting)) {
            key = instance.model_key;
        } else {
            key = store_.settings().default_model;
        }
    }
    std::optional<std::string> id;
    ScannedModel model;
    Args args;
    std::string alias;
    {
        std::lock_guard lock(library_mutex_);
        if (const std::optional<ConfigRef> ref = key ? find_config_locked(*key) : std::nullopt) {
            id    = ref->id;
            model = *ref->model;
            args  = effective_args(ref->id, *ref->model, library_);
            alias = effective_alias(ref->id, *ref->model, library_);
        }
    }
    if (!id) {
        // Nothing to describe: llama-server's router answer.
        send_json(response, {{"role", "router"},
                             {"model_alias", "infernix-launcher"},
                             {"model_path", "none"},
                             {"default_generation_settings", {{"params", Json::object()}, {"n_ctx", 0}}},
                             {"build_info", "infernix-launcher " + options_.version}});
        return;
    }
    send_json(response, props_json(*id, model, args, alias));
}

void Launcher::handle_models_load(const Request& request, Response& response, ApiCall& call) {
    if (!store_.settings().llamacpp_api) {
        call.quiet      = true; // a probe of a path the launcher does not answer
        response.status = 404;
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    bool stream = false, parsed = false;
    const std::string name = json_body_model(request.body, stream, parsed);
    if (!name.empty()) { call.model = name; }
    if (!parsed) {
        llamacpp_error(response, 400, "invalid_request_error", "the request body must be a JSON object");
        return;
    }
    const std::optional<std::string> key = resolve_model(name);
    if (!key) {
        llamacpp_error(response, 404, "not_found_error", "model is not found");
        return;
    }
    call.key                        = key;
    const InstanceSnapshot instance = supervisor_->snapshot();
    if ((instance.model_key == *key &&
         (instance.state == InstanceState::Ready || instance.state == InstanceState::Starting)) ||
        router_->loading_model() == key) {
        llamacpp_error(response, 400, "invalid_request_error", "model is already running");
        return;
    }
    // Queued like a request (the loaded model's running requests finish first); it returns at once.
    router_->request_load(*key, false);
    send_json(response, {{"success", true}});
}

void Launcher::handle_models_unload(const Request& request, Response& response, ApiCall& call) {
    if (!store_.settings().llamacpp_api) {
        call.quiet      = true;
        response.status = 404;
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    bool stream = false, parsed = false;
    const std::string name = json_body_model(request.body, stream, parsed);
    if (!name.empty()) { call.model = name; }
    const std::optional<std::string> key = parsed ? resolve_model(name) : std::nullopt;
    if (!key) {
        llamacpp_error(response, 400, "invalid_request_error", "model is not found");
        return;
    }
    call.key                        = key;
    const InstanceSnapshot instance = supervisor_->snapshot();
    if (instance.model_key != *key ||
        (instance.state != InstanceState::Ready && instance.state != InstanceState::Starting)) {
        llamacpp_error(response, 400, "invalid_request_error", "model is not running");
        return;
    }
    router_->request_unload();
    send_json(response, {{"success", true}});
}

void Launcher::handle_models_sse(const Request& request, Response& response, ApiCall& call) {
    call.quiet = true;
    if (!store_.settings().llamacpp_api) {
        response.status = 404;
        return;
    }
    if (!client_authorized(request, response, false, call)) { return; }
    auto subscription = model_events_.subscribe();
    response.status   = 200;
    response.set_header("Cache-Control", "no-cache");
    response.set_header("X-Accel-Buffering", "no");
    response.set_chunked_content_provider(
        "text/event-stream",
        [subscription](size_t, httplib::DataSink& sink) {
            const std::optional<std::string> event = subscription->next(std::chrono::seconds(15));
            if (!event) {
                if (subscription->closed()) {
                    sink.done();
                    return true;
                }
                static constexpr std::string_view kPing = ": ping\n\n"; // llama.cpp clients skip comments
                return sink.write(kPing.data(), kPing.size());
            }
            return sink.write(event->data(), event->size());
        },
        [subscription](bool) { subscription->close(); });
}

void Launcher::handle_generation(const Request& request, Response& response, ApiCall& call) {
    const bool anthropic = request.path.find("/messages") != std::string::npos;
    if (!client_authorized(request, response, anthropic, call)) {
        call.error = "invalid API key";
        return;
    }
    bool stream = false, parsed = false;
    const std::string name = json_body_model(request.body, stream, parsed);
    // Only the generation routes stream: count_tokens, input_tokens, compact, decide and score answer JSON whatever
    // `stream` says, so they wait for a load and relay that JSON instead of starting an event stream for it.
    stream      = stream && route_streams(request.path);
    call.stream = stream;
    if (!name.empty()) { call.model = name; }
    if (!parsed) {
        call.error = "the body is not a JSON object";
        if (anthropic) {
            anthropic_error(response, 400, "invalid_request_error", "the request body must be a JSON object");
        } else {
            openai_error(response, 400, "invalid_request_error", "invalid_json", "the request body must be a JSON object");
        }
        return;
    }
    std::string routed_by;
    const std::optional<std::string> key = route_request(name, routed_by);
    call.routed_by                       = routed_by.empty() ? std::nullopt : std::optional(routed_by);
    if (!key) {
        const std::string message = name.empty() ? "the request names no model and none is loaded or set as default"
                                                 : "The model '" + name + "' is not in the launcher's library";
        call.error = message;
        if (anthropic) {
            anthropic_error(response, 404, "not_found_error", message);
        } else {
            openai_error(response, 404, "invalid_request_error", "model_not_found", message);
        }
        return;
    }
    call.key          = key;
    std::string alias = *key;
    {
        std::lock_guard lock(library_mutex_);
        if (const std::optional<ConfigRef> ref = find_config_locked(*key)) {
            alias = effective_alias(ref->id, *ref->model, library_);
        }
    }
    // infernix-serve requires a model (the web UI's chat sends none for "the loaded model"): a request
    // that names none goes out naming the configuration it was routed to.
    std::optional<Request> named;
    if (name.empty()) {
        named.emplace(request);
        named->body = body_with_model(request.body, alias);
    }
    const Request& forwarded     = named ? *named : request;
    const std::uint64_t activity = activity_begin(request, call);
    call.id                      = activity;
    const auto finished          = [this, activity](const ProxyOutcome& outcome) {
        const bool failed = !outcome.error.empty() || outcome.status >= 400 || outcome.status == 0;
        activity_update(activity, failed ? "failed" : "done", outcome.status,
                        outcome.error.empty() ? std::nullopt : std::optional(outcome.error));
    };
    const auto refuse = [&](const Router::Refusal& refusal) {
        activity_update(activity, "failed", refusal.status, refusal.message);
        call.error = refusal.code + ": " + refusal.message;
        // Whole seconds, as clients (Claude Code among them) expect.
        if (refusal.retry_after_s > 0) { response.set_header("Retry-After", std::to_string(refusal.retry_after_s)); }
        if (anthropic) {
            anthropic_error(response, refusal.status == 499 ? 499 : 529, "overloaded_error", refusal.message);
        } else {
            openai_error(response, refusal.status, "server_error", refusal.code, refusal.message);
        }
    };
    // A stream that would wait for a model load gets its answer started at once, kept alive while
    // the model loads; a model held after a failed load is refused before any stream starts.
    if (stream) {
        if (std::shared_ptr<Router::Lease> ready = router_->try_acquire(*key)) {
            activity_update(activity, "running", 0, std::nullopt);
            proxy(forwarded, response, std::move(ready), call, finished);
            return;
        }
        if (Router::Refusal held; router_->held(*key, held)) {
            refuse(held);
            return;
        }
        const std::string config_id = *key;
        proxy_after_load(
            forwarded, response,
            [this, config_id, activity](const std::function<bool()>& cancelled, Router::Refusal& refusal) {
                std::shared_ptr<Router::Lease> lease = router_->acquire(config_id, cancelled, refusal);
                if (lease) { activity_update(activity, "running", 0, std::nullopt); }
                return lease;
            },
            anthropic ? StreamDialect::Anthropic : StreamDialect::OpenAI, "loading " + alias, call, finished);
        return;
    }
    Router::Refusal refusal;
    std::shared_ptr<Router::Lease> lease =
        router_->acquire(*key, [&request] { return request.is_connection_closed(); }, refusal);
    if (!lease) {
        refuse(refusal);
        return;
    }
    activity_update(activity, "running", 0, std::nullopt);
    proxy(forwarded, response, std::move(lease), call, finished);
}

std::optional<std::string> Launcher::route_request(const std::string& name, std::string& routed_by) const {
    const Settings settings         = store_.settings();
    const InstanceSnapshot instance = supervisor_->snapshot();
    // The loaded (or loading) model, else the default model.
    const auto loaded = [&]() -> std::optional<std::string> {
        if (!instance.model_key.empty() &&
            (instance.state == InstanceState::Ready || instance.state == InstanceState::Starting)) {
            return instance.model_key;
        }
        if (std::optional<std::string> loading = router_->loading_model()) { return loading; }
        return std::nullopt;
    };
    const auto default_model = [&]() -> std::optional<std::string> {
        if (!settings.default_model) { return std::nullopt; }
        std::lock_guard lock(library_mutex_);
        if (find_config_locked(*settings.default_model)) { return settings.default_model; }
        return std::nullopt;
    };
    if (settings.pin_loaded) {
        if (std::optional<std::string> key = loaded()) {
            routed_by = "pinned";
            return key;
        }
    }
    if (const ModelRoute* route = match_route(settings.model_routes, name)) {
        std::optional<std::string> key;
        if (route->target == "loaded") {
            key = loaded();
            if (!key) { key = default_model(); }
        } else {
            std::lock_guard lock(library_mutex_);
            if (find_config_locked(route->target)) { key = route->target; }
        }
        if (key) {
            routed_by = "route";
            return key;
        }
    }
    if (std::optional<std::string> key = resolve_model(name)) {
        routed_by = "name";
        return key;
    }
    if (settings.unknown_model == "loaded") {
        // Like infernix-serve --any-model-name: the loaded (or loading) model, else the default.
        std::optional<std::string> key = loaded();
        if (!key) { key = default_model(); }
        if (key) {
            routed_by = "loaded";
            return key;
        }
    }
    return std::nullopt;
}

void Launcher::handle_stateful(const Request& request, Response& response, ApiCall& call) {
    if (!client_authorized(request, response, false, call)) {
        call.error = "invalid API key";
        return;
    }
    // Stored Responses live in the loaded server; nothing is loaded for them.
    const InstanceSnapshot instance = supervisor_->snapshot();
    if (instance.model_key.empty() || instance.state != InstanceState::Ready) {
        call.error = "no model is loaded";
        openai_error(response, 404, "invalid_request_error", "not_found",
                     "no model is loaded; stored responses live in the loaded model's server");
        return;
    }
    call.key = instance.model_key;
    Router::Refusal refusal;
    std::shared_ptr<Router::Lease> lease =
        router_->acquire(instance.model_key, [&request] { return request.is_connection_closed(); }, refusal);
    if (!lease) {
        call.error = refusal.code + ": " + refusal.message;
        if (refusal.retry_after_s > 0) { response.set_header("Retry-After", std::to_string(refusal.retry_after_s)); }
        openai_error(response, refusal.status, "server_error", refusal.code, refusal.message);
        return;
    }
    proxy(request, response, std::move(lease), call, nullptr);
}

std::function<void(const ProxyOutcome&)> Launcher::exchange_finished(const Request& request, ApiCall& call,
                                                                     std::function<void(const ProxyOutcome&)> finished) {
    // The exchange ends after this handler returns (streams); its log line is written then.
    call.proxied      = true;
    const bool bodies = store_.settings().log_bodies;
    std::string request_body = bodies ? request.body : std::string();
    return [this, call, bodies, request_body = std::move(request_body), finished = std::move(finished)](
               const ProxyOutcome& outcome) {
        if (finished) { finished(outcome); }
        ApiCall ended = call;
        if (!outcome.error.empty()) { ended.error = outcome.error; }
        log_api_request(ended, outcome.status, outcome.first_byte_ms, outcome.bytes_out);
        if (bodies) { log_request_body(ended, request_body, outcome.status, outcome.body, outcome.body_truncated); }
    };
}

void Launcher::proxy(const Request& request, Response& response, std::shared_ptr<Router::Lease> lease, ApiCall& call,
                     std::function<void(const ProxyOutcome&)> finished) {
    const int port = lease->port();
    if (call.id != 0) { response.set_header("X-Infernix-Request-Id", std::to_string(call.id)); }
    const std::size_t capture = store_.settings().log_bodies ? kBodyCaptureLimit : 0;
    forward_request(request, response, port, std::move(lease), exchange_finished(request, call, std::move(finished)),
                    call.arrived, capture, call.id != 0 ? "launcher-" + std::to_string(call.id) : std::string());
}

void Launcher::proxy_after_load(const Request& request, Response& response, AcquireLease acquire, StreamDialect dialect,
                                const std::string& loading_note, ApiCall& call,
                                std::function<void(const ProxyOutcome&)> finished) {
    if (call.id != 0) { response.set_header("X-Infernix-Request-Id", std::to_string(call.id)); }
    const std::size_t capture = store_.settings().log_bodies ? kBodyCaptureLimit : 0;
    forward_after_load(request, response, std::move(acquire), dialect, loading_note,
                       exchange_finished(request, call, std::move(finished)), call.arrived, capture,
                       call.id != 0 ? "launcher-" + std::to_string(call.id) : std::string());
}

void Launcher::register_public_routes() {
    server_.Get(R"((?:/v1)?/health)", [this](const Request&, Response& response) {
        const InstanceSnapshot instance = supervisor_->snapshot();
        send_json(response, {{"status", "ok"},
                             {"state", state_name(instance.state)},
                             {"model", instance.alias.empty() ? Json(nullptr) : Json(instance.alias)}});
    });
    // Every public request becomes one event-log line (a proxied one when its exchange ends).
    using PublicHandler = void (Launcher::*)(const Request&, Response&, ApiCall&);
    const auto logged   = [this](PublicHandler handler) {
        return [this, handler](const Request& q, Response& s) {
            ApiCall call;
            call.arrived    = std::chrono::steady_clock::now();
            call.method     = q.method;
            call.path       = q.path;
            call.client     = q.remote_addr;
            call.user_agent = q.get_header_value("User-Agent");
            call.bytes_in   = q.body.size();
            (this->*handler)(q, s, call);
            if (call.proxied) { return; }
            log_api_request(call, s.status, std::nullopt, s.body.size());
            if (store_.settings().log_bodies && !call.quiet) { log_request_body(call, q.body, s.status, s.body, false); }
        };
    };
    // llama-server's router API when Settings.llamacpp_api is on (else 404, as unknown paths);
    // before /models/{id}, which would take /models/sse for a model id.
    server_.Get(R"((?:/v1)?/props)", logged(&Launcher::handle_props));
    server_.Post("/models/load", logged(&Launcher::handle_models_load));
    server_.Post("/models/unload", logged(&Launcher::handle_models_unload));
    server_.Get("/models/sse", logged(&Launcher::handle_models_sse));
    // Also without /v1, for tools given the bare address as their base URL (llama-swap and
    // llama.cpp answer /models too).
    server_.Get(R"((?:/v1){0,2}/models)", logged(&Launcher::handle_models));
    server_.Get(R"((?:/v1){0,2}/models/(.+))", logged(&Launcher::handle_models));
    server_.Post(
        R"((?:/v1)?/v1/(chat/completions|completions|responses|responses/input_tokens|responses/compact|messages|messages/count_tokens|decide|systemone|score))",
        logged(&Launcher::handle_generation));
    // Stored Responses live in the loaded server.
    server_.Get(R"((?:/v1)?/v1/responses/([^/]+)(/input_items)?)", logged(&Launcher::handle_stateful));
    server_.Post(R"((?:/v1)?/v1/responses/([^/]+)/cancel)", logged(&Launcher::handle_stateful));
    server_.Delete(R"((?:/v1)?/v1/responses/([^/]+))", logged(&Launcher::handle_stateful));
}

} // namespace infernix::launcher
