#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/score.h"

#include <exception>
#include <string>
#include <utility>

namespace infernix::serve {

// The decide route's shape (decide_http.cpp): parse the whole body, run, read the result. A
// generation server's backend refuses (scoring_not_enabled) before any Engine work.
void HttpServer::handle_score(const httplib::Request& req, httplib::Response& res) {
    try {
        ScoreRequest request = parse_score_request(parse_json_body(req));
        if (!request.model.empty()) {
            request.model = resolve_openai_model(request.model, public_model_id_, options_.any_model_name);
        }
        const ScoreOutcome outcome = service_->score(request, [&req] { return client_disconnected(req); });
        res.set_content(make_score_response(public_model_id_, outcome)
                            .dump(-1, ' ', false, RequestJson::error_handler_t::replace),
                        "application/json");
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
    } catch (const std::exception& exception) {
        write_openai_error(res, ApiError{.status  = 500,
                                         .type    = "internal_error",
                                         .message = exception.what(),
                                         .code    = "internal_error"});
    }
}

} // namespace infernix::serve
