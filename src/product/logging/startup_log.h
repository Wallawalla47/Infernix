#pragma once

#include "infernix/types.h"

#include <memory>
#include <string_view>

namespace infernix::product {

class LoggingRuntime;

// Product-side renderer for the neutral Engine startup observer. Returned observers retain the
// rendering state, so the callback remains self-contained if copied with Engine options.
class StartupLogRenderer {
public:
    explicit StartupLogRenderer(LoggingRuntime& logging);
    ~StartupLogRenderer();

    StartupLogRenderer(const StartupLogRenderer&)            = delete;
    StartupLogRenderer& operator=(const StartupLogRenderer&) = delete;
    StartupLogRenderer(StartupLogRenderer&&)                 = delete;
    StartupLogRenderer& operator=(StartupLogRenderer&&)      = delete;

    [[nodiscard]] StartupObserver observer();
    void engine_ready(const LoadSummary& load);
    // Warns once when preparation lowers instruction turns that the active chat template cannot
    // render where they arrived. An explicit --chat-template override is named, because the
    // measured support comes from whichever template is actually active.
    void chat_template_roles(infernix::ChatTemplateRoleSupport roles,
                             std::string_view active_override_path = {});

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace infernix::product
