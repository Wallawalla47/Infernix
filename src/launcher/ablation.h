#pragma once

// The Ablation page (docs/launcher.md, "Ablation"): the launcher runs tools/ablate, which fits a refusal-ablation
// sidecar (--ablate) for one of the library's models on this computer.
//
// It finds the tool (an `ablate` folder beside the launcher, as a release ships it, else tools/ablate of the checkout
// the launcher was built in) and a Python 3.11 or newer with pip (else Set up downloads a portable one, pinned, into
// `ablate-python` beside the launcher), installs the tool's packages (numpy and optuna, from PyPI) into
// <data>/ablation/packages, downloads public prompt sets (the tool's `datasets fetch`, pinned files checked by
// their SHA-256) into <data>/ablation/prompts, and runs one fit at a time: `init` with the configuration's command
// line, then `auto`, whose sidecar lands in <data>/ablation/sidecars/<name>. After a run, `choose` keeps another
// trial of its study (a point of the front its progress shows) as a sidecar of its own.
//
// A fit starts its own infernix-serve engines on the whole card, so the launcher unloads its model first and loads
// none until the run ends (Hooks::take_gpu, release_gpu). A run is resumable: the same configuration (its command
// line and environment, the infernix-serve file), prompt set and name continue the same run folder, whose finished
// work (recordings, trials) is kept.

#include "launcher/process.h"
#include "launcher/util.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace infernix::launcher {

struct AblationRun {
    std::string config_id;
    std::string model_name;
    // The configuration's command line: infernix-serve, the artifact (absolute), its launch parameters without the
    // flags the launcher manages and without --ablate (a fit measures the plain model).
    std::vector<std::string> argv;
    EnvOverlay env;
    std::string preset;
    int trials = 300;
    std::string name; // the sidecar's folder name
};

class Ablation {
public:
    static constexpr int kMinTrials = 10;
    static constexpr int kMaxTrials = 5000;

    struct Hooks {
        std::string version; // the launcher's build: the tool version the sidecars it writes record
        // Takes the GPU: no model loads from now on, the loaded one is unloaded, and it returns once no server runs;
        // false, with the GPU not taken, when `abort` turned true first.
        std::function<bool(const std::function<bool()>& abort)> take_gpu;
        std::function<void()> release_gpu;
        std::function<void(const Json& state)> changed;
        std::function<void(const std::string& level, const std::string& message)> event;
        std::function<void(const std::string& level, const std::string& message)> toast;
        // A sidecar was exported into `folder`, inside sidecars_dir() (the library lists it).
        std::function<void(const std::filesystem::path& folder)> exported;
    };

    // `tool_dir`: the folder holding ablate.py (the launcher's --ablate-tool), else found as above.
    Ablation(std::filesystem::path data_dir, std::optional<std::filesystem::path> tool_dir, Hooks hooks);
    ~Ablation();
    Ablation(const Ablation&)            = delete;
    Ablation& operator=(const Ablation&) = delete;

    // The page's state (the first call looks for a Python: it runs a few short processes).
    [[nodiscard]] Json state();
    // Looks for a Python again: the owner installed one, or chose another.
    void detect_python();
    // Uses this Python from now on ("" finds one again); kept in <data>/ablation/python.json.
    void choose_python(const std::string& executable);
    // Installs the tool's packages in the background, after downloading a portable Python into `ablate-python` beside
    // the launcher when no Python 3.11 or newer with pip was found; std::invalid_argument when it cannot start (a run
    // uses the packages, say).
    void setup();
    // Downloads and converts a prompt set in the background; std::invalid_argument when it cannot start.
    void fetch(const std::string& preset);
    // Starts a run: checks the request, runs the tool's `init` (which refuses a configuration it cannot fit, before
    // anything is unloaded), then takes the GPU and runs `auto` on a thread of its own. std::invalid_argument with the
    // reason when it cannot start.
    void start(const AblationRun& run);
    // Keeps trial `trial` of the last run's study as the sidecar `name`: the tool's `choose` grades it on the test pair
    // and exports it, with the GPU taken as for a run. std::invalid_argument with the reason when it cannot start (a
    // run in progress, no finished trials, a name that is taken or not a folder name).
    void choose(int trial, const std::string& name);
    // Asks the run to stop: the tool stops its engines and keeps its finished work (ended after three minutes if it
    // has not). `now` ends it at once, its engines too: work since the last finished trial is lost.
    void stop(bool now = false);
    // A run is taking, or holds, the GPU.
    [[nodiscard]] bool busy() const;
    // The launcher is exiting: the run is stopped and awaited.
    void shutdown();

    [[nodiscard]] std::filesystem::path sidecars_dir() const { return sidecars_; }

private:
    struct Python {
        std::filesystem::path executable;
        std::string version; // "3.13.5"
        int major = 0;
        int minor = 0;
    };
    struct Task {
        std::string kind; // "setup" or "fetch"
        std::string preset;
        std::string state; // running, done, failed
        std::string message;
        std::deque<std::string> log;
        std::int64_t started = 0;
        std::int64_t ended   = 0;
    };
    struct Choice {
        int trial = 0;
        std::string name; // the sidecar's folder name
    };
    struct Job {
        AblationRun run;
        std::filesystem::path run_dir;
        std::optional<Choice> choice; // a `choose` job: the trial it keeps; else a fit (`auto`)
        // waiting_gpu, running, stopping, done, no_sidecar, stopped, failed
        std::string state;
        std::string message;
        std::int64_t started = 0;
        std::int64_t ended   = 0;
        Json progress        = nullptr; // the run's progress.json
        std::deque<std::string> log;
        std::optional<std::string> exported; // the sidecar.json written
    };

    // Looks for a Python if nothing has yet (state() does on its first call): a request that comes before the page's
    // first read, from a page left open across a launcher restart say, must not act as if none were installed.
    void ensure_detected();
    [[nodiscard]] std::vector<std::string> tool_args(std::vector<std::string> args) const;
    [[nodiscard]] EnvOverlay tool_env(const EnvOverlay& extra = {}) const;
    [[nodiscard]] bool packages_ready_locked() const;
    [[nodiscard]] Json fetched_presets() const;
    [[nodiscard]] Json job_json_locked() const;
    void load_catalogue();
    // The portable Python's interpreter (it need not exist).
    [[nodiscard]] std::filesystem::path portable_python() const;
    // Downloads the pinned portable Python, checks its SHA-256 and unpacks it into portable_ (task thread): "" when it
    // is in place, else why not.
    std::string install_portable();
    // A line in the running task's log.
    void task_line(std::string line);
    // Runs `task` on the task thread: `prepare` (on that thread) names the executable or says why it cannot run,
    // then the executable runs with `args`; `finish` turns its exit status into an error ("" when it succeeded). The
    // caller holds control_mutex_ and has set task_running_.
    void run_task(Task task, std::function<std::string(std::filesystem::path& executable)> prepare,
                  std::vector<std::string> args, EnvOverlay env, std::function<std::string(int code)> finish);
    void job_main();
    // The checks start and choose share (control_mutex_ held): the launcher, the tool, Python and its packages are
    // ready and no run or installation is under way. Returns the Python to run.
    [[nodiscard]] Python ready_for_job();
    // Installs `job` as the current one and starts its thread (control_mutex_ held).
    void launch_job(Job job);
    void finish_job(const std::string& state, const std::string& message);
    // The engines a run's tool left (run/engines/<pid>.json) when it was ended: they are ended too.
    void end_leftover_engines(const std::filesystem::path& run_dir);
    void save_job_locked() const;
    void publish();

    std::filesystem::path data_dir_;
    std::filesystem::path root_;
    std::filesystem::path prompts_;
    std::filesystem::path runs_;
    std::filesystem::path sidecars_;
    std::filesystem::path packages_;
    std::filesystem::path portable_; // `ablate-python` beside the launcher
    std::optional<std::filesystem::path> tool_;
    std::vector<std::string> tool_looked_in_;
    Hooks hooks_;

    std::mutex detect_mutex_;  // one detection at a time (it runs processes; mutex_ is not held meanwhile)
    std::mutex control_mutex_; // start, setup, fetch and shutdown: the threads they start and join
    mutable std::mutex mutex_; // the state below
    bool detected_ = false;
    std::optional<Python> python_;
    std::string python_chosen_; // the setting the detection used
    Json python_tried_ = Json::array();
    Json catalogue_    = nullptr;
    std::string catalogue_error_;

    std::optional<Task> task_;
    bool task_running_ = false;
    ChildProcess* task_child_ = nullptr; // while it runs: shutdown ends it
    std::atomic<bool> cancel_task_{false}; // shutdown: a download in progress ends
    std::thread task_thread_;

    std::optional<Job> job_;
    bool job_active_ = false; // the job thread runs
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> stop_now_{false};
    bool shutting_down_ = false;
    std::thread job_thread_;
};

} // namespace infernix::launcher
