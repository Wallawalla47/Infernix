#include "infernix/types.h"

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <utility>

namespace infernix {

CancellationView::CancellationView(std::function<bool()> requested)
    : requested_(std::move(requested)) {}

bool CancellationView::requested() const { return requested_ && requested_(); }

struct PrefixCacheSaveControl::State {
    std::mutex mutex;
    std::condition_variable ended;
    std::atomic<bool> abandoned{false};
    bool saving = false;
    bool saved  = false;
};

PrefixCacheSaveControl::PrefixCacheSaveControl() : state_(std::make_shared<State>()) {}

PrefixCacheSaveControl::Abandon
PrefixCacheSaveControl::abandon(std::chrono::milliseconds timeout) const noexcept {
    try {
        std::unique_lock lock(state_->mutex);
        state_->abandoned.store(true, std::memory_order_relaxed);
        if (!state_->ended.wait_for(lock, timeout, [&] { return !state_->saving; })) {
            return Abandon::StillWriting;
        }
        return state_->saved ? Abandon::Saved : Abandon::Unsaved;
    } catch (...) { return Abandon::StillWriting; }
}

bool PrefixCacheSaveControl::begin() const noexcept {
    std::lock_guard lock(state_->mutex);
    if (state_->abandoned.load(std::memory_order_relaxed)) { return false; }
    state_->saving = true;
    return true;
}

bool PrefixCacheSaveControl::abandoned() const noexcept {
    return state_->abandoned.load(std::memory_order_relaxed);
}

void PrefixCacheSaveControl::end(bool saved) const noexcept {
    {
        std::lock_guard lock(state_->mutex);
        state_->saving = false;
        state_->saved  = state_->saved || saved;
    }
    state_->ended.notify_all();
}

} // namespace infernix
