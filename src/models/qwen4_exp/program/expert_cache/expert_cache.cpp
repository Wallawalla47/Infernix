#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::expert_cache {

// ---------------------------------------------------------------------------- residency entries

std::uint32_t ResidencyEntry::encode() const {
    if (frame > kMaxFrame || generation > 63) { throw std::invalid_argument("residency entry out of range"); }
    return (static_cast<std::uint32_t>(state) << 30) | (generation << 24) | frame;
}

ResidencyEntry ResidencyEntry::decode(std::uint32_t word) {
    const std::uint32_t state = word >> 30;
    if (state > 2) { throw std::invalid_argument("invalid residency state"); }
    return {static_cast<ResidencyState>(state), (word >> 24) & 63U, word & kMaxFrame};
}

ResidencyEntry ResidencyEntry::next(ResidencyState new_state, std::uint32_t new_frame) const {
    return {new_state, (generation + 1) & 63U, new_state == ResidencyState::kAbsent ? 0U : new_frame};
}

// ---------------------------------------------------------------------------- LFRU

LfruPolicy::LfruPolicy(std::uint32_t num_keys, std::uint32_t capacity, std::uint32_t halving_period)
    : capacity_(capacity), halving_period_(halving_period), count_(num_keys, 0), last_(num_keys, 0),
      slot_(num_keys, kNone), mark_(num_keys, 0) {
    if (capacity == 0) { throw std::invalid_argument("LFRU capacity must be positive"); }
    residents_.reserve(capacity);
}

double LfruPolicy::score(std::uint32_t key) const {
    return static_cast<double>(count_[key]) / static_cast<double>(now_ - last_[key] + 1);
}

void LfruPolicy::insert(std::uint32_t key) {
    slot_[key] = static_cast<std::uint32_t>(residents_.size());
    residents_.push_back(key);
}

void LfruPolicy::erase(std::uint32_t key) {
    const std::uint32_t at   = slot_[key];
    const std::uint32_t tail = residents_.back();
    residents_[at]           = tail;
    slot_[tail]              = at;
    residents_.pop_back();
    slot_[key] = kNone;
}

void LfruPolicy::select_victims(std::size_t need, std::span<const std::uint32_t> protect,
                                std::vector<std::uint32_t>& out) {
    for (std::uint32_t k : protect) { mark_[k] = 1; }
    scratch_.clear();
    for (std::uint32_t k : residents_) {
        if (mark_[k] == 0) { scratch_.emplace_back(score(k), k); }
    }
    for (std::uint32_t k : protect) { mark_[k] = 0; }
    need = std::min(need, scratch_.size());
    std::partial_sort(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(need), scratch_.end());
    for (std::size_t i = 0; i < need; ++i) { out.push_back(scratch_[i].second); }
}

void LfruPolicy::step(std::span<const std::uint32_t> group, Step& out, std::size_t admission_budget) {
    out.hits.clear();
    out.misses.clear();
    out.admitted.clear();
    out.victims.clear();
    ++now_;
    if (halving_period_ != 0 && now_ % halving_period_ == 0) {
        for (auto& c : count_) { c /= 2; }
    }
    for (std::uint32_t k : group) {
        ++count_[k];
        last_[k] = now_;
        (resident(k) ? out.hits : out.misses).push_back(k);
    }
    if (out.misses.empty()) { return; }
    if (admission_budget >= out.misses.size()) {
        const std::size_t total = residents_.size() + out.misses.size();
        if (total > capacity_) { select_victims(total - capacity_, group, out.victims); }
        for (std::uint32_t v : out.victims) { erase(v); }
        for (std::uint32_t k : out.misses) {
            if (residents_.size() < capacity_) {
                insert(k);
                out.admitted.push_back(k);
            }
        }
        return;
    }
    // Budgeted admission: best candidates first, each only if it outranks the victim it replaces.
    std::vector<std::uint32_t> candidates = out.misses;
    std::sort(candidates.begin(), candidates.end(), [this](std::uint32_t a, std::uint32_t b) {
        const double sa = score(a), sb = score(b);
        return sa != sb ? sa > sb : a < b;
    });
    for (std::uint32_t k : candidates) {
        if (out.admitted.size() >= admission_budget) { break; }
        if (residents_.size() >= capacity_) {
            std::vector<std::uint32_t> v;
            select_victims(1, group, v);
            if (v.empty() || !(score(k) > score(v[0]))) { break; }
            erase(v[0]);
            out.victims.push_back(v[0]);
        }
        insert(k);
        out.admitted.push_back(k);
    }
}

std::optional<std::uint32_t> LfruPolicy::promote(std::uint32_t key, std::span<const std::uint32_t> protect) {
    if (resident(key)) { return std::nullopt; }
    std::optional<std::uint32_t> victim;
    if (residents_.size() >= capacity_) {
        std::vector<std::uint32_t> v;
        select_victims(1, protect, v);
        if (v.empty()) { throw std::logic_error("LFRU: every resident is protected"); }
        erase(v[0]);
        victim = v[0];
    }
    insert(key);
    return victim;
}

void LfruPolicy::seed(std::span<const std::uint32_t> resident_keys, std::span<const std::uint32_t> counts) {
    if (!counts.empty()) {
        if (counts.size() != count_.size()) { throw std::invalid_argument("LFRU seed: count size mismatch"); }
        std::copy(counts.begin(), counts.end(), count_.begin());
    }
    for (std::uint32_t k : resident_keys) {
        if (!resident(k) && residents_.size() < capacity_) { insert(k); }
    }
}

// ---------------------------------------------------------------------------- frames

FramePool::FramePool(std::uint32_t frames, std::uint32_t rounds_in_flight)
    : rounds_in_flight_(rounds_in_flight), state_(frames, kFree) {
    free_.reserve(frames);
    for (std::uint32_t f = frames; f-- > 0;) { free_.push_back(f); } // acquire hands out frame 0 first
}

std::optional<std::uint32_t> FramePool::acquire() {
    if (free_.empty()) { return std::nullopt; }
    const std::uint32_t f = free_.back();
    free_.pop_back();
    state_[f] = kHeld;
    return f;
}

void FramePool::release_now(std::uint32_t frame) {
    if (state_.at(frame) != kHeld) { throw std::logic_error("frame released while not held"); }
    state_[frame] = kFree;
    free_.push_back(frame);
}

void FramePool::retire(std::uint32_t frame, std::uint64_t round_started_sample) {
    if (state_.at(frame) != kHeld) { throw std::logic_error("frame retired while not held"); }
    state_[frame] = kPending;
    pending_.emplace_back(round_started_sample + rounds_in_flight_, frame);
}

void FramePool::on_round_done(std::uint64_t round_done) {
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->first <= round_done) {
            state_[it->second] = kFree;
            free_.push_back(it->second);
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
}

void FramePool::on_quiescent() {
    for (const auto& [round, frame] : pending_) {
        state_[frame] = kFree;
        free_.push_back(frame);
    }
    pending_.clear();
}

// ---------------------------------------------------------------------------- controller

CacheController::CacheController(std::uint32_t num_keys, std::uint32_t frames, std::uint32_t slack_frames,
                                 std::uint32_t rounds_in_flight)
    : policy_(num_keys, frames - slack_frames), frames_(frames, rounds_in_flight), table_(num_keys),
      is_queued_(num_keys, 0) {
    if (slack_frames >= frames) { throw std::invalid_argument("slack must leave frames for experts"); }
}

void CacheController::load(std::uint32_t key, std::uint32_t frame, std::vector<Command>& out) {
    // ABSENT -> LOADING(f), copy, READY(f): all on the copy stream, in this order.
    table_[key] = table_[key].next(ResidencyState::kLoading, frame);
    out.push_back({Command::Kind::kWriteEntry, key, frame, table_[key].encode()});
    out.push_back({Command::Kind::kCopy, key, frame, 0});
    table_[key] = table_[key].next(ResidencyState::kReady, frame);
    out.push_back({Command::Kind::kWriteEntry, key, frame, table_[key].encode()});
}

void CacheController::drain_queue(std::vector<Command>& out) {
    while (!queued_.empty()) {
        const std::uint32_t key = queued_.front();
        if (!policy_.resident(key)) { // evicted before it could load
            queued_.pop_front();
            is_queued_[key] = 0;
            continue;
        }
        auto frame = frames_.acquire();
        if (!frame) { return; }
        queued_.pop_front();
        is_queued_[key] = 0;
        load(key, *frame, out);
    }
}

void CacheController::on_route(std::span<const std::uint32_t> group, std::uint64_t round_started,
                               std::vector<Command>& out, std::size_t admission_budget) {
    policy_.step(group, step_, admission_budget);
    for (std::uint32_t v : step_.victims) {
        if (table_[v].state == ResidencyState::kAbsent) { continue; } // was still queued
        const std::uint32_t frame = table_[v].frame;
        table_[v] = table_[v].next(ResidencyState::kAbsent, 0);
        out.push_back({Command::Kind::kWriteEntry, v, frame, table_[v].encode()});
        frames_.retire(frame, round_started);
    }
    for (std::uint32_t k : step_.admitted) {
        if (!is_queued_[k]) {
            queued_.push_back(k);
            is_queued_[k] = 1;
        }
    }
    drain_queue(out);
}

void CacheController::on_round_done(std::uint64_t round_done, std::vector<Command>& out) {
    frames_.on_round_done(round_done);
    drain_queue(out);
}

void CacheController::on_quiescent(std::vector<Command>& out) {
    frames_.on_quiescent();
    drain_queue(out);
}

} // namespace ninfer::models::qwen4_exp::expert_cache
