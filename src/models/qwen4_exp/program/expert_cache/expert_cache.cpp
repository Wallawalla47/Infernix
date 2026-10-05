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

void LfruPolicy::evict(std::uint32_t key) {
    if (resident(key)) { erase(key); }
}

void LfruPolicy::set_capacity(std::uint32_t capacity, std::vector<std::uint32_t>& victims) {
    if (residents_.size() > capacity) {
        const std::size_t first = victims.size();
        select_victims(residents_.size() - capacity, {}, victims);
        for (std::size_t i = first; i < victims.size(); ++i) { erase(victims[i]); }
    }
    capacity_ = capacity;
}

// ---------------------------------------------------------------------------- frames

FramePool::FramePool(std::uint32_t frames, std::uint32_t rounds_in_flight, std::uint32_t max_frames)
    : rounds_in_flight_(rounds_in_flight), backed_(frames), state_(std::max(frames, max_frames), kUnbacked) {
    free_.reserve(state_.size());
    for (std::uint32_t f = frames; f-- > 0;) { // acquire hands out frame 0 first
        state_[f] = kFree;
        free_.push_back(f);
    }
}

std::optional<std::uint32_t> FramePool::acquire_below(std::uint32_t limit) {
    for (std::size_t i = free_.size(); i-- > 0;) {
        const std::uint32_t f = free_[i];
        if (f >= limit) { continue; }
        free_[i] = free_.back();
        free_.pop_back();
        state_[f] = kHeld;
        return f;
    }
    return std::nullopt;
}

void FramePool::resize(std::uint32_t frames) {
    if (frames > state_.size()) { throw std::invalid_argument("frame pool resized beyond its maximum"); }
    if (frames < backed_) {
        for (std::uint32_t f = frames; f < backed_; ++f) {
            if (state_[f] != kFree) { throw std::logic_error("frame pool shrunk over a frame in use"); }
            state_[f] = kUnbacked;
        }
        free_.erase(std::remove_if(free_.begin(), free_.end(), [frames](std::uint32_t f) { return f >= frames; }),
                    free_.end());
    } else {
        for (std::uint32_t f = backed_; f < frames; ++f) {
            state_[f] = kFree;
            free_.push_back(f);
        }
    }
    backed_ = frames;
}

void FramePool::lend(std::uint32_t first, std::uint32_t count) {
    if (count == 0 || first + count > backed_) { throw std::invalid_argument("frame lending outside the backed pool"); }
    for (std::uint32_t f = first; f < first + count; ++f) {
        if (state_[f] != kFree) { throw std::logic_error("frame lent while not free"); }
        state_[f] = kLoaned;
    }
    free_.erase(std::remove_if(free_.begin(), free_.end(),
                               [&](std::uint32_t f) { return f >= first && f < first + count; }),
                free_.end());
    loaned_ += count;
}

void FramePool::give_back(std::uint32_t first, std::uint32_t count) {
    if (first + count > backed_) { throw std::invalid_argument("frame give-back outside the backed pool"); }
    for (std::uint32_t f = first; f < first + count; ++f) {
        if (state_[f] != kLoaned) { throw std::logic_error("frame given back while not lent"); }
        state_[f] = kFree;
        free_.push_back(f);
    }
    loaned_ -= count;
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
                                 std::uint32_t rounds_in_flight, std::uint32_t max_frames)
    : policy_(num_keys, frames > slack_frames ? frames - slack_frames : 0), slack_(slack_frames),
      frames_(frames, rounds_in_flight, max_frames), table_(num_keys), is_queued_(num_keys, 0),
      load_serial_(num_keys, 0), frame_key_(frames_.max_frames(), kNoKey) {}

void CacheController::load(std::uint32_t key, std::uint32_t frame, std::vector<Command>& out) {
    // ABSENT -> LOADING(f), copy, READY(f): all on the copy stream, in this order.
    table_[key] = table_[key].next(ResidencyState::kLoading, frame);
    out.push_back({Command::Kind::kWriteEntry, key, frame, table_[key].encode()});
    load_serial_[key] = ++next_serial_;
    frame_key_[frame] = key;
    out.push_back({Command::Kind::kCopy, key, frame, 0, 0, load_serial_[key]});
    table_[key] = table_[key].next(ResidencyState::kReady, frame);
    out.push_back({Command::Kind::kWriteEntry, key, frame, table_[key].encode()});
}

std::uint32_t CacheController::seed(std::span<const std::uint32_t> keys, std::span<const std::uint32_t> counts,
                                    std::vector<Command>& out) {
    policy_.seed({}, counts);
    std::uint32_t loaded = 0;
    for (const std::uint32_t key : keys) {
        if (key >= table_.size() || policy_.resident(key) || policy_.resident_count() >= policy_.capacity()) { continue; }
        const auto frame = frames_.acquire();
        if (!frame) { break; }
        const std::uint32_t one[] = {key};
        policy_.seed(one, {});
        load(key, *frame, out);
        ++loaded;
    }
    return loaded;
}

std::uint32_t CacheController::reserve_free(std::uint32_t count, std::vector<std::uint32_t>& out) {
    const std::size_t residents = policy_.resident_count();
    const std::size_t room      = policy_.capacity() > residents ? policy_.capacity() - residents : 0;
    std::uint32_t reserved      = 0;
    while (reserved < count && reserved < room) {
        const auto frame = frames_.acquire();
        if (!frame) { break; }
        out.push_back(*frame);
        ++reserved;
    }
    return reserved;
}

bool CacheController::adopt(std::uint32_t key, std::uint32_t frame, std::span<const std::uint32_t> protect,
                            std::uint64_t round_started, std::vector<Command>& out) {
    if (key >= table_.size() || table_[key].state != ResidencyState::kAbsent) { return false; }
    if (const auto victim = policy_.promote(key, protect)) {
        if (table_[*victim].state != ResidencyState::kAbsent) { frames_.retire(make_absent(*victim, out), round_started); }
    }
    // A queued load of the key is dropped by drain_queue (the key is no longer absent); a copy of
    // it still in flight never publishes (complete_load compares the serial).
    table_[key] = table_[key].next(ResidencyState::kLoading, frame);
    table_[key] = table_[key].next(ResidencyState::kReady, frame);
    out.push_back({Command::Kind::kWriteEntry, key, frame, table_[key].encode()});
    load_serial_[key] = ++next_serial_;
    frame_key_[frame] = key;
    return true;
}

void CacheController::release(std::uint32_t frame) { frames_.release_now(frame); }

bool CacheController::complete_load(std::uint32_t key, std::uint32_t frame, std::uint64_t serial) const {
    return table_[key].state == ResidencyState::kReady && table_[key].frame == frame && load_serial_[key] == serial;
}

void CacheController::drain_queue(std::vector<Command>& out) {
    while (!queued_.empty()) {
        const std::uint32_t key = queued_.front();
        // Evicted before it could load, or landed by a round meanwhile (adopt).
        if (!policy_.resident(key) || table_[key].state != ResidencyState::kAbsent) {
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
        frames_.retire(make_absent(v, out), round_started);
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

void CacheController::resize(std::uint32_t frames, std::vector<Command>& out) {
    if (frames_.loaned_count() != 0) { throw std::logic_error("expert cache resized while frames are lent"); }
    frames_.on_quiescent();
    const std::uint32_t capacity = frames > slack_ ? frames - slack_ : 0;
    if (frames >= frames_.backed()) {
        frames_.resize(frames);
        std::vector<std::uint32_t> none;
        policy_.set_capacity(capacity, none);
        drain_queue(out);
        return;
    }
    std::vector<std::uint32_t> victims;
    policy_.set_capacity(capacity, victims);
    for (std::uint32_t v : victims) {
        if (table_[v].state == ResidencyState::kAbsent) { continue; } // still queued: drain drops it
        frames_.release_now(make_absent(v, out));
    }
    // Survivors above the new top move into free frames below it: at most `capacity` experts are
    // resident, so frames below the top suffice.
    for (std::uint32_t key : policy_.residents()) {
        if (table_[key].state == ResidencyState::kAbsent || table_[key].frame < frames) { continue; }
        const auto target = frames_.acquire_below(frames);
        if (!target) { throw std::logic_error("frame pool shrink found no relocation target"); }
        const std::uint32_t from = table_[key].frame;
        table_[key]              = table_[key].next(ResidencyState::kReady, *target);
        out.push_back({Command::Kind::kRelocate, key, *target, table_[key].encode(), from});
        frame_key_[*target] = key;
        frame_key_[from]    = kNoKey;
        frames_.release_now(from);
    }
    frames_.resize(frames);
}

std::uint32_t CacheController::make_absent(std::uint32_t key, std::vector<Command>& out) {
    const std::uint32_t frame = table_[key].frame;
    table_[key]               = table_[key].next(ResidencyState::kAbsent, 0);
    out.push_back({Command::Kind::kWriteEntry, key, frame, table_[key].encode()});
    frame_key_[frame] = kNoKey;
    return frame;
}

std::optional<std::uint32_t> CacheController::frame_key(std::uint32_t frame) const {
    if (frame >= frame_key_.size() || frame_key_[frame] == kNoKey) { return std::nullopt; }
    return frame_key_[frame];
}

std::optional<std::uint32_t> CacheController::choose_run(std::uint32_t count, std::span<const std::uint8_t> busy) const {
    const std::uint32_t n = frames_.backed();
    if (count == 0 || count > n) { return std::nullopt; }
    // Sliding window over the backed frames: the summed score of held experts (free frames 0),
    // with lent and retiring frames and, in the first pass, busy frames breaking the window.
    std::optional<std::uint32_t> best;
    for (const bool avoid_busy : {true, false}) {
        double best_score = 0.0, sum = 0.0;
        std::uint32_t run = 0; // eligible frames ending at f
        for (std::uint32_t f = 0; f < n; ++f) {
            const bool held     = frame_key_[f] != kNoKey;
            const bool eligible = !frames_.is_loaned(f) && (frames_.is_free(f) || held) &&
                                  !(avoid_busy && f < busy.size() && busy[f] != 0);
            if (!eligible) {
                run = 0;
                sum = 0.0;
                continue;
            }
            sum += held ? policy_.score(frame_key_[f]) : 0.0;
            if (++run > count) {
                const std::uint32_t out = f - count;
                sum -= frame_key_[out] != kNoKey ? policy_.score(frame_key_[out]) : 0.0;
                run = count;
            }
            if (run == count && (!best || sum < best_score)) {
                best_score = sum;
                best       = f + 1 - count;
            }
        }
        if (best) { return best; }
    }
    return std::nullopt;
}

std::uint32_t CacheController::lend(std::uint32_t first, std::uint32_t count, std::vector<Command>& out) {
    if (frames_.pending_count() != 0) { throw std::logic_error("frames lent while some still retire"); }
    std::uint32_t evicted = 0;
    for (std::uint32_t f = first; f < first + count; ++f) {
        if (frame_key_[f] == kNoKey) { continue; }
        const std::uint32_t key = frame_key_[f];
        policy_.evict(key);
        make_absent(key, out);
        frames_.release_now(f);
        ++evicted;
    }
    frames_.lend(first, count);
    // The policy keeps no more residents than the frames it may still use.
    const std::uint32_t capacity = policy_.capacity() > count ? policy_.capacity() - count : 0;
    std::vector<std::uint32_t> victims;
    policy_.set_capacity(capacity, victims);
    for (std::uint32_t v : victims) {
        if (table_[v].state == ResidencyState::kAbsent) { continue; } // still queued: drain drops it
        frames_.release_now(make_absent(v, out));
        ++evicted;
    }
    return evicted;
}

void CacheController::give_back(std::uint32_t first, std::uint32_t count, std::vector<Command>& out) {
    frames_.give_back(first, count);
    std::vector<std::uint32_t> none;
    policy_.set_capacity(policy_.capacity() + count, none);
    drain_queue(out);
}

} // namespace ninfer::models::qwen4_exp::expert_cache
