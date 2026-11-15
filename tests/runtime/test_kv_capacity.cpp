#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

template <class Attempt>
std::string failure_message(const Attempt& attempt) {
    try {
        attempt();
    } catch (const std::invalid_argument& failure) {
        return failure.what();
    }
    return {};
}

} // namespace

int main() {
    int failures = 0;
    const infernix::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        infernix::runtime::resolve_kv_capacity(infernix::KvCapacityPolicy::automatic(), curve, 1360, 50);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        infernix::runtime::resolve_kv_capacity(infernix::KvCapacityPolicy::automatic(), curve, 10000, 50);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = infernix::runtime::resolve_kv_capacity(
        infernix::KvCapacityPolicy::explicit_capacity(129), curve, 1200, 50);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128 &&
                  explicit_capacity.automatic_headroom_bytes == 0,
              "explicit KV capacity did not use page-aligned token semantics or kept a headroom");

    const std::string insufficient_message = failure_message([&] {
        (void)infernix::runtime::resolve_kv_capacity(infernix::KvCapacityPolicy::automatic(), curve,
                                                   1049, 50);
    });
    failures += check(!insufficient_message.empty(),
                      "automatic KV capacity accepted less than the minimum reservation");
    failures += check(
        insufficient_message ==
            "automatic KV capacity requires 1050 bytes total (1000 bytes minimum Engine runtime "
            "reservation + 50 bytes automatic headroom), but only 1049 bytes are available after "
            "weights",
        "automatic KV capacity error does not state the combined reservation + headroom total");

    const std::string explicit_message = failure_message([&] {
        (void)infernix::runtime::resolve_kv_capacity(infernix::KvCapacityPolicy::explicit_capacity(384),
                                                   curve, 1400, 50);
    });
    failures += check(
        explicit_message ==
            "requested Engine runtime reservation requires 1512 bytes, but only 1400 bytes are "
            "available after weights",
        "explicit KV capacity error does not state the after-weights allowance");

    // A plan grown beside the ordinary one (idle-wide-prefill's wider prefill buffers): same KV
    // stride, 100 more bytes of fixed reservation.
    {
        using infernix::KvCapacityPolicy;
        using infernix::runtime::resolve_kv_capacity;
        using infernix::runtime::resolve_kv_capacity_beside;
        infernix::runtime::SequenceCapacityCurve grown = curve;
        grown.minimum_device_reservation_bytes         = 1100;

        // Automatic: the ordinary plan takes 4 page groups of a 1310-byte budget, the grown one 3.
        const auto ordinary = resolve_kv_capacity(KvCapacityPolicy::automatic(), curve, 1360, 50);
        const auto lanes_fit =
            resolve_kv_capacity_beside(KvCapacityPolicy::automatic(), grown, 1360, 50, ordinary, 3);
        failures += check(ordinary.main_page_groups == 4 && lanes_fit &&
                              lanes_fit->main_page_groups == 3 &&
                              lanes_fit->runtime_reservation_bytes == 1228,
                          "automatic capacity may shrink to the lanes' full contexts");
        failures += check(!resolve_kv_capacity_beside(KvCapacityPolicy::automatic(), grown, 1360,
                                                      50, ordinary, 4),
                          "automatic capacity must not shrink below the lanes' full contexts");
        const auto roomy = resolve_kv_capacity(KvCapacityPolicy::automatic(), curve, 10000, 50);
        const auto capped = resolve_kv_capacity_beside(KvCapacityPolicy::automatic(), grown, 10000,
                                                       50, roomy, 6);
        failures += check(capped && capped->main_page_groups == 6,
                          "a roomy automatic budget keeps the full capacity");
        const auto tight = resolve_kv_capacity(KvCapacityPolicy::automatic(), curve, 1140, 50);
        failures += check(tight.main_page_groups == 2 &&
                              !resolve_kv_capacity_beside(KvCapacityPolicy::automatic(), grown,
                                                          1140, 50, tight, 2),
                          "a grown plan below its minimum reservation does not fit");

        // Explicit: the requested capacity is kept whole or the grown plan does not fit.
        const KvCapacityPolicy requested = KvCapacityPolicy::explicit_capacity(129);
        const auto explicit_ordinary     = resolve_kv_capacity(requested, curve, 1300, 50);
        const auto explicit_fit =
            resolve_kv_capacity_beside(requested, grown, 1300, 50, explicit_ordinary, 18);
        failures += check(explicit_fit && explicit_fit->main_page_groups == 3 &&
                              explicit_fit->runtime_reservation_bytes == 1228,
                          "an explicit capacity fits beside the grown reservation");
        const auto explicit_short = resolve_kv_capacity(requested, curve, 1200, 50);
        failures += check(!resolve_kv_capacity_beside(requested, grown, 1200, 50, explicit_short, 18),
                          "an explicit capacity that no longer fits rejects the grown plan");
    }

    // What an automatic capacity may plan in under an OS budget: the headroom comes off free device
    // memory and the margin off the budget, never both off the smaller of the two.
    {
        using infernix::runtime::kv_available_bytes;
        constexpr std::size_t kHeadroom = 1024, kMargin = 64;
        const auto planned = [&](std::size_t free_bytes, std::optional<std::uint64_t> budget) {
            return kv_available_bytes(true, free_bytes, budget, kHeadroom, kMargin) - kHeadroom;
        };
        failures += check(planned(20000, std::nullopt) == 18976, "no budget: free memory less the headroom");
        failures += check(planned(20000, 30000) == 18976, "free memory below the budget: the headroom alone");
        failures += check(planned(20000, 19500) == 18976,
                          "a budget below free memory by less than the headroom is not stacked on it");
        failures += check(planned(20000, 19000) == 18936,
                          "between them, the budget's margin binds once it is the smaller");
        failures += check(planned(20000, 17000) == 16936,
                          "a budget below free memory by more than the headroom leaves its margin only");
        failures += check(kv_available_bytes(true, 20000, 40, kHeadroom, kMargin) == kHeadroom,
                          "a budget inside its margin leaves nothing to plan");
        // Past the budget the source reports free memory as the widened budget less usage and the caller
        // names no budget: the headroom comes off it once, as before.
        failures += check(planned(19500, std::nullopt) == 18476, "past the budget: the headroom off the widened budget");
        // An explicit capacity has no headroom and must fit both.
        failures += check(kv_available_bytes(false, 20000, 17000, kHeadroom, kMargin) == 17000 &&
                              kv_available_bytes(false, 16000, 17000, kHeadroom, kMargin) == 16000 &&
                              kv_available_bytes(false, 16000, std::nullopt, kHeadroom, kMargin) == 16000,
                          "an explicit capacity fits free memory and the budget");
        // Through the resolver: 2.2 units of budget shortfall no longer cost the headroom as well.
        infernix::runtime::SequenceCapacityCurve wide = curve;
        wide.maximum_main_page_groups                = 1000;
        const auto stacked = infernix::runtime::resolve_kv_capacity(infernix::KvCapacityPolicy::automatic(), wide,
                                                                    17000, kHeadroom);
        const auto fixed   = infernix::runtime::resolve_kv_capacity(
            infernix::KvCapacityPolicy::automatic(), wide, kv_available_bytes(true, 20000, 17000, kHeadroom, kMargin),
            kHeadroom);
        failures += check(stacked.main_page_groups == 2 + (17000 - 1024 - 1000) / 128 &&
                              fixed.main_page_groups == 2 + (17000 - 64 - 1000) / 128,
                          "the resolved capacity takes the budget less its margin");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
