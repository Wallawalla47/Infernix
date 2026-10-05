#include "ninfer/ops/token_constraint.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kRecordCriterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-5,
    /*gross_relative_to_max_reference=*/0.0,
};

constexpr std::int32_t kChoices = ops::kTokenConstraintChoices;
constexpr std::int32_t kRecord  = ops::kTokenConstraintRecord;
const std::uint16_t kNegativeInfinity = 0xff80;

std::vector<std::uint16_t> random_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                         std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + seed) * 1013904223u;
            const float value =
                row < valid_rows ? -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f)
                                 : 96.0f;
            logits[static_cast<std::size_t>(column) * physical_rows + row] = f32_to_bf16(value);
        }
    }
    return logits;
}

struct Expected {
    std::vector<std::uint16_t> logits;
    std::vector<double> records;
    std::vector<std::int32_t> argmax;
};

// Independent FP64 evaluation of the contract from the represented BF16 inputs.
Expected oracle(const std::vector<std::uint16_t>& logits, std::int32_t physical_rows,
                std::int32_t valid_rows, const std::vector<std::int32_t>& descriptors,
                const std::vector<std::int32_t>& choices, const std::vector<std::int32_t>& counts,
                std::int32_t untouched_argmax, double untouched_record) {
    Expected expected{logits,
                      std::vector<double>(descriptors.size() * kRecord, untouched_record),
                      std::vector<std::int32_t>(descriptors.size(), untouched_argmax)};
    for (std::size_t column = 0; column < descriptors.size(); ++column) {
        const std::int32_t descriptor = descriptors[column];
        if (descriptor == -1) { continue; }
        std::vector<std::int32_t> permitted;
        if (descriptor >= 0) {
            if (descriptor >= static_cast<std::int32_t>(counts.size())) { continue; }
            for (std::int32_t i = 0; i < counts[descriptor]; ++i) {
                permitted.push_back(choices[descriptor * kChoices + i]);
            }
        } else {
            permitted.push_back(-(descriptor + 2));
        }
        // A set naming a token outside the valid rows leaves its column untouched.
        if (std::any_of(permitted.begin(), permitted.end(),
                        [&](std::int32_t token) { return token < 0 || token >= valid_rows; })) {
            continue;
        }
        const std::size_t base = column * static_cast<std::size_t>(physical_rows);
        const auto value       = [&](std::int32_t row) {
            return static_cast<double>(bf16_to_f32(logits[base + row]));
        };
        double full = 0.0;
        double top  = -std::numeric_limits<double>::infinity();
        for (std::int32_t row = 0; row < valid_rows; ++row) { top = std::max(top, value(row)); }
        for (std::int32_t row = 0; row < valid_rows; ++row) { full += std::exp(value(row) - top); }
        double restricted = 0.0;
        std::int32_t best = permitted.front();
        for (const std::int32_t token : permitted) {
            restricted += std::exp(value(token) - top);
            if (value(token) > value(best) || (value(token) == value(best) && token < best)) {
                best = token;
            }
        }
        for (std::int32_t i = 0; i < kChoices; ++i) {
            expected.records[column * kRecord + i] =
                i < static_cast<std::int32_t>(permitted.size())
                    ? std::exp(value(permitted[i]) - top) / restricted
                    : 0.0;
        }
        expected.records[column * kRecord + kChoices] = restricted / full;
        expected.argmax[column]                       = best;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            if (std::find(permitted.begin(), permitted.end(), row) == permitted.end()) {
                expected.logits[base + row] = kNegativeInfinity;
            }
        }
    }
    return expected;
}

int run_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
             const std::vector<std::uint16_t>& logits, const std::vector<std::int32_t>& descriptors,
             const std::vector<std::int32_t>& choices, const std::vector<std::int32_t>& counts,
             bool with_argmax) {
    const auto columns = static_cast<std::int32_t>(descriptors.size());
    const auto sets    = static_cast<std::int32_t>(counts.size());
    // Unwritten outputs keep the 0xcd fill: argmax 0xcdcdcdcd, records the float of that pattern.
    constexpr std::int32_t kFilledI32 = static_cast<std::int32_t>(0xcdcdcdcdu);
    const float filled_f32            = [] {
        float value         = 0.0f;
        const std::uint32_t bits = 0xcdcdcdcdu;
        static_assert(sizeof(value) == sizeof(bits));
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }();
    const Expected expected = oracle(logits, physical_rows, valid_rows, descriptors, choices,
                                     counts, kFilledI32, static_cast<double>(filled_f32));

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_descriptors(descriptors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_choices(choices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_counts(counts.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_argmax(descriptors.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_records(descriptors.size() * kRecord * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    device_descriptors.copy_from_host(descriptors.data(), device_descriptors.bytes());
    device_choices.copy_from_host(choices.data(), device_choices.bytes());
    device_counts.copy_from_host(counts.data(), device_counts.bytes());
    device_argmax.fill(0xcd);
    device_records.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    Tensor descriptor_tensor(device_descriptors.data(), DType::I32, {columns});
    Tensor choice_tensor(device_choices.data(), DType::I32, {kChoices, sets});
    Tensor count_tensor(device_counts.data(), DType::I32, {sets});
    Tensor argmax_tensor(device_argmax.data(), DType::I32, {columns});
    Tensor record_tensor(device_records.data(), DType::FP32, {kRecord, columns});
    ops::constrain_logits(logits_tensor, with_argmax ? &argmax_tensor : nullptr, descriptor_tensor,
                          choice_tensor, count_tensor, valid_rows, record_tensor, nullptr);
    cuda_synchronize();

    const auto records = from_device<float>(device_records.data(), expected.records.size());
    std::vector<double> constrained_records;
    std::vector<double> constrained_expected;
    std::vector<std::uint32_t> untouched_bits;
    for (std::size_t column = 0; column < descriptors.size(); ++column) {
        for (std::int32_t i = 0; i < kRecord; ++i) {
            const std::size_t index = column * kRecord + static_cast<std::size_t>(i);
            if (expected.argmax[column] == kFilledI32) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &records[index], sizeof(bits));
                untouched_bits.push_back(bits);
            } else {
                constrained_records.push_back(records[index]);
                constrained_expected.push_back(expected.records[index]);
            }
        }
    }
    int failures = 0;
    if (!constrained_records.empty()) {
        failures += verify_reduction(label + " records", constrained_records, constrained_expected,
                                     kRecordCriterion);
    }
    failures += verify_exact((label + " leaves unconstrained records").c_str(), untouched_bits,
                             std::vector<std::uint32_t>(untouched_bits.size(), 0xcdcdcdcdu));
    failures += verify_exact((label + " masked logits").c_str(),
                             from_device<std::uint16_t>(device_logits.data(), logits.size()),
                             expected.logits);
    if (with_argmax) {
        failures += verify_exact((label + " argmax").c_str(),
                                 from_device<std::int32_t>(device_argmax.data(), descriptors.size()),
                                 expected.argmax);
    }
    failures += verify_exact((label + " preserves descriptors").c_str(),
                             from_device<std::int32_t>(device_descriptors.data(), descriptors.size()),
                             descriptors);
    failures += device_logits.verify_guards(label + " logits guards");
    failures += device_argmax.verify_guards(label + " argmax guards");
    failures += device_records.verify_guards(label + " record guards");
    return failures;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(8 * 3 * sizeof(std::uint16_t));
    DeviceBuffer descriptor_data(3 * sizeof(std::int32_t));
    DeviceBuffer choice_data(kChoices * 2 * sizeof(std::int32_t));
    DeviceBuffer count_data(2 * sizeof(std::int32_t));
    DeviceBuffer record_data(kRecord * 3 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {8, 3});
    Tensor descriptors(descriptor_data.p, DType::I32, {3});
    Tensor choices(choice_data.p, DType::I32, {kChoices, 2});
    Tensor counts(count_data.p, DType::I32, {2});
    Tensor records(record_data.p, DType::FP32, {kRecord, 3});

    int failures = 0;
    failures += expect_invalid("constrain_logits rejects valid_rows=0", [&] {
        ops::constrain_logits(logits, nullptr, descriptors, choices, counts, 0, records, nullptr);
    });
    failures += expect_invalid("constrain_logits rejects a descriptor count mismatch", [&] {
        Tensor wrong(descriptor_data.p, DType::I32, {2});
        ops::constrain_logits(logits, nullptr, wrong, choices, counts, 8, records, nullptr);
    });
    failures += expect_invalid("constrain_logits rejects a choice table of the wrong width", [&] {
        Tensor wrong(choice_data.p, DType::I32, {kChoices - 1, 2});
        ops::constrain_logits(logits, nullptr, descriptors, wrong, counts, 8, records, nullptr);
    });
    failures += expect_invalid("constrain_logits rejects a short record", [&] {
        Tensor wrong(record_data.p, DType::FP32, {kChoices, 3});
        ops::constrain_logits(logits, nullptr, descriptors, choices, counts, 8, wrong, nullptr);
    });
    failures += expect_invalid("constrain_logits rejects records aliasing logits", [&] {
        Tensor alias(logits_data.p, DType::FP32, {kRecord, 3});
        ops::constrain_logits(logits, nullptr, descriptors, choices, counts, 8, alias, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // Set 0: ten "digits"; set 1: digits plus three structural tokens; set 2: one distinct id.
    std::vector<std::int32_t> choices(3 * kChoices, 0);
    std::vector<std::int32_t> counts{10, 13, 1};
    for (std::int32_t i = 0; i < 10; ++i) { choices[0 * kChoices + i] = 15 + i; }
    for (std::int32_t i = 0; i < 10; ++i) { choices[1 * kChoices + i] = 15 + i; }
    choices[1 * kChoices + 10] = 13;
    choices[1 * kChoices + 11] = 92;
    choices[1 * kChoices + 12] = 248000;
    choices[2 * kChoices]      = 7;

    int failures = 0;
    // Production vocabulary: unconstrained, both sets, forced and an untouched invalid set index.
    const std::vector<std::int32_t> mixed{-1, 0, 1, -2 - 92, 0, -1, 3, 2};
    failures += run_case("constrain_logits full vocabulary", 248320, 248077,
                         random_logits(248320, 248077, 8, 1), mixed, choices, counts, true);
    failures += run_case("constrain_logits without argmax", 248320, 248077,
                         random_logits(248320, 248077, 8, 5), mixed, choices, counts, false);
    // Equal logits: the smallest permitted id wins, and the set is uniform.
    std::vector<std::uint16_t> flat(523 * 4, f32_to_bf16(2.5f));
    const std::vector<std::int32_t> small_choices = [] {
        std::vector<std::int32_t> table(kChoices, 0);
        const std::int32_t ids[] = {400, 31, 202, 7};
        std::copy(std::begin(ids), std::end(ids), table.begin());
        return table;
    }();
    failures += run_case("constrain_logits ties go to the smallest id", 523, 509, flat,
                         {0, -1, 0, -2 - 508}, small_choices, {4}, true);
    // Every column unconstrained: nothing is written.
    failures += run_case("constrain_logits all unconstrained", 523, 509, random_logits(523, 509, 5, 9),
                         {-1, -1, -1, -1, -1}, small_choices, {4}, true);
    // A forced token outside the valid rows and a set naming one are left untouched.
    std::vector<std::int32_t> bad_choices = small_choices;
    bad_choices[1]                        = 515;
    failures += run_case("constrain_logits ignores tokens beyond the valid rows", 523, 509,
                         random_logits(523, 509, 3, 3), {-2 - 515, 0, -1}, bad_choices, {4}, true);
    failures += run_validation_cases();
    if (failures == 0) { std::cout << "constrain_logits tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
