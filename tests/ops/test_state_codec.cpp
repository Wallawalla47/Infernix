#include "infernix/ops/state_codec.h"
#include "ops/op_tester.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

std::uint32_t bits_of(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_of(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

// Independent oracle: `value` rounded to nearest, ties to even, onto the grid of an FP32-range
// format with `mantissa_bits` explicit mantissa bits, by arithmetic on the value (not its bits).
float round_reference(float value, int mantissa_bits) {
    if (std::isnan(value) || std::isinf(value)) { return value; }
    int exponent = 0;
    (void)std::frexp(value, &exponent); // value = m * 2^exponent, 0.5 <= |m| < 1
    // The format keeps FP32's exponent range: its quantum is 2^(E - mantissa_bits) for the FP32
    // exponent E = exponent - 1, and subnormals share the smallest normal binade's quantum.
    const int quantum_exponent =
        std::max(exponent - 1 - mantissa_bits, -126 - mantissa_bits);
    const double quantum = std::ldexp(1.0, quantum_exponent);
    const double rounded = std::nearbyint(static_cast<double>(value) / quantum) * quantum;
    if (std::fabs(rounded) > static_cast<double>(FLT_MAX)) {
        return std::copysign(std::numeric_limits<float>::infinity(), value);
    }
    return static_cast<float>(rounded);
}

// The packed little-endian bytes of `reference`: bits 31..8 (Fp24) or 31..16 (Bf16). A NaN's
// packed bytes are not defined beyond decoding to a NaN; zeros stand in for them.
void append_packed(std::vector<std::uint8_t>& packed, float reference,
                   ops::PackedStateFormat format) {
    const std::uint32_t bits = std::isnan(reference) ? 0U : bits_of(reference);
    const std::uint32_t first = format == ops::PackedStateFormat::Fp24 ? 8U : 16U;
    for (std::uint32_t shift = first; shift < 32U; shift += 8U) {
        packed.push_back(static_cast<std::uint8_t>(bits >> shift));
    }
}

std::vector<float> make_inputs(std::size_t random_count, std::uint32_t seed, int mantissa_bits) {
    std::vector<float> values;
    // Every class of FP32 value by raw bit pattern, then values of a recurrent state's scale.
    std::mt19937 generator(seed);
    std::uniform_int_distribution<std::uint32_t> bits(0U, 0xffffffffU);
    for (std::size_t i = 0; i < random_count; ++i) { values.push_back(float_of(bits(generator))); }
    std::normal_distribution<float> state(0.0F, 0.05F);
    for (std::size_t i = 0; i < random_count; ++i) { values.push_back(state(generator)); }
    // Exact ties on both parities, just below and above them, in a normal and a subnormal binade.
    const std::uint32_t drop = 23U - static_cast<std::uint32_t>(mantissa_bits);
    const std::uint32_t half = 1U << (drop - 1U);
    for (const std::uint32_t base : {0x3f800000U, 0x3f800000U | (1U << drop), 0x00120000U,
                                     0x00120000U | (1U << drop), 0x7f000000U}) {
        for (const std::uint32_t low : {half, half - 1U, half + 1U}) {
            values.push_back(float_of(base | low));
            values.push_back(-float_of(base | low));
        }
    }
    for (const float special :
         {0.0F, -0.0F, FLT_MAX, -FLT_MAX, FLT_MIN, -FLT_MIN, std::numeric_limits<float>::denorm_min(),
          std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
          float_of(0x7f800001U), float_of(0xff812345U)}) {
        values.push_back(special);
    }
    while (values.size() % 4U != 0U) { values.push_back(1.0F); }
    return values;
}

int run_case(const char* label, ops::PackedStateFormat format, std::size_t random_count,
             std::uint32_t seed) {
    const int mantissa_bits = format == ops::PackedStateFormat::Fp24 ? 15 : 7;
    const std::vector<float> source = make_inputs(random_count, seed, mantissa_bits);
    const std::size_t count         = source.size();
    std::vector<float> expected(count);
    std::vector<std::uint8_t> expected_packed;
    for (std::size_t i = 0; i < count; ++i) {
        expected[i] = round_reference(source[i], mantissa_bits);
        append_packed(expected_packed, expected[i], format);
    }
    const std::size_t packed_bytes =
        ops::packed_state_bytes(static_cast<std::int64_t>(count), format);
    if (packed_bytes != expected_packed.size()) {
        std::cerr << label << ": packed_state_bytes disagrees with the packed layout\n";
        return 1;
    }

    GuardedDeviceBuffer device_source(count * sizeof(float));
    GuardedDeviceBuffer device_packed(packed_bytes);
    GuardedDeviceBuffer device_unpacked(count * sizeof(float));
    device_source.copy_from_host(source.data(), device_source.bytes());
    const auto shape = static_cast<std::int32_t>(count);
    Tensor source_tensor(device_source.data(), DType::FP32, {shape});
    Tensor unpacked_tensor(device_unpacked.data(), DType::FP32, {shape});
    ops::pack_fp32_state(source_tensor, device_packed.data(), format, nullptr);
    ops::unpack_fp32_state(device_packed.data(), unpacked_tensor, format, nullptr);
    cuda_synchronize();

    int failures = 0;
    std::vector<std::uint8_t> got_packed =
        from_device<std::uint8_t>(device_packed.data(), packed_bytes);
    const std::size_t width = format == ops::PackedStateFormat::Fp24 ? 3U : 2U;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isnan(source[i])) { continue; }
        for (std::size_t b = 0; b < width; ++b) { got_packed[i * width + b] = 0; }
    }
    failures += verify_exact(label, got_packed, expected_packed);

    const std::vector<float> unpacked = from_device<float>(device_unpacked.data(), count);
    std::vector<std::uint32_t> got_bits(count);
    std::vector<std::uint32_t> want_bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        // A NaN must decode to a NaN; its payload and sign are not defined.
        if (std::isnan(source[i])) {
            got_bits[i]  = std::isnan(unpacked[i]) ? 0U : 1U;
            want_bits[i] = 0U;
            continue;
        }
        got_bits[i]  = bits_of(unpacked[i]);
        want_bits[i] = bits_of(expected[i]);
    }
    failures += verify_exact("unpacked values", got_bits, want_bits);
    failures += device_source.verify_guards("state source");
    failures += device_packed.verify_guards("packed state");
    failures += device_unpacked.verify_guards("unpacked state");
    return failures;
}

int reject_unsupported() {
    GuardedDeviceBuffer source(6U * sizeof(float));
    GuardedDeviceBuffer packed(32U);
    Tensor six(source.data(), DType::FP32, {6});
    try {
        ops::pack_fp32_state(six, packed.data(), ops::PackedStateFormat::Fp24, nullptr);
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "pack_fp32_state accepted an element count not divisible by four\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    try {
        failures += run_case("pack fp24 mixed", ops::PackedStateFormat::Fp24, 4096, 11U);
        failures += run_case("pack bf16 mixed", ops::PackedStateFormat::Bf16, 4096, 13U);
        // One Qwen3.6-27B recurrent layer: 128 x 128 x 48 values (grid-stride loop past the cap).
        failures += run_case("pack fp24 27B layer", ops::PackedStateFormat::Fp24, 393216, 17U);
        failures += run_case("pack bf16 27B layer", ops::PackedStateFormat::Bf16, 393216, 19U);
        failures += reject_unsupported();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " pack_fp32_state / unpack_fp32_state\n";
    return failures ? 1 : 0;
}
