#pragma once

// TEMPORARY (INFERNIX_TMP_MACHINE_CALIBRATION=<file>, candidates §7.3 D11): the last good startup
// measurements of this machine (the PCIe link rate, the CPU expert team's rates), kept in a text file
// so that a startup whose probe reads far low (another process busy for a moment, a probe flake: the
// CPU team once read 10,840 against ~16,400 experts/s and slowed the whole session by a fifth)
// re-measures, and keeps the saved rate when the second reading is low too. A reading within
// kOutlierShare of the saved rate, or above it, replaces it. Each value has a key naming what it
// depends on (the GPU's PCI bus id, the worker count and placement, the expert arithmetic, the record
// stride), so another configuration keeps its own line. Without the switch nothing is read or written.
//
// The file holds one line per value, "key value", the value in decimal.

#include <optional>
#include <string>

namespace infernix {

class MachineCalibration {
public:
    // A reading below this share of the saved value is an outlier.
    static constexpr double kOutlierShare = 0.85;

    // The switch's file, or none (the switch unset).
    [[nodiscard]] static std::optional<MachineCalibration> from_switch();

    explicit MachineCalibration(std::string path);

    // The saved value of `key`, if any.
    [[nodiscard]] std::optional<double> saved(const std::string& key) const;
    // Saves `value` under `key` (rewrites the file; other keys are kept).
    void store(const std::string& key, double value) const;

    [[nodiscard]] static bool outlier(double measured, double saved) noexcept {
        return saved > 0.0 && measured < kOutlierShare * saved;
    }

    // The value to use for `key`: `measured`, unless it is an outlier against the saved value, in which
    // case `remeasure()` is taken once more and the better reading stands unless it is an outlier too,
    // when the saved value does. A reading that stands is saved. `note` says which was taken.
    template <class Remeasure>
    double resolve(const std::string& key, double measured, Remeasure&& remeasure, std::string& note) const {
        const std::optional<double> before = saved(key);
        if (before && outlier(measured, *before)) {
            const double again = remeasure();
            measured           = again > measured ? again : measured;
            if (outlier(measured, *before)) {
                note = "saved " + std::to_string(*before) + " kept: measured " + std::to_string(measured) +
                       " twice below " + std::to_string(kOutlierShare) + " of it";
                return *before;
            }
            note = "re-measured after a low first reading";
        }
        store(key, measured);
        return measured;
    }

private:
    std::string path_;
};

} // namespace infernix
