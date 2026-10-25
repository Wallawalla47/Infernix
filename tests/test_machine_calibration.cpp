// MachineCalibration (core/machine_calibration.h, INFERNIX_TMP_MACHINE_CALIBRATION): values saved and read
// back per key, other keys kept on a rewrite, and resolve()'s rule: a reading within 0.85 of the saved
// value (or above it) stands and is saved; a low one is measured again, and the saved value is kept
// when the second reading is low too.

#include "core/machine_calibration.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    const auto path = (std::filesystem::temp_directory_path() / "infernix_machine_calibration_test.txt").string();
    std::filesystem::remove(path);
    const infernix::MachineCalibration calibration(path);
    check(!calibration.saved("link|0000:01:00.0|1").has_value(), "nothing saved yet");
    calibration.store("link|0000:01:00.0|1", 27.5e9);
    calibration.store("cpu|6|pinned P-cores|0|0|1|experts", 16400.0);
    check(calibration.saved("link|0000:01:00.0|1") == 27.5e9, "a value reads back");
    calibration.store("link|0000:01:00.0|1", 27.0e9);
    check(calibration.saved("link|0000:01:00.0|1") == 27.0e9, "a store replaces its key");
    check(calibration.saved("cpu|6|pinned P-cores|0|0|1|experts") == 16400.0, "other keys (spaces included) are kept");

    std::string note;
    int remeasured = 0;
    const std::string key = "cpu|6|pinned P-cores|0|0|1|experts";
    // Within the share: stands and is saved, no re-measure.
    double rate = calibration.resolve(key, 15000.0, [&] { ++remeasured; return 0.0; }, note);
    check(rate == 15000.0 && remeasured == 0 && note.empty(), "a reading within 0.85 stands");
    check(calibration.saved(key) == 15000.0, "and is saved");
    // Low once, normal again: the second reading stands.
    rate = calibration.resolve(key, 10840.0, [&] { ++remeasured; return 15500.0; }, note);
    check(rate == 15500.0 && remeasured == 1 && !note.empty(), "a low reading is measured again");
    // Low twice: the saved value is kept and not overwritten.
    note.clear();
    rate = calibration.resolve(key, 10000.0, [&] { ++remeasured; return 11000.0; }, note);
    check(rate == 15500.0 && remeasured == 2 && !note.empty(), "two low readings keep the saved value");
    check(calibration.saved(key) == 15500.0, "the saved value survives two low readings");
    std::filesystem::remove(path);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "machine calibration: all checks passed\n";
    return 0;
}
