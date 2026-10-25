#include "core/machine_calibration.h"

#include "core/temporary_switches.h"

#include <fstream>
#include <sstream>
#include <utility>
#include <vector>

namespace infernix {
namespace {

std::vector<std::pair<std::string, std::string>> read_lines(const std::string& path) {
    std::vector<std::pair<std::string, std::string>> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto space = line.rfind(' ');
        if (space == std::string::npos || space == 0) { continue; }
        out.emplace_back(line.substr(0, space), line.substr(space + 1));
    }
    return out;
}

} // namespace

std::optional<MachineCalibration> MachineCalibration::from_switch() {
    const std::optional<std::string> path = temporary_switches::value("INFERNIX_TMP_MACHINE_CALIBRATION");
    if (!path || path->empty()) { return std::nullopt; }
    return MachineCalibration(*path);
}

MachineCalibration::MachineCalibration(std::string path) : path_(std::move(path)) {}

std::optional<double> MachineCalibration::saved(const std::string& key) const {
    for (const auto& [k, v] : read_lines(path_)) {
        if (k != key) { continue; }
        try {
            return std::stod(v);
        } catch (...) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

void MachineCalibration::store(const std::string& key, double value) const {
    auto lines = read_lines(path_);
    std::ostringstream text;
    text.precision(17);
    text << value;
    bool replaced = false;
    for (auto& [k, v] : lines) {
        if (k == key) {
            v        = text.str();
            replaced = true;
        }
    }
    if (!replaced) { lines.emplace_back(key, text.str()); }
    std::ofstream out(path_, std::ios::trunc);
    for (const auto& [k, v] : lines) { out << k << ' ' << v << '\n'; }
}

} // namespace infernix
