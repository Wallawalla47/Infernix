#include "ops/op_family.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace infernix::test {
namespace {

std::vector<std::pair<std::string_view, OpTestEntry>>& registry() {
    static std::vector<std::pair<std::string_view, OpTestEntry>> tests;
    return tests;
}

} // namespace

OpTestRegistration::OpTestRegistration(const char* name, OpTestEntry entry) {
    registry().emplace_back(name, entry);
}

int run_op_family(int argc, char** argv) {
    auto& tests = registry();
    std::sort(tests.begin(), tests.end());
    if (argc >= 2 && std::strcmp(argv[1], "--list") == 0) {
        for (const auto& [name, entry] : tests) { std::cout << name << '\n'; }
        return 0;
    }
    if (argc >= 2) {
        for (const auto& [name, entry] : tests) {
            if (name == argv[1]) { return entry(argc - 1, argv + 1); }
        }
    }
    std::cerr << "usage: " << (argc > 0 ? argv[0] : "op test family") << " <test> [arguments...] | --list\n";
    return 2;
}

} // namespace infernix::test

int main(int argc, char** argv) { return infernix::test::run_op_family(argc, argv); }
