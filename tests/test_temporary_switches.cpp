// INFERNIX_TMP_CONTINUING_PROMPT (core/temporary_switches.h): describe(), which every Qwen-family Engine start calls,
// parses it, so a malformed value fails the start instead of the first offloaded prompt pass, and a fraction in (0, 1]
// is read as given (unset: 0); Inkling 2's list leaves it out. The CTest entries set the variable; the argument is
// "malformed" or the fraction expected.

#include "core/temporary_switches.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    namespace switches = infernix::temporary_switches;
    if (argc != 2) {
        std::cerr << "usage: infernix_temporary_switches_test malformed|<expected fraction>\n";
        return 2;
    }
    const std::string expected = argv[1];
    // Inkling 2's startup line lists the switches that act on it (the decode graphs' edge census), not the
    // offloaded-expert ones, and an offload switch's value, malformed or not, does not fail its start.
    try {
        const std::string inkling = switches::describe(switches::Scope::Inkling2);
        if (inkling.find("INFERNIX_TMP_GRAPH_EDGES=") == std::string::npos ||
            inkling.find("INFERNIX_TMP_CONTINUING_PROMPT") != std::string::npos ||
            inkling.find("INFERNIX_TMP_LINK_SHARE") != std::string::npos ||
            inkling.find("INFERNIX_TMP_GW_SHAPE") != std::string::npos) {
            std::cerr << "Inkling 2's switches listed wrongly: " << inkling << '\n';
            return 1;
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << "describe(Inkling2) refused an offload switch's value: " << error.what() << '\n';
        return 1;
    }
    try {
        (void)switches::describe(switches::Scope::All);
    } catch (const std::invalid_argument& error) {
        if (expected == "malformed") { return 0; }
        std::cerr << "describe() refused a valid INFERNIX_TMP_CONTINUING_PROMPT: " << error.what() << '\n';
        return 1;
    }
    if (expected == "malformed") {
        std::cerr << "describe() accepted a malformed INFERNIX_TMP_CONTINUING_PROMPT\n";
        return 1;
    }
    if (switches::continuing_prompt() != std::stod(expected)) {
        std::cerr << "INFERNIX_TMP_CONTINUING_PROMPT read as " << switches::continuing_prompt() << ", expected "
                  << expected << '\n';
        return 1;
    }
    std::cout << "PASS temporary switches\n";
    return 0;
}
