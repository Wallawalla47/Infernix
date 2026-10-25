#pragma once

// Op tests run from one executable per family (tests/ops/tests.cmake): each test keeps its own entry
// point, registered under its CTest name's stem (infernix_<name>_test), and the family's main runs one
// by name before any CUDA work starts:
//   infernix_ops_<family>_test <name> [arguments...]   (--list: the family's tests)
// The entry receives argv[0] = <name> and the test's own arguments after it.

namespace infernix::test {

using OpTestEntry = int (*)(int argc, char** argv);

struct OpTestRegistration {
    OpTestRegistration(const char* name, OpTestEntry entry);
};

int run_op_family(int argc, char** argv);

} // namespace infernix::test

// Defines the test's entry point in place of main: INFERNIX_OP_TEST(argmax) { ... }.
#define INFERNIX_OP_TEST(name)                                                                     \
    static int infernix_op_test_##name([[maybe_unused]] int argc, [[maybe_unused]] char** argv);  \
    static const ::infernix::test::OpTestRegistration infernix_op_test_registration_##name{        \
        #name, &infernix_op_test_##name};                                                          \
    static int infernix_op_test_##name([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
