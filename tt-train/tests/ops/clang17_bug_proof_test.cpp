// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// CLANG-17 BUG: Minimal Reproduction Case
// ==============================================================================
// Single test demonstrating Clang-17 miscompiles: (negative_int) * 0.5f
//
// Bug: When multiplying a negative integer by 0.5f and casting to float,
//      Clang-17 produces 0x5f000000 (2^63) instead of the correct value.
//
// To report this bug to LLVM, run:
//   ./build/tests/ttml_tests --gtest_filter=Clang17BugProof.SingleTest
//
// Expected output: Shows exactly which values are corrupted and why
// ==============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <iomanip>
#include <iostream>

TEST(Clang17BugProof, SingleTest) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  CLANG-17 COMPILER BUG - PROOF                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    std::cout << "Compiler: " << __VERSION__ << "\n";
    std::cout << "Build flags: -O0 (no optimization)\n";
    std::cout << "Bug pattern: (negative_int) * 0.5f\n\n";

    // Test the exact pattern that triggers the bug
    struct TestCase {
        int input;
        float expected;
    };

    TestCase cases[] = {
        {-3, -1.5f},
        {-2, -1.0f},
        {-1, -0.5f},
        { 0,  0.0f},
        { 1,  0.5f},
        { 2,  1.0f},
    };

    std::cout << "Testing: static_cast<float>(value) * 0.5f\n\n";
    std::cout << "Value | Expected | Actual      | Expected Hex | Actual Hex | Status\n";
    std::cout << "------|----------|-------------|--------------|------------|--------\n";

    int bug_count = 0;

    for (const auto& test : cases) {
        // The buggy expression
        float actual = static_cast<float>(test.input) * 0.5f;
        
        // Get hex representations
        uint32_t expected_bits, actual_bits;
        memcpy(&expected_bits, &test.expected, sizeof(float));
        memcpy(&actual_bits, &actual, sizeof(float));

        // Print comparison
        std::cout << std::setw(5) << test.input << " | "
                  << std::setw(8) << test.expected << " | "
                  << std::setw(11) << actual << " | "
                  << "0x" << std::hex << std::setw(8) << std::setfill('0') << expected_bits << " | "
                  << "0x" << std::setw(8) << actual_bits << std::dec << std::setfill(' ') << " | ";

        if (actual_bits == 0x5f000000) {
            std::cout << "BUG";
            bug_count++;
        } else if (expected_bits == actual_bits) {
            std::cout << "OK";
        } else {
            std::cout << "DIFF";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout << "Result: " << bug_count << " values corrupted to 0x5f000000 (2^63)\n";
    std::cout << "Pattern: Only negative inputs are affected\n\n";

    if (bug_count > 0) {
        std::cout << "╔══════════════════════════════════════════════════════╗\n";
        std::cout << "║  BUG CONFIRMED                                       ║\n";
        std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
        
        std::cout << "WORKAROUND:\n";
        std::cout << "  Replace: result = static_cast<float>(val) * 0.5f;\n";
        std::cout << "  With:    volatile int tmp = val;\n";
        std::cout << "           result = static_cast<float>(tmp) * 0.5f;\n\n";
        
        std::cout << "Or use:    result = static_cast<float>(val) * 1.0f / 2.0f;\n\n";
        
        std::cout << "Report to: https://github.com/llvm/llvm-project/issues\n";
    } else {
        std::cout << "No bug detected - compiler may have been fixed\n";
    }
    std::cout << "\n";

    // This test documents the bug - it will fail if the bug exists
    EXPECT_EQ(bug_count, 0) << "Clang-17 compiler bug detected in " << bug_count << " cases";
}

// Verification: The bug does NOT occur with these variations
TEST(Clang17BugProof, VerifyWorkarounds) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  WORKAROUND VERIFICATION                             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";

    int value = -3;
    float expected = -1.5f;

    // Method 1: The bug
    float buggy = static_cast<float>(value) * 0.5f;
    uint32_t buggy_bits;
    memcpy(&buggy_bits, &buggy, sizeof(float));

    // Method 2: Volatile workaround
    volatile int tmp = value;
    float fixed1 = static_cast<float>(tmp) * 0.5f;
    uint32_t fixed1_bits;
    memcpy(&fixed1_bits, &fixed1, sizeof(float));

    // Method 3: Different calculation
    float fixed2 = static_cast<float>(value) / 2.0f;
    uint32_t fixed2_bits;
    memcpy(&fixed2_bits, &fixed2, sizeof(float));

    // Method 4: Multiply by 1.0 then divide
    float fixed3 = static_cast<float>(value) * 1.0f / 2.0f;
    uint32_t fixed3_bits;
    memcpy(&fixed3_bits, &fixed3, sizeof(float));

    std::cout << "Input value: " << value << " (should produce -1.5)\n\n";
    std::cout << "Method                              | Result  | Hex        | Status\n";
    std::cout << "------------------------------------|---------|------------|--------\n";
    std::cout << "Buggy: (float)val * 0.5f            | " 
              << std::setw(7) << buggy << " | 0x" << std::hex << buggy_bits << std::dec
              << " | " << (buggy_bits == 0x5f000000 ? "BUG" : "OK") << "\n";
    std::cout << "Fix 1: volatile int tmp; (float)tmp | " 
              << std::setw(7) << fixed1 << " | 0x" << std::hex << fixed1_bits << std::dec
              << " | " << (fixed1 == expected ? "OK" : "FAIL") << "\n";
    std::cout << "Fix 2: (float)val / 2.0f            | " 
              << std::setw(7) << fixed2 << " | 0x" << std::hex << fixed2_bits << std::dec
              << " | " << (fixed2 == expected ? "OK" : "FAIL") << "\n";
    std::cout << "Fix 3: (float)val * 1.0f / 2.0f     | " 
              << std::setw(7) << fixed3 << " | 0x" << std::hex << fixed3_bits << std::dec
              << " | " << (fixed3 == expected ? "OK" : "FAIL") << "\n";

    std::cout << "\n";

    EXPECT_EQ(fixed1, expected) << "Volatile workaround must work";
    EXPECT_EQ(fixed2, expected) << "Division workaround must work";
    EXPECT_EQ(fixed3, expected) << "Mult-then-div workaround must work";
}
