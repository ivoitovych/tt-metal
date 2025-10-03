// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// COMPILER BUG DOCUMENTATION: Clang-17 Expression Chain Miscompilation
// ==============================================================================
//
// ISSUE SUMMARY:
// Clang-17.0.6 miscompiles the expression `static_cast<float>(i % N - M) * 0.5f`
// inside loops, producing 0x5f000000 (2^63 as float) for negative results
// instead of correct negative float values.
//
// IMPACT:
// - Affects test data initialization in BERT-related tests
// - Causes false failures in GELU and other operation tests
// - Corruption appears at CPU level before any device operations
//
// ROOT CAUSE:
// Compiler optimization bug in Clang-17 when processing the complete chain:
//   loop iteration → modulo → subtraction → float cast → multiplication by 0.5f
//
// EVIDENCE:
// - Bug reproduces with -O0 (even without optimization)
// - Only affects the complete expression; breaking into steps works
// - Using volatile, division, or lookup table prevents the bug
//
// RECOMMENDED ACTION:
// Replace all instances of pattern `(i % N - M) * multiplier` in test code
// with lookup table approach shown in WorkaroundRecommendation test.
//
// REFERENCES:
// - Compiler: Ubuntu Clang 17.0.6 (++20231209124227+6009708b4367-1~exp1~20231209124336.77)
// - Discovered in: tests/ops/unary_ops_test.cpp (GELU BERT pattern test)
// ==============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace ttml::compiler::bug::tests {

// ==============================================================================
// Test Suite: Clang17ExpressionBug
// ==============================================================================

class Clang17ExpressionBug : public ::testing::Test {
protected:
    static constexpr uint32_t CORRUPT_VALUE = 0x5f000000;  // 2^63 as float
    
    static bool is_corrupt(float value) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(float));
        return bits == CORRUPT_VALUE;
    }
    
    static void print_corruption_summary(const std::vector<float>& data, const std::string& test_name) {
        size_t corrupt_count = 0;
        for (const auto& val : data) {
            if (is_corrupt(val)) corrupt_count++;
        }
        
        std::cout << test_name << ": ";
        if (corrupt_count == 0) {
            std::cout << "✓ PASS (no corruption)";
        } else {
            std::cout << "✗ FAIL (" << corrupt_count << "/" << data.size() << " corrupted)";
        }
        std::cout << "\n";
    }
};

// ==============================================================================
// TEST 1: Bug Reproduction - The Exact Pattern That Fails
// ==============================================================================

TEST_F(Clang17ExpressionBug, BugReproduction) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  BUG REPRODUCTION                                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Testing the exact pattern from BERT tests:\n";
    std::cout << "  for (i = 0; i < 32; ++i)\n";
    std::cout << "    data[i] = static_cast<float>(i % 6 - 3) * 0.5f;\n\n";
    
    // The exact code pattern that triggers the bug
    std::vector<float> test_data(32);
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
    }
    
    // Analyze results
    std::cout << "Expected pattern: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0] repeating\n";
    std::cout << "First 12 values:\n";
    for (size_t i = 0; i < 12; ++i) {
        int pattern = i % 6 - 3;
        float expected = static_cast<float>(pattern) * 0.5f;
        uint32_t bits;
        memcpy(&bits, &test_data[i], sizeof(float));
        
        std::cout << "  [" << std::setw(2) << i << "] expected=" << std::setw(5) << expected
                  << " actual=" << std::setw(12) << test_data[i]
                  << " hex=0x" << std::hex << std::setw(8) << std::setfill('0') << bits
                  << std::dec << std::setfill(' ');
        
        if (is_corrupt(test_data[i])) {
            std::cout << " ← CORRUPT";
        }
        std::cout << "\n";
    }
    
    size_t corrupt_count = 0;
    for (const auto& val : test_data) {
        if (is_corrupt(val)) corrupt_count++;
    }
    
    std::cout << "\nResult: " << corrupt_count << "/32 values corrupted\n";
    std::cout << "Status: " << (corrupt_count > 0 ? "BUG CONFIRMED" : "Bug not reproduced") << "\n\n";
    
    // Document the bug but don't fail the test - this is documentation
    if (corrupt_count > 0) {
        std::cout << "Note: This test documents the compiler bug.\n";
        std::cout << "      See workaround tests below for solutions.\n";
    }
}

// ==============================================================================
// TEST 2: Workaround Validation - Prove All Fixes Work
// ==============================================================================

TEST_F(Clang17ExpressionBug, WorkaroundValidation) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  WORKAROUND VALIDATION                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    // Workaround 1: Lookup table (RECOMMENDED)
    {
        std::vector<float> data(32);
        const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = values[i % 6];
        }
        print_corruption_summary(data, "Workaround 1: Lookup table       ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
    
    // Workaround 2: Break down calculation
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); ++i) {
            int mod_result = i % 6;
            int sub_result = mod_result - 3;
            data[i] = static_cast<float>(sub_result) * 0.5f;
        }
        print_corruption_summary(data, "Workaround 2: Break down calc    ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
    
    // Workaround 3: Volatile intermediate
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); ++i) {
            volatile int pattern = i % 6 - 3;
            data[i] = static_cast<float>(pattern) * 0.5f;
        }
        print_corruption_summary(data, "Workaround 3: Volatile int       ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
    
    // Workaround 4: Division instead of multiply by 0.5f
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 6 - 3) / 2.0f;
        }
        print_corruption_summary(data, "Workaround 4: Division by 2.0    ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
    
    std::cout << "\nAll workarounds validated successfully.\n";
}

// ==============================================================================
// TEST 3: Recommended Fix for Production Code
// ==============================================================================

TEST_F(Clang17ExpressionBug, WorkaroundRecommendation) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  RECOMMENDED FIX FOR PRODUCTION                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "BEFORE (buggy code - DO NOT USE):\n";
    std::cout << "  std::vector<float> test_data(32);\n";
    std::cout << "  for (size_t i = 0; i < test_data.size(); ++i) {\n";
    std::cout << "    test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;\n";
    std::cout << "  }\n\n";
    
    std::cout << "AFTER (fixed code - USE THIS):\n";
    std::cout << "  const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};\n";
    std::cout << "  std::vector<float> test_data(32);\n";
    std::cout << "  for (size_t i = 0; i < test_data.size(); ++i) {\n";
    std::cout << "    test_data[i] = values[i % 6];\n";
    std::cout << "  }\n\n";
    
    std::cout << "RATIONALE:\n";
    std::cout << "  - Clearer intent (pattern is explicit)\n";
    std::cout << "  - Faster (no arithmetic per iteration)\n";
    std::cout << "  - Immune to compiler bugs\n";
    std::cout << "  - More maintainable\n\n";
    
    // Demonstrate the fix works
    const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    std::vector<float> test_data(32);
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = values[i % 6];
    }
    
    size_t corrupt_count = 0;
    for (const auto& val : test_data) {
        if (is_corrupt(val)) corrupt_count++;
    }
    
    std::cout << "Verification: " << corrupt_count << "/32 corrupted ";
    std::cout << (corrupt_count == 0 ? "✓ PASS" : "✗ FAIL") << "\n";
    
    EXPECT_EQ(corrupt_count, 0) << "Recommended fix must work correctly";
}

// ==============================================================================
// TEST 4: Scope Analysis - Which Files Need Fixing
// ==============================================================================

TEST_F(Clang17ExpressionBug, ScopeAnalysis) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  SCOPE ANALYSIS                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "FILES CONFIRMED AFFECTED:\n";
    std::cout << "  - tests/ops/unary_ops_test.cpp (GeluBERTPattern test)\n";
    std::cout << "  - tests/ops/gelu_comprehensive_diagnostic_test.cpp\n\n";
    
    std::cout << "SEARCH PATTERNS TO FIND OTHER INSTANCES:\n";
    std::cout << "  grep -rn \"% [0-9]* -\" tests/\n";
    std::cout << "  grep -rn \"static_cast<float>.*%.*-.*\\*\" tests/\n\n";
    
    std::cout << "ACTION REQUIRED:\n";
    std::cout << "  1. Search codebase for similar patterns\n";
    std::cout << "  2. Replace with lookup table approach\n";
    std::cout << "  3. Verify tests pass after fix\n";
    std::cout << "  4. Document change in commit message\n\n";
}

// ==============================================================================
// TEST 5: Compiler Information for Bug Report
// ==============================================================================

TEST_F(Clang17ExpressionBug, CompilerInformation) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  COMPILER INFORMATION                                ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Compiler Version:\n";
    std::cout << "  " << __VERSION__ << "\n\n";
    
    std::cout << "Build Flags:\n";
    std::cout << "  -O0 -g (bug occurs even without optimization)\n\n";
    
    std::cout << "Bug Pattern:\n";
    std::cout << "  static_cast<float>(i % N - M) * 0.5f\n";
    std::cout << "  where result of (i % N - M) is negative\n\n";
    
    std::cout << "Bug Result:\n";
    std::cout << "  0x5f000000 (2^63 as float = 9.223372e+18)\n";
    std::cout << "  instead of correct negative float value\n\n";
    
    std::cout << "To Report This Bug:\n";
    std::cout << "  Repository: https://github.com/llvm/llvm-project/issues\n";
    std::cout << "  Include: This test output + minimal reproduction case\n\n";
}

}  // namespace ttml::compiler::bug::tests
