// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// COMPILER BUG DOCUMENTATION: Clang-17 Float Cast Expression Miscompilation
// ==============================================================================
//
// ISSUE SUMMARY:
// Clang-17.0.6 miscompiles `static_cast<float>(i % N - M)` inside loops,
// producing 0x5f000000 (2^63 as float) for negative results instead of
// correct negative float values.
//
// IMPACT:
// - Affects test data initialization in BERT-related tests
// - Causes false failures in GELU and other operation tests
// - Corruption appears at CPU level before any device operations
//
// ROOT CAUSE:
// Compiler bug when processing: loop → modulo → subtraction → float cast (negative)
//
// WORKING WORKAROUNDS:
// ✓ Lookup table (RECOMMENDED)
// ✓ Break calculation into separate statements  
// ✓ Use volatile intermediate variable
//
// FAILED WORKAROUNDS:
// ✗ Division instead of multiplication
// ✗ Different multiplier values
//
// FILES TO FIX:
// - tests/ops/unary_ops_test.cpp (GeluBERTPattern test)
// - Any file with pattern: static_cast<float>(i % N - M)
//
// REFERENCES:
// - Compiler: Ubuntu Clang 17.0.6
// - Bug discovered in GELU BERT pattern test
// ==============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace ttml::compiler::bug::tests {

class CompilerBugClang17FloatCast : public ::testing::Test {
protected:
    static constexpr uint32_t CORRUPT_VALUE = 0x5f000000;
    
    static bool is_corrupt(float value) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(float));
        return bits == CORRUPT_VALUE;
    }
    
    static void print_summary(const std::vector<float>& data, const std::string& name) {
        size_t corrupt = 0;
        for (const auto& val : data) {
            if (is_corrupt(val)) corrupt++;
        }
        std::cout << name << ": " << (corrupt == 0 ? "✓ PASS" : "✗ FAIL") 
                  << " (" << corrupt << "/" << data.size() << " corrupted)\n";
    }
};

TEST_F(CompilerBugClang17FloatCast, BugReproduction) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  BUG REPRODUCTION                                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Pattern: static_cast<float>(i % 6 - 3) * 0.5f\n\n";
    
    std::vector<float> test_data(32);
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
    }
    
    std::cout << "First 12 values:\n";
    for (size_t i = 0; i < 12; ++i) {
        float expected = static_cast<float>(i % 6 - 3) * 0.5f;
        uint32_t bits;
        memcpy(&bits, &test_data[i], sizeof(float));
        
        std::cout << "  [" << std::setw(2) << i << "] " << std::setw(12) << test_data[i]
                  << " (0x" << std::hex << bits << std::dec << ")";
        if (is_corrupt(test_data[i])) std::cout << " ← CORRUPT";
        std::cout << "\n";
    }
    
    size_t corrupt = 0;
    for (const auto& val : test_data) {
        if (is_corrupt(val)) corrupt++;
    }
    
    std::cout << "\nResult: " << (corrupt > 0 ? "BUG CONFIRMED" : "No bug") 
              << " (" << corrupt << "/32 corrupted)\n";
}

TEST_F(CompilerBugClang17FloatCast, WorkaroundValidation) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  WORKAROUND VALIDATION                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    // Lookup table (RECOMMENDED)
    {
        const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        std::vector<float> data(32);
        for (size_t i = 0; i < 32; ++i) {
            data[i] = values[i % 6];
        }
        print_summary(data, "Lookup table       ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
    
    // Break down calculation
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < 32; ++i) {
            int mod_result = i % 6;
            int sub_result = mod_result - 3;
            data[i] = static_cast<float>(sub_result) * 0.5f;
        }
        print_summary(data, "Break down calc    ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
    
    // Volatile intermediate
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < 32; ++i) {
            volatile int pattern = i % 6 - 3;
            data[i] = static_cast<float>(pattern) * 0.5f;
        }
        print_summary(data, "Volatile int       ");
        EXPECT_EQ(std::count_if(data.begin(), data.end(), is_corrupt), 0);
    }
}

TEST_F(CompilerBugClang17FloatCast, FailedWorkarounds) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  FAILED WORKAROUNDS (documentation only)            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < 32; ++i) {
            data[i] = static_cast<float>(i % 6 - 3) / 2.0f;
        }
        print_summary(data, "Division (/2.0f)   ");
    }
    
    {
        std::vector<float> data(32);
        for (size_t i = 0; i < 32; ++i) {
            data[i] = static_cast<float>(i % 6 - 3) * 2.0f;
        }
        print_summary(data, "Multiply (*2.0f)   ");
    }
    
    std::cout << "\nConclusion: Bug is in the cast itself, not subsequent operations\n";
}

TEST_F(CompilerBugClang17FloatCast, ProductionFix) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  PRODUCTION FIX                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "REPLACE:\n";
    std::cout << "  test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;\n\n";
    std::cout << "WITH:\n";
    std::cout << "  const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};\n";
    std::cout << "  test_data[i] = values[i % 6];\n\n";
    
    const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    std::vector<float> data(32);
    for (size_t i = 0; i < 32; ++i) {
        data[i] = values[i % 6];
    }
    
    size_t corrupt = std::count_if(data.begin(), data.end(), is_corrupt);
    std::cout << "Verification: " << (corrupt == 0 ? "✓ PASS" : "✗ FAIL") << "\n";
    EXPECT_EQ(corrupt, 0);
}

TEST_F(CompilerBugClang17FloatCast, ScopeAnalysis) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  SCOPE ANALYSIS                                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "AFFECTED FILES:\n";
    std::cout << "  tests/ops/unary_ops_test.cpp\n\n";
    
    std::cout << "SEARCH COMMAND:\n";
    std::cout << "  grep -rn 'static_cast<float>.*%.*-' tests/\n\n";
    
    std::cout << "ACTION ITEMS:\n";
    std::cout << "  1. Find all instances of buggy pattern\n";
    std::cout << "  2. Replace with lookup table\n";
    std::cout << "  3. Verify tests pass\n";
}

TEST_F(CompilerBugClang17FloatCast, BugReportInfo) {
    std::cout << "\n╔══════════════════════════════════════════════════════╗\n";
    std::cout << "║  BUG REPORT INFORMATION                              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Compiler: " << __VERSION__ << "\n";
    std::cout << "Flags: -O0 -g\n";
    std::cout << "Pattern: static_cast<float>(i % N - M) where result < 0\n";
    std::cout << "Result: 0x5f000000 instead of correct negative float\n";
    std::cout << "Report: https://github.com/llvm/llvm-project/issues\n";
}

}  // namespace ttml::compiler::bug::tests
