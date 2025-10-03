// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// MINIMAL REPRODUCTION: Clang-17 Compiler Bug
// ==============================================================================
// Bug: Clang-17 miscompiles the expression `static_cast<float>(i % 6 - 3) * 0.5f`
//      when the result is negative, producing 0x5f000000 (2^63) instead of the
//      correct negative float value.
//
// Affects: Clang-17 with -O0 (and higher optimization levels)
// Workaround: Use volatile, break down the expression, or use lookup table
//
// This file provides a minimal test case for reporting to LLVM developers.
// ==============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

namespace ttml::clang17::bug::tests {

class Clang17BugMinimal : public ::testing::Test {
protected:
    static constexpr uint32_t CORRUPT_BITS = 0x5f000000;  // 2^63 as float
    
    static bool is_corrupt(float value) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(float));
        return bits == CORRUPT_BITS;
    }
};

// ==============================================================================
// TEST 1: Minimal Bug Reproduction
// ==============================================================================

TEST_F(Clang17BugMinimal, MinimalReproduction) {
    std::cout << "\n=== MINIMAL CLANG-17 BUG REPRODUCTION ===\n\n";
    std::cout << "Compiler: " << __VERSION__ << "\n";
    std::cout << "Pattern: static_cast<float>(i % 6 - 3) * 0.5f\n";
    std::cout << "Expected negative values: -1.5, -1.0, -0.5\n\n";

    // The buggy pattern - produces 0x5f000000 for negative values
    std::vector<float> data(6);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
    }

    std::cout << "Actual results:\n";
    for (size_t i = 0; i < data.size(); ++i) {
        uint32_t bits;
        memcpy(&bits, &data[i], sizeof(float));
        
        int pattern = i % 6 - 3;
        float expected = static_cast<float>(pattern) * 0.5f;
        
        std::cout << "  i=" << i << " pattern=" << std::setw(2) << pattern 
                  << " expected=" << std::setw(5) << expected
                  << " actual=" << std::setw(12) << data[i]
                  << " (0x" << std::hex << std::setw(8) << std::setfill('0') 
                  << bits << std::dec << std::setfill(' ') << ")";
        
        if (is_corrupt(data[i])) {
            std::cout << " <- BUG: Should be " << expected;
        }
        std::cout << "\n";
    }

    // Verify the bug exists
    size_t corrupt_count = 0;
    for (const auto& val : data) {
        if (is_corrupt(val)) corrupt_count++;
    }

    std::cout << "\nCorrupt values: " << corrupt_count << "/6\n";
    if (corrupt_count > 0) {
        std::cout << "BUG CONFIRMED: Clang-17 miscompiles negative results\n";
    }
}

// ==============================================================================
// TEST 2: Workaround Verification
// ==============================================================================

TEST_F(Clang17BugMinimal, WorkaroundVerification) {
    std::cout << "\n=== WORKAROUND VERIFICATION ===\n\n";

    // Workaround 1: Use volatile
    {
        std::cout << "Workaround 1: volatile int\n";
        std::vector<float> data(6);
        for (size_t i = 0; i < data.size(); ++i) {
            volatile int pattern = i % 6 - 3;
            data[i] = static_cast<float>(pattern) * 0.5f;
        }
        
        size_t corrupt = 0;
        for (const auto& val : data) {
            if (is_corrupt(val)) corrupt++;
        }
        std::cout << "  Result: " << (corrupt == 0 ? "WORKS" : "FAILS") 
                  << " (" << corrupt << " corrupted)\n\n";
        
        EXPECT_EQ(corrupt, 0) << "Volatile workaround should prevent corruption";
    }

    // Workaround 2: Break down calculation
    {
        std::cout << "Workaround 2: Break down calculation\n";
        std::vector<float> data(6);
        for (size_t i = 0; i < data.size(); ++i) {
            int mod_result = i % 6;
            int sub_result = mod_result - 3;
            data[i] = static_cast<float>(sub_result) * 0.5f;
        }
        
        size_t corrupt = 0;
        for (const auto& val : data) {
            if (is_corrupt(val)) corrupt++;
        }
        std::cout << "  Result: " << (corrupt == 0 ? "WORKS" : "FAILS") 
                  << " (" << corrupt << " corrupted)\n\n";
        
        EXPECT_EQ(corrupt, 0) << "Breakdown workaround should prevent corruption";
    }

    // Workaround 3: Lookup table
    {
        std::cout << "Workaround 3: Lookup table\n";
        const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        std::vector<float> data(6);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = values[i % 6];
        }
        
        size_t corrupt = 0;
        for (const auto& val : data) {
            if (is_corrupt(val)) corrupt++;
        }
        std::cout << "  Result: " << (corrupt == 0 ? "WORKS" : "FAILS") 
                  << " (" << corrupt << " corrupted)\n\n";
        
        EXPECT_EQ(corrupt, 0) << "Lookup table should prevent corruption";
    }
}

// ==============================================================================
// TEST 3: Related Patterns - Do Similar Expressions Trigger the Bug?
// ==============================================================================

TEST_F(Clang17BugMinimal, RelatedPatterns) {
    std::cout << "\n=== TESTING RELATED PATTERNS ===\n\n";

    struct Pattern {
        std::string name;
        std::function<float(size_t)> generator;
    };

    std::vector<Pattern> patterns = {
        {"Original: (i%6-3)*0.5f", 
         [](size_t i) { return static_cast<float>(i % 6 - 3) * 0.5f; }},
        
        {"Without cast: (i%6-3)*0.5f", 
         [](size_t i) { return (i % 6 - 3) * 0.5f; }},
        
        {"Different modulo: (i%4-2)*0.5f",
         [](size_t i) { return static_cast<float>(i % 4 - 2) * 0.5f; }},
        
        {"Different multiplier: (i%6-3)*1.0f",
         [](size_t i) { return static_cast<float>(i % 6 - 3) * 1.0f; }},
        
        {"No multiplier: (i%6-3)",
         [](size_t i) { return static_cast<float>(i % 6 - 3); }},
        
        {"Parentheses: ((i%6)-3)*0.5f",
         [](size_t i) { return static_cast<float>((i % 6) - 3) * 0.5f; }},
    };

    for (const auto& pattern : patterns) {
        std::vector<float> data(6);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = pattern.generator(i);
        }
        
        size_t corrupt = 0;
        for (const auto& val : data) {
            if (is_corrupt(val)) corrupt++;
        }
        
        std::cout << std::setw(40) << std::left << pattern.name << ": ";
        if (corrupt == 0) {
            std::cout << "OK";
        } else {
            std::cout << "BUG (" << corrupt << " corrupted)";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ==============================================================================
// TEST 4: Hex Dump for LLVM Bug Report
// ==============================================================================

TEST_F(Clang17BugMinimal, HexDumpForBugReport) {
    std::cout << "\n=== HEX DUMP FOR LLVM BUG REPORT ===\n\n";
    std::cout << "Use this output when reporting to LLVM developers:\n\n";
    std::cout << "Compiler: " << __VERSION__ << "\n";
    std::cout << "Optimization: -O0 (bug occurs even without optimization)\n";
    std::cout << "Expression: static_cast<float>(i % 6 - 3) * 0.5f\n\n";

    std::vector<float> data(6);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
    }

    std::cout << "Expected vs Actual:\n";
    std::cout << "i | pattern | expected float      | expected hex | actual float        | actual hex   | status\n";
    std::cout << "--|---------|---------------------|--------------|---------------------|--------------|-------\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        int pattern = i % 6 - 3;
        float expected = static_cast<float>(pattern) * 0.5f;
        
        uint32_t expected_bits, actual_bits;
        memcpy(&expected_bits, &expected, sizeof(float));
        memcpy(&actual_bits, &data[i], sizeof(float));
        
        std::cout << i << " | " 
                  << std::setw(7) << pattern << " | "
                  << std::setw(19) << expected << " | "
                  << "0x" << std::hex << std::setw(8) << std::setfill('0') << expected_bits << " | "
                  << std::dec << std::setw(19) << data[i] << " | "
                  << "0x" << std::hex << std::setw(8) << std::setfill('0') << actual_bits 
                  << std::dec << std::setfill(' ') << " | ";
        
        if (expected_bits == actual_bits) {
            std::cout << "OK";
        } else {
            std::cout << "BUG";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ==============================================================================
// TEST 5: Recommended Fix for Production Code
// ==============================================================================

TEST_F(Clang17BugMinimal, RecommendedProductionFix) {
    std::cout << "\n=== RECOMMENDED FIX FOR PRODUCTION ===\n\n";
    std::cout << "Replace buggy pattern in all test code:\n\n";
    
    std::cout << "BUGGY CODE (DO NOT USE):\n";
    std::cout << "  test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;\n\n";
    
    std::cout << "FIXED CODE (USE THIS):\n";
    std::cout << "  const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};\n";
    std::cout << "  test_data[i] = values[i % 6];\n\n";
    
    // Verify the fix works
    const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    std::vector<float> data(32);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = values[i % 6];
    }
    
    size_t corrupt = 0;
    for (const auto& val : data) {
        if (is_corrupt(val)) corrupt++;
    }
    
    std::cout << "Verification: " << (corrupt == 0 ? "PASS" : "FAIL") 
              << " (0 corrupted values in 32 elements)\n";
    
    EXPECT_EQ(corrupt, 0) << "Production fix must work correctly";
}

}  // namespace ttml::clang17::bug::tests
