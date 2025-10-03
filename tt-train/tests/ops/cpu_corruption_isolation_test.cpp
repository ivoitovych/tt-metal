// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// CPU-LEVEL CORRUPTION ISOLATION TEST
// ==============================================================================
// This test isolates the CPU-side corruption issue discovered in the 
// comprehensive diagnostic. The corruption occurs BEFORE any device operations,
// during basic C++ vector initialization.
//
// FINDINGS FROM PREVIOUS TEST:
// - Corruption happens in CPU memory during vector creation
// - Pattern: (i % 6 - 3) * 0.5f produces corrupted values
// - Only affects specific vector sizes (32 elements)
// - All positive or all negative patterns work fine
//
// This test eliminates ALL device operations to prove it's a pure CPU issue.
// ==============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>  // For memcpy
#include <iomanip>
#include <iostream>
#include <vector>

namespace ttml::cpu::corruption::tests {

class CPUCorruptionIsolation : public ::testing::Test {
protected:
    static bool is_corrupted(float value) {
        return std::abs(value) > 1e10f || std::isnan(value) || std::isinf(value);
    }

    static void print_vector_hex(const std::vector<float>& vec, const std::string& name, size_t max_elements = 16) {
        std::cout << name << ":\n";
        for (size_t i = 0; i < std::min(max_elements, vec.size()); ++i) {
            // Print both float value and hex representation
            uint32_t bits;
            memcpy(&bits, &vec[i], sizeof(float));  // Changed from std::memcpy
            std::cout << "  [" << std::setw(2) << i << "] = " 
                      << std::setw(12) << vec[i] 
                      << " (0x" << std::hex << std::setw(8) << std::setfill('0') 
                      << bits << std::dec << std::setfill(' ') << ")";
            if (is_corrupted(vec[i])) {
                std::cout << " ❌";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

// ==============================================================================
// TEST 1: Pure C++ - No TTML, No Device, No Autograd
// ==============================================================================

TEST_F(CPUCorruptionIsolation, PureCPlusPlus_NoTTML) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 1: Pure C++ Vector Initialization               ║\n";
    std::cout << "║  NO TTML dependencies, NO device operations            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Method 1: The problematic pattern
    {
        std::cout << "Method 1: Using (i % 6 - 3) * 0.5f pattern\n";
        std::cout << "─────────────────────────────────────────────\n";
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
        }
        print_vector_hex(test_data, "Result", 12);
        
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "Corrupted: " << corrupted << "/32\n\n";
        
        if (corrupted > 0) {
            std::cout << "❌ CORRUPTION CONFIRMED IN PURE C++\n\n";
        }
    }

    // Method 2: Using lookup table (safe alternative)
    {
        std::cout << "Method 2: Using lookup table\n";
        std::cout << "─────────────────────────────────────────────\n";
        const float values[] = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = values[i % 6];
        }
        print_vector_hex(test_data, "Result", 12);
        
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "Corrupted: " << corrupted << "/32\n\n";
        
        EXPECT_EQ(corrupted, 0) << "Lookup table method should never corrupt";
    }

    // Method 3: Explicit calculation breakdown
    {
        std::cout << "Method 3: Breaking down the calculation\n";
        std::cout << "─────────────────────────────────────────────\n";
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            int mod_result = i % 6;
            int sub_result = mod_result - 3;
            float float_result = static_cast<float>(sub_result);
            test_data[i] = float_result * 0.5f;
        }
        print_vector_hex(test_data, "Result", 12);
        
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "Corrupted: " << corrupted << "/32\n\n";
    }

    // Method 4: Using different type for intermediate
    {
        std::cout << "Method 4: Using int32_t explicitly\n";
        std::cout << "─────────────────────────────────────────────\n";
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            int32_t pattern = static_cast<int32_t>(i % 6) - 3;
            test_data[i] = static_cast<float>(pattern) * 0.5f;
        }
        print_vector_hex(test_data, "Result", 12);
        
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "Corrupted: " << corrupted << "/32\n\n";
    }
}

// ==============================================================================
// TEST 2: Size Dependency Analysis
// ==============================================================================

TEST_F(CPUCorruptionIsolation, SizeDependency) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 2: Vector Size Dependency                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::vector<size_t> sizes = {4, 8, 16, 24, 32, 40, 48, 64, 128};

    for (size_t size : sizes) {
        std::vector<float> test_data(size);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
        }
        
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        float percent = 100.0f * corrupted / size;
        
        std::cout << "Size " << std::setw(3) << size << ": ";
        if (corrupted == 0) {
            std::cout << "✓ WORKS\n";
        } else {
            std::cout << "❌ FAILS (" << corrupted << "/" << size 
                      << " = " << std::setprecision(1) << std::fixed << percent << "%)\n";
        }
    }
    std::cout << "\n";
}

// ==============================================================================
// TEST 3: Compiler Optimization Test
// ==============================================================================

// Volatile prevents compiler optimization
volatile int global_dummy = 0;

__attribute__((noinline))
float compute_value_noinline(size_t i) {
    return static_cast<float>(i % 6 - 3) * 0.5f;
}

TEST_F(CPUCorruptionIsolation, CompilerOptimization) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 3: Compiler Optimization Investigation           ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Test 1: Normal inline
    {
        std::cout << "Test 1: Normal inline (optimizable)\n";
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
        }
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "  Corrupted: " << corrupted << "/32 " 
                  << (corrupted > 0 ? "❌" : "✓") << "\n\n";
    }

    // Test 2: Noinline function
    {
        std::cout << "Test 2: Noinline function call\n";
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = compute_value_noinline(i);
        }
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "  Corrupted: " << corrupted << "/32 " 
                  << (corrupted > 0 ? "❌" : "✓") << "\n\n";
    }

    // Test 3: With volatile (prevents optimization)
    {
        std::cout << "Test 3: Using volatile to prevent optimization\n";
        std::vector<float> test_data(32);
        for (size_t i = 0; i < test_data.size(); ++i) {
            volatile int pattern = i % 6 - 3;
            test_data[i] = static_cast<float>(pattern) * 0.5f;
            global_dummy = pattern; // Use volatile
        }
        size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
        std::cout << "  Corrupted: " << corrupted << "/32 " 
                  << (corrupted > 0 ? "❌" : "✓") << "\n\n";
    }
}

// ==============================================================================
// TEST 4: Memory Pattern Analysis
// ==============================================================================

TEST_F(CPUCorruptionIsolation, MemoryPatternAnalysis) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 4: Memory Pattern Analysis                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::vector<float> test_data(32);
    
    // Initialize
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
    }

    // Analyze corruption pattern
    std::cout << "Analyzing which indices get corrupted:\n\n";
    std::cout << "Index | Expected | Actual     | i%6 | (i%6-3) | Corrupted?\n";
    std::cout << "------|----------|------------|-----|---------|------------\n";
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        int mod6 = i % 6;
        int sub3 = mod6 - 3;
        float expected = static_cast<float>(sub3) * 0.5f;
        
        std::cout << std::setw(5) << i << " | "
                  << std::setw(8) << expected << " | "
                  << std::setw(10) << test_data[i] << " | "
                  << std::setw(3) << mod6 << " | "
                  << std::setw(7) << sub3 << " | ";
        
        if (is_corrupted(test_data[i])) {
            std::cout << "YES ❌";
        } else if (std::abs(test_data[i] - expected) > 0.001f) {
            std::cout << "MISMATCH ⚠";
        } else {
            std::cout << "no";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ==============================================================================
// TEST 5: Reproduction Outside Google Test Framework
// ==============================================================================

TEST_F(CPUCorruptionIsolation, OutsideTestFramework) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 5: Check if Google Test Framework is involved   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Create data in a regular function scope (not in test fixture)
    auto create_vector = []() -> std::vector<float> {
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
        }
        return data;
    };

    auto test_data = create_vector();
    
    size_t corrupted = std::count_if(test_data.begin(), test_data.end(), is_corrupted);
    std::cout << "Vector created outside test fixture:\n";
    std::cout << "  Corrupted: " << corrupted << "/32 " 
              << (corrupted > 0 ? "❌" : "✓") << "\n\n";
    
    print_vector_hex(test_data, "First 12 values", 12);
}

// ==============================================================================
// TEST 6: Raw Memory Dump
// ==============================================================================

TEST_F(CPUCorruptionIsolation, RawMemoryDump) {
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 6: Raw Memory Dump at Each Step                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    std::vector<float> test_data(32);
    
    std::cout << "Step-by-step memory state during initialization:\n\n";
    
    for (size_t i = 0; i < std::min(size_t(12), test_data.size()); ++i) {
        // Compute value
        int mod_result = i % 6;
        int sub_result = mod_result - 3;
        float value = static_cast<float>(sub_result) * 0.5f;
        
        // Assign
        test_data[i] = value;
        
        // Read back immediately
        float readback = test_data[i];
        
        // Print
        uint32_t bits_written, bits_read;
        memcpy(&bits_written, &value, sizeof(float));  // Changed from std::memcpy
        memcpy(&bits_read, &readback, sizeof(float));  // Changed from std::memcpy
        
        std::cout << "i=" << std::setw(2) << i 
                  << " | mod=" << mod_result 
                  << " sub=" << std::setw(2) << sub_result
                  << " | wrote=" << std::setw(6) << value 
                  << " (0x" << std::hex << bits_written << ")"
                  << " | read=" << std::dec << std::setw(6) << readback
                  << " (0x" << std::hex << bits_read << ")";
        
        if (bits_written != bits_read) {
            std::cout << " ❌ MISMATCH";
        }
        std::cout << std::dec << "\n";
    }
    std::cout << "\n";
}

}  // namespace ttml::cpu::corruption::tests
