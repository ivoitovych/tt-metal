// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ==============================================================================
// GELU TENSOR CORRUPTION DIAGNOSTIC TEST
// ==============================================================================
// This test file diagnoses a critical tensor corruption issue that occurs
// during tensor creation/upload with specific shape and value patterns.
//
// ISSUE: When creating tensors with shape [1,1,4,8] containing the pattern
//        [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, ...], negative values at certain
//        indices become corrupted to 9.22337e+18 (near 2^63 max signed int).
//
// CORRUPTION PATTERN: Indices 0,1,2, 6,7,8, 12,13,14, 18,19,20, 24,25,26, 30,31
//                     (Every 6 elements, the first 3 are corrupted)
//                     These correspond to ALL negative values in the pattern.
//
// ROOT CAUSE HYPOTHESIS: BFLOAT16 encoding or tile padding issue in
//                        core::from_vector() when uploading specific patterns
//                        of negative values to device memory.
//
// This test provides evidence to show colleagues:
// 1. Where exactly the corruption occurs (during tensor upload vs GELU)
// 2. Which shapes/patterns trigger it
// 3. Which shapes/patterns work correctly
// ==============================================================================

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::ops::tests {

class GeluCorruptionDiagnostic : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }

    // Helper to check if a value is corrupted (near 2^63)
    static bool is_corrupted(float value) {
        return std::abs(value) > 1e10f || std::isnan(value) || std::isinf(value);
    }

    // Helper to print corruption statistics
    static void print_corruption_stats(
        const std::vector<float>& expected,
        const std::vector<float>& actual,
        const std::string& stage) {
        
        size_t corruption_count = 0;
        std::vector<size_t> corrupted_indices;

        for (size_t i = 0; i < actual.size(); ++i) {
            if (is_corrupted(actual[i])) {
                corruption_count++;
                if (corrupted_indices.size() < 20) {  // Limit output
                    corrupted_indices.push_back(i);
                }
            }
        }

        std::cout << "\n=== " << stage << " ===\n";
        std::cout << "Total elements: " << actual.size() << "\n";
        std::cout << "Corrupted elements: " << corruption_count 
                  << " (" << (100.0 * corruption_count / actual.size()) << "%)\n";

        if (corruption_count > 0) {
            std::cout << "Corrupted indices: ";
            for (size_t idx : corrupted_indices) {
                std::cout << idx << " ";
            }
            if (corruption_count > 20) {
                std::cout << "... (+" << (corruption_count - 20) << " more)";
            }
            std::cout << "\n";

            std::cout << "\nFirst 10 corrupted values:\n";
            size_t shown = 0;
            for (size_t i = 0; i < actual.size() && shown < 10; ++i) {
                if (is_corrupted(actual[i])) {
                    std::cout << "  [" << i << "] expected=" << expected[i] 
                              << " actual=" << actual[i] << "\n";
                    shown++;
                }
            }
        } else {
            std::cout << "✓ All values correct!\n";
        }
    }
};

// ==============================================================================
// TEST 1: Pinpoint Corruption Location - The Critical Test
// ==============================================================================
// This test adds checkpoints at every step to determine EXACTLY where
// corruption occurs in the data pipeline.
// ==============================================================================

TEST_F(GeluCorruptionDiagnostic, PinpointCorruptionLocation) {
    auto* device = &autograd::ctx().get_device();

    // Use the exact shape and pattern that causes corruption
    uint32_t batch = 1;
    uint32_t seq_len = 4;
    uint32_t intermediate_size = 8;
    
    std::vector<float> test_data(batch * seq_len * intermediate_size);
    
    // Create the problematic pattern: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, ...]
    for (size_t i = 0; i < test_data.size(); ++i) {
        test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
    }

    // CHECKPOINT 0: Verify CPU data before any operations
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CHECKPOINT 0: Original CPU data (before any ops)     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    std::cout << "Shape: [" << batch << ", 1, " << seq_len << ", " << intermediate_size << "]\n";
    std::cout << "Pattern: [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, ...] repeating\n\n";
    
    std::cout << "First 16 values:\n";
    for (size_t i = 0; i < std::min(size_t(16), test_data.size()); ++i) {
        std::cout << "  [" << i << "] = " << test_data[i] << "\n";
    }

    // CHECKPOINT 1: Upload to device and read back immediately
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CHECKPOINT 1: After upload to device                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    auto shape = ttnn::Shape({batch, 1, seq_len, intermediate_size});
    auto tensor = core::from_vector(test_data, shape, device);
    auto readback = core::to_vector(tensor);
    
    print_corruption_stats(test_data, readback, "After Device Upload");

    // CHECKPOINT 2: Wrap in autograd tensor (no computation yet)
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CHECKPOINT 2: After autograd wrapping                ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    auto tensor_ptr = autograd::create_tensor(tensor);
    auto after_wrap = core::to_vector(tensor_ptr->get_value());
    
    print_corruption_stats(test_data, after_wrap, "After Autograd Wrapping");

    // CHECKPOINT 3: After GELU operation
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  CHECKPOINT 3: After GELU operation                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    auto result = gelu(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());
    
    print_corruption_stats(test_data, result_data, "After GELU");

    // FINAL VERDICT
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  DIAGNOSTIC VERDICT                                    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";
    
    bool corrupted_at_upload = std::any_of(readback.begin(), readback.end(), is_corrupted);
    bool corrupted_after_wrap = std::any_of(after_wrap.begin(), after_wrap.end(), is_corrupted);
    bool corrupted_after_gelu = std::any_of(result_data.begin(), result_data.end(), is_corrupted);

    if (corrupted_at_upload) {
        std::cout << "❌ CORRUPTION OCCURS AT: Device upload (core::from_vector)\n";
        std::cout << "   This is a bug in tensor creation/tilization, NOT in GELU!\n";
    } else if (corrupted_after_wrap) {
        std::cout << "❌ CORRUPTION OCCURS AT: Autograd tensor wrapping\n";
    } else if (corrupted_after_gelu) {
        std::cout << "❌ CORRUPTION OCCURS AT: GELU operation\n";
    } else {
        std::cout << "✓ NO CORRUPTION DETECTED at any checkpoint\n";
    }

    std::cout << "\n";
}

// ==============================================================================
// TEST 2: Shape Dependency Test
// ==============================================================================
// This test shows which shapes trigger corruption and which don't.
// Helps identify the pattern that causes the issue.
// ==============================================================================

TEST_F(GeluCorruptionDiagnostic, ShapeDependencyTest) {
    auto* device = &autograd::ctx().get_device();

    struct ShapeTest {
        std::vector<uint32_t> shape;
        std::string description;
    };

    std::vector<ShapeTest> test_cases = {
        {{1, 1, 2, 4}, "Small shape [1,1,2,4] - KNOWN WORKING"},
        {{1, 1, 4, 8}, "Problem shape [1,1,4,8] - KNOWN FAILING"},
        {{1, 1, 8, 4}, "Swapped dims [1,1,8,4]"},
        {{1, 1, 1, 32}, "Single row [1,1,1,32]"},
        {{1, 1, 32, 1}, "Single col [1,1,32,1]"},
        {{2, 1, 2, 4}, "Batch=2 [2,1,2,4]"},
        {{1, 1, 4, 6}, "Non-power-2 width [1,1,4,6]"},
    };

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  SHAPE DEPENDENCY ANALYSIS                             ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    for (const auto& test_case : test_cases) {
        size_t total_elements = test_case.shape[0] * test_case.shape[2] * test_case.shape[3];
        std::vector<float> test_data(total_elements);
        
        // Same problematic pattern
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
        }

        auto shape = ttnn::Shape(test_case.shape);
        auto tensor = core::from_vector(test_data, shape, device);
        auto readback = core::to_vector(tensor);

        size_t corruption_count = std::count_if(
            readback.begin(), readback.end(), is_corrupted);

        std::cout << test_case.description << "\n";
        std::cout << "  Result: ";
        if (corruption_count == 0) {
            std::cout << "✓ WORKS (no corruption)\n";
        } else {
            std::cout << "❌ FAILS (" << corruption_count << "/" << total_elements 
                      << " corrupted = " << (100.0 * corruption_count / total_elements) 
                      << "%)\n";
        }
        std::cout << "\n";
    }
}

// ==============================================================================
// TEST 3: Value Pattern Dependency Test
// ==============================================================================
// This test checks if corruption depends on the specific values (negative vs
// positive) or the pattern itself.
// ==============================================================================

TEST_F(GeluCorruptionDiagnostic, ValuePatternDependencyTest) {
    auto* device = &autograd::ctx().get_device();
    auto shape = ttnn::Shape({1, 1, 4, 8});
    size_t total_elements = 32;

    struct PatternTest {
        std::string name;
        std::function<float(size_t)> generator;
    };

    std::vector<PatternTest> patterns = {
        {"Original pattern (mixed neg/pos)", 
         [](size_t i) { return static_cast<float>(i % 6 - 3) * 0.5f; }},
        
        {"All positive values",
         [](size_t i) { return static_cast<float>(i % 6) * 0.5f; }},
        
        {"All negative values",
         [](size_t i) { return -static_cast<float>(i % 6 + 1) * 0.5f; }},
        
        {"All zeros",
         [](size_t) { return 0.0f; }},
        
        {"All ones",
         [](size_t) { return 1.0f; }},
        
        {"Sequential positive",
         [](size_t i) { return static_cast<float>(i) * 0.1f; }},
        
        {"Sequential negative",
         [](size_t i) { return -static_cast<float>(i) * 0.1f; }},
    };

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  VALUE PATTERN DEPENDENCY ANALYSIS                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    for (const auto& pattern : patterns) {
        std::vector<float> test_data(total_elements);
        for (size_t i = 0; i < test_data.size(); ++i) {
            test_data[i] = pattern.generator(i);
        }

        auto tensor = core::from_vector(test_data, shape, device);
        auto readback = core::to_vector(tensor);

        size_t corruption_count = std::count_if(
            readback.begin(), readback.end(), is_corrupted);

        std::cout << pattern.name << ":\n";
        std::cout << "  First 8 values: [";
        for (size_t i = 0; i < 8; ++i) {
            std::cout << test_data[i];
            if (i < 7) std::cout << ", ";
        }
        std::cout << "]\n";
        
        std::cout << "  Result: ";
        if (corruption_count == 0) {
            std::cout << "✓ WORKS\n";
        } else {
            std::cout << "❌ FAILS (" << corruption_count << "/" << total_elements 
                      << " corrupted)\n";
        }
        std::cout << "\n";
    }
}

// ==============================================================================
// TEST 4: Comparison with Working Shape
// ==============================================================================
// Side-by-side comparison of working vs failing shape to highlight differences
// ==============================================================================

TEST_F(GeluCorruptionDiagnostic, WorkingVsFailingComparison) {
    auto* device = &autograd::ctx().get_device();

    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║  WORKING vs FAILING SHAPE COMPARISON                   ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";

    // Working shape
    {
        std::cout << "WORKING SHAPE: [1, 1, 2, 4] (8 elements)\n";
        std::cout << "────────────────────────────────────────\n";
        
        std::vector<float> data = {-1.5f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, -1.5f, -1.0f};
        auto shape = ttnn::Shape({1, 1, 2, 4});
        auto tensor = core::from_vector(data, shape, device);
        auto readback = core::to_vector(tensor);

        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << "  [" << i << "] sent=" << data[i] 
                      << " received=" << readback[i];
            if (is_corrupted(readback[i])) {
                std::cout << " ❌ CORRUPTED";
            } else if (std::abs(data[i] - readback[i]) > 0.01f) {
                std::cout << " ⚠ MISMATCH";
            } else {
                std::cout << " ✓";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\n";

    // Failing shape  
    {
        std::cout << "FAILING SHAPE: [1, 1, 4, 8] (32 elements)\n";
        std::cout << "─────────────────────────────────────────\n";
        
        std::vector<float> data(32);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 6 - 3) * 0.5f;
        }
        
        auto shape = ttnn::Shape({1, 1, 4, 8});
        auto tensor = core::from_vector(data, shape, device);
        auto readback = core::to_vector(tensor);

        for (size_t i = 0; i < std::min(size_t(16), data.size()); ++i) {
            std::cout << "  [" << i << "] sent=" << data[i] 
                      << " received=" << readback[i];
            if (is_corrupted(readback[i])) {
                std::cout << " ❌ CORRUPTED";
            } else if (std::abs(data[i] - readback[i]) > 0.01f) {
                std::cout << " ⚠ MISMATCH";
            } else {
                std::cout << " ✓";
            }
            std::cout << "\n";
        }
        std::cout << "  ... (showing first 16 of 32 elements)\n";
    }
}

}  // namespace ttml::ops::tests

