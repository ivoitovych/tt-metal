// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Real Data Test - NO PYTHON BINDINGS
 *
 * This test uses REAL data extracted from prajjwal1/bert-tiny and embedded
 * directly in C++ to bypass all Python bindings and test the C++ implementation
 * directly with real learned BERT weights.
 *
 * This will tell us definitively if the issue is in:
 * - C++ computation (if this test fails)
 * - Python bindings (if this test passes but Python tests fail)
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/scaled_dot_product_attention.hpp"

using namespace ttml;

namespace {

float compute_pcc(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size())
        return 0.0F;

    float mean_x = 0.0F, mean_y = 0.0F;
    for (size_t i = 0; i < x.size(); ++i) {
        mean_x += x[i];
        mean_y += y[i];
    }
    mean_x /= x.size();
    mean_y /= y.size();

    float numerator = 0.0F, denom_x = 0.0F, denom_y = 0.0F;
    for (size_t i = 0; i < x.size(); ++i) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    float denominator = std::sqrt(denom_x * denom_y);
    return (denominator > 0) ? (numerator / denominator) : 0.0F;
}

void print_comparison(
    const char* name, const std::vector<float>& expected, const std::vector<float>& actual, float threshold) {
    float pcc = compute_pcc(expected, actual);

    float mean_diff = 0.0F, max_diff = 0.0F;
    for (size_t i = 0; i < expected.size(); ++i) {
        float diff = std::abs(expected[i] - actual[i]);
        mean_diff += diff;
        max_diff = std::max(max_diff, diff);
    }
    mean_diff /= expected.size();

    std::cout << "\n================================================================================\n";
    std::cout << name << ": PCC = " << pcc;
    if (pcc >= threshold) {
        std::cout << " ✅ PASS\n";
    } else {
        std::cout << " ❌ FAIL\n";
    }
    std::cout << "================================================================================\n";
    std::cout << "Size: " << expected.size() << "\n";
    std::cout << "Mean abs diff: " << mean_diff << ", Max abs diff: " << max_diff << "\n";
    std::cout << "First 5 expected: ";
    for (size_t i = 0; i < std::min(size_t(5), expected.size()); ++i) {
        std::cout << expected[i] << " ";
    }
    std::cout << "\nFirst 5 actual: ";
    for (size_t i = 0; i < std::min(size_t(5), actual.size()); ++i) {
        std::cout << actual[i] << " ";
    }
    std::cout << "\n";
}

}  // namespace

// Insert the extracted data here - this will be included from the generated file
TEST(BERTRealDataTest, AttentionWithRealBERTData) {
    std::cout << "\n################################################################################\n";
    std::cout << "TEST: Attention with REAL BERT Data (No Python Bindings)\n";
    std::cout << "################################################################################\n";
    std::cout << "Data source: prajjwal1/bert-tiny\n";
    std::cout << "Text: 'The quick brown fox'\n\n";
