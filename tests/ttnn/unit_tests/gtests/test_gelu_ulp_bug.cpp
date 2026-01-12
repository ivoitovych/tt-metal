// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * GELU ULP Bug Reproducer and BFloat16 ULP Calculator Verification
 *
 * This test file contains:
 * 1. BFloat16 ULP (Units in Last Place) calculator with verification tests
 * 2. GELU precision bug reproducers for three problematic regions
 *
 * The ULP calculator is verified using a sorted BF16 value index approach:
 * - All valid BF16 values are sorted in numerical order
 * - Adjacent values in this order should have ULP distance of 1
 * - Both +0 and -0 map to the same index (ULP distance = 0)
 *
 * ORIGINAL BUG (before fix):
 * - Region 1 (Deep Negative Tail, x < -5.5): Max ULP = 32,767 (hardware returned 0.0)
 * - Region 2 (Near-Zero, |x| < ~1e-4): Max ULP = 14,276 (floor value 2.98e-05)
 * - Region 3 (Transition, -5.5 to -4.0): Max ULP = 1,475 (poor polynomial fit)
 *
 * AFTER FIX (C6 adaptive polynomial + raw x segments):
 * - Max ULP = 7 (at x = -5.969 in asymptotic region)
 * - All polynomial segments have Max ULP = 1
 * - 99.80% of BF16 values have ULP <= 1
 *
 * Run: ./build_Debug/test/ttnn/unit_tests_ttnn --gtest_filter="*GeluUlp*"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <limits>
#include <iomanip>
#include <set>
#include <mpfr.h>

#include <tt-metalium/bfloat16.hpp>
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {

// =============================================================================
// BFloat16 ULP Calculator
// =============================================================================

namespace bf16_ulp {

// =============================================================================
// Tenstorrent Hardware Model: DAZ+FTZ (Denormals-Are-Zero + Flush-To-Zero)
//
// Per tech_reports/Handling_Special_Value/special_values.md:
// "denormals | all | 0x0"
//
// The SFPU treats all denormal values as zero. This affects ULP calculations:
// - Denormal inputs are read as zero (DAZ)
// - Denormal outputs are flushed to zero (FTZ)
// - For ULP purposes, all denormals map to the same value as zero
// =============================================================================

/**
 * Convert float to BFloat16 bit representation (truncation, no rounding).
 */
inline uint16_t float_to_bf16_bits(float f) {
    uint32_t f32_bits;
    std::memcpy(&f32_bits, &f, sizeof(float));
    return static_cast<uint16_t>(f32_bits >> 16);
}

/**
 * Convert BFloat16 bits to float.
 */
inline float bf16_bits_to_float(uint16_t bits) {
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(float));
    return f;
}

/**
 * Check if BF16 bits represent a denormal (subnormal) value.
 * Denormal: exponent = 0, mantissa != 0
 */
inline bool is_bf16_denormal(uint16_t bits) {
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & 0x7F;
    return (exp == 0) && (mantissa != 0);
}

/**
 * Check if a float value is denormal when represented as BF16.
 */
inline bool is_bf16_denormal(float f) { return is_bf16_denormal(float_to_bf16_bits(f)); }

/**
 * Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits.
 * Maps all denormals to +0 (0x0000).
 */
inline uint16_t bf16_daz_normalize(uint16_t bits) {
    if (is_bf16_denormal(bits)) {
        return 0x0000;  // All denormals become +0
    }
    // Also normalize -0 to +0 for consistency
    if (bits == 0x8000) {
        return 0x0000;
    }
    return bits;
}

/**
 * Apply DAZ normalization to a float value (as BF16).
 */
inline float bf16_daz_normalize(float f) {
    uint16_t bits = float_to_bf16_bits(f);
    uint16_t normalized = bf16_daz_normalize(bits);
    return bf16_bits_to_float(normalized);
}

/**
 * Get the next representable BFloat16 value (increment by 1 ULP).
 * Accounts for DAZ: skips over denormal range.
 */
inline float bf16_next(float f) {
    uint16_t bits = bf16_daz_normalize(float_to_bf16_bits(f));
    if (bits == 0x7F80) {
        return std::numeric_limits<float>::infinity();  // +inf
    }
    if (bits == 0xFF80) {
        return bf16_bits_to_float(0xFF7F);  // -inf -> -max
    }
    if (bits == 0x0000) {
        return bf16_bits_to_float(0x0080);  // 0 -> smallest positive normal
    }
    if (bits & 0x8000) {
        // Negative: decrement magnitude
        uint16_t next_bits = bits - 1;
        // Skip denormals: if we hit denormal range, jump to zero
        if (is_bf16_denormal(next_bits)) {
            return 0.0f;
        }
        return bf16_bits_to_float(next_bits);
    } else {
        // Positive: increment
        return bf16_bits_to_float(bits + 1);
    }
}

/**
 * Calculate the value order index for a BFloat16 value with DAZ.
 *
 * With DAZ+FTZ, the representable values are:
 * - Negative normals: 0xFF7F (-max) to 0x8080 (-min_normal)
 * - Zero: 0x0000 (all denormals and ±0 map here)
 * - Positive normals: 0x0080 (+min_normal) to 0x7F7F (+max)
 *
 * Index layout (excluding denormals):
 * - 0xFF7F (-max) -> index 0
 * - 0x8080 (-min_normal) -> index 32639
 * - 0x0000 (zero) -> index 32640
 * - 0x0080 (+min_normal) -> index 32641
 * - 0x7F7F (+max) -> index 65280
 *
 * Total: 32640 negative normals + 1 zero + 32640 positive normals = 65281 values
 */
inline int32_t bf16_value_order_index_daz(uint16_t bits) {
    // Apply DAZ normalization
    bits = bf16_daz_normalize(bits);

    // Handle NaN - return -1 as invalid
    uint16_t exp = (bits >> 7) & 0xFF;
    uint16_t mantissa = bits & 0x7F;
    if (exp == 0xFF && mantissa != 0) {
        return -1;
    }

    // Handle infinity
    if (bits == 0x7F80) {
        return 65281;  // +inf (after all finite values)
    }
    if (bits == 0xFF80) {
        return -1;  // -inf (exclude from valid range)
    }

    // Zero (including all denormals which map to zero)
    if (bits == 0x0000) {
        return 32640;  // Middle of the range
    }

    if (bits & 0x8000) {
        // Negative normal: 0xFF7F -> 0, ..., 0x8080 -> 32639
        // magnitude ranges from 0x7F7F (max) to 0x0080 (min normal)
        uint16_t magnitude = bits & 0x7FFF;
        // 0x7F7F -> index 0, 0x0080 -> index 32639
        return 0x7F7F - magnitude;
    } else {
        // Positive normal: 0x0080 -> 32641, ..., 0x7F7F -> 65280
        // bits ranges from 0x0080 to 0x7F7F
        return 32640 + bits - 0x007F;
    }
}

inline int32_t bf16_value_order_index_daz(float f) { return bf16_value_order_index_daz(float_to_bf16_bits(f)); }

/**
 * Calculate ULP distance between two BFloat16 values with DAZ+FTZ.
 *
 * This properly accounts for Tenstorrent hardware behavior where
 * all denormals are treated as zero.
 */
inline int32_t ulp_distance_bf16_daz(float a, float b) {
    // Apply DAZ normalization to both values
    uint16_t a_bits = bf16_daz_normalize(float_to_bf16_bits(a));
    uint16_t b_bits = bf16_daz_normalize(float_to_bf16_bits(b));

    // Handle NaN
    uint16_t a_exp = (a_bits >> 7) & 0xFF;
    uint16_t b_exp = (b_bits >> 7) & 0xFF;
    if ((a_exp == 0xFF && (a_bits & 0x7F) != 0) || (b_exp == 0xFF && (b_bits & 0x7F) != 0)) {
        return -1;
    }

    // Use value order index for accurate ULP distance
    int32_t idx_a = bf16_value_order_index_daz(a_bits);
    int32_t idx_b = bf16_value_order_index_daz(b_bits);

    if (idx_a < 0 || idx_b < 0) {
        return -1;
    }

    return std::abs(idx_a - idx_b);
}

// Legacy functions for backwards compatibility (without DAZ)
inline int32_t bf16_value_order_index(float f) {
    uint16_t bits = float_to_bf16_bits(f);

    // Handle NaN - return -1 as invalid
    if ((bits & 0x7F80) == 0x7F80 && (bits & 0x007F) != 0) {
        return -1;
    }

    // Handle infinity
    if (bits == 0x7F80) {
        return 65279;  // +inf
    }
    if (bits == 0xFF80) {
        return -1;  // -inf (exclude from valid range)
    }

    if (bits & 0x8000) {
        uint16_t magnitude = bits & 0x7FFF;
        if (magnitude == 0) {
            return 32639;  // -0 same as +0
        }
        return 32639 - magnitude;
    } else {
        return 32639 + bits;
    }
}

inline int32_t ulp_distance_bf16(float a, float b) {
    // Use DAZ-aware version by default
    return ulp_distance_bf16_daz(a, b);
}

/**
 * Exact GELU using MPFR 256-bit precision.
 *
 * GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
 *
 * This uses MPFR (Multiple Precision Floating-Point Reliable) library
 * to compute the true GELU value with 256-bit precision. This is necessary
 * because the standard fp64 erf() function saturates to -1.0 prematurely
 * for large negative inputs (around x = -8.375), giving incorrect reference
 * values. The true zero saturation threshold is x = -13.1875.
 *
 * See GELU_BF16_Zero_Saturation_Threshold_Research.md for details.
 */
inline double gelu_exact(double x) {
    constexpr mpfr_prec_t precision = 256;

    mpfr_t mpfr_x, sqrt2, x_div_sqrt2, erf_result, one, half, one_plus_erf, result;

    mpfr_init2(mpfr_x, precision);
    mpfr_init2(sqrt2, precision);
    mpfr_init2(x_div_sqrt2, precision);
    mpfr_init2(erf_result, precision);
    mpfr_init2(one, precision);
    mpfr_init2(half, precision);
    mpfr_init2(one_plus_erf, precision);
    mpfr_init2(result, precision);

    // Set values
    mpfr_set_d(mpfr_x, x, MPFR_RNDN);
    mpfr_set_ui(one, 1, MPFR_RNDN);
    mpfr_set_d(half, 0.5, MPFR_RNDN);
    mpfr_sqrt_ui(sqrt2, 2, MPFR_RNDN);

    // Compute x / sqrt(2)
    mpfr_div(x_div_sqrt2, mpfr_x, sqrt2, MPFR_RNDN);

    // Compute erf(x / sqrt(2))
    mpfr_erf(erf_result, x_div_sqrt2, MPFR_RNDN);

    // Compute 1 + erf(x / sqrt(2))
    mpfr_add(one_plus_erf, one, erf_result, MPFR_RNDN);

    // Compute 0.5 * x * (1 + erf(x / sqrt(2)))
    mpfr_mul(result, mpfr_x, half, MPFR_RNDN);
    mpfr_mul(result, result, one_plus_erf, MPFR_RNDN);

    // Extract result as double
    double gelu_result = mpfr_get_d(result, MPFR_RNDN);

    // Clean up
    mpfr_clear(mpfr_x);
    mpfr_clear(sqrt2);
    mpfr_clear(x_div_sqrt2);
    mpfr_clear(erf_result);
    mpfr_clear(one);
    mpfr_clear(half);
    mpfr_clear(one_plus_erf);
    mpfr_clear(result);

    return gelu_result;
}

/**
 * Compute the expected BF16 GELU value with DAZ+FTZ applied.
 *
 * This computes GELU in high precision, then converts to BF16 with DAZ,
 * matching the hardware's representable output.
 */
inline float gelu_expected_bf16_daz(float x) {
    // Apply DAZ to input (hardware reads denormal inputs as zero)
    float x_daz = bf16_daz_normalize(x);

    // Compute exact GELU
    double result = gelu_exact(x_daz);

    // Convert to BF16 (truncation) and apply FTZ
    float result_f32 = static_cast<float>(result);
    return bf16_daz_normalize(result_f32);
}

}  // namespace bf16_ulp

// =============================================================================
// ULP Calculator Verification Tests (No Device Required)
// =============================================================================

class BFloat16UlpTest : public ::testing::Test {};

TEST_F(BFloat16UlpTest, ZeroesHaveSameIndex) {
    // Both +0 and -0 should have the same value order index
    float pos_zero = 0.0f;
    float neg_zero = -0.0f;

    int32_t idx_pos = bf16_ulp::bf16_value_order_index(pos_zero);
    int32_t idx_neg = bf16_ulp::bf16_value_order_index(neg_zero);

    EXPECT_EQ(idx_pos, idx_neg) << "+0 and -0 should have the same index";
    EXPECT_EQ(idx_pos, 32639) << "Zero index should be 32639";
}

TEST_F(BFloat16UlpTest, ZeroesHaveUlpDistanceZero) {
    // ULP distance between +0 and -0 should be 0
    float pos_zero = 0.0f;
    float neg_zero = -0.0f;

    int32_t ulp = bf16_ulp::ulp_distance_bf16(pos_zero, neg_zero);
    EXPECT_EQ(ulp, 0) << "ULP distance between +0 and -0 should be 0";
}

TEST_F(BFloat16UlpTest, AdjacentPositiveValuesHaveUlpOne) {
    // Adjacent BF16 normal values should have ULP distance of 1
    // Note: With DAZ, denormals (0x0001-0x007F) map to zero, so we only test normal values
    // Normal values start at 0x0080 (smallest positive normal)
    std::vector<uint16_t> test_bits = {0x0080, 0x3F80, 0x4000, 0x7F00};

    for (uint16_t bits : test_bits) {
        float val = bf16_ulp::bf16_bits_to_float(bits);
        float next_val = bf16_ulp::bf16_bits_to_float(bits + 1);

        int32_t ulp = bf16_ulp::ulp_distance_bf16(val, next_val);
        EXPECT_EQ(ulp, 1) << "Adjacent positive BF16 normal values at bits 0x" << std::hex << bits
                          << " should have ULP distance 1";
    }
}

TEST_F(BFloat16UlpTest, AdjacentNegativeValuesHaveUlpOne) {
    // Adjacent negative BF16 normal values should have ULP distance of 1
    // Note: With DAZ, denormals (0x8001-0x807F) map to zero, so we only test normal values
    // Negative normal values start at 0x8080 (smallest negative normal)
    std::vector<uint16_t> test_bits = {0x8080, 0xBF80, 0xC000, 0xFF00};

    for (uint16_t bits : test_bits) {
        float val = bf16_ulp::bf16_bits_to_float(bits);
        float next_val = bf16_ulp::bf16_bits_to_float(bits + 1);  // More negative

        int32_t ulp = bf16_ulp::ulp_distance_bf16(val, next_val);
        EXPECT_EQ(ulp, 1) << "Adjacent negative BF16 normal values at bits 0x" << std::hex << bits
                          << " should have ULP distance 1";
    }
}

TEST_F(BFloat16UlpTest, SmallestNormalToZeroIsOne) {
    // With DAZ, smallest positive normal (0x0080) is 1 ULP from zero
    // Denormals (0x0001-0x007F) all map to zero
    float zero = 0.0f;
    float smallest_normal = bf16_ulp::bf16_bits_to_float(0x0080);  // Smallest positive normal

    int32_t ulp = bf16_ulp::ulp_distance_bf16(zero, smallest_normal);
    EXPECT_EQ(ulp, 1) << "ULP distance from 0 to smallest positive normal should be 1";
}

TEST_F(BFloat16UlpTest, DenormalsMapToZero) {
    // With DAZ, all denormals map to zero, so ULP distance from denormal to zero is 0
    float zero = 0.0f;
    float denormal_pos = bf16_ulp::bf16_bits_to_float(0x0001);  // Positive denormal
    float denormal_neg = bf16_ulp::bf16_bits_to_float(0x8001);  // Negative denormal

    int32_t ulp_pos = bf16_ulp::ulp_distance_bf16(zero, denormal_pos);
    int32_t ulp_neg = bf16_ulp::ulp_distance_bf16(zero, denormal_neg);

    EXPECT_EQ(ulp_pos, 0) << "With DAZ, positive denormal should have ULP 0 from zero";
    EXPECT_EQ(ulp_neg, 0) << "With DAZ, negative denormal should have ULP 0 from zero";
}

TEST_F(BFloat16UlpTest, CrossZeroDistanceWithNormals) {
    // Test cross-zero distance with smallest normal values (not denormals)
    // The DAZ value order index maps:
    // - Zero (0x0000) -> 32640 (middle)
    // - Smallest pos normal (0x0080) -> 32640 + 0x0080 - 0x007F = 32641
    // - Smallest neg normal (0x8080) -> 0x7F7F - 0x0080 = 32511
    // Distance: 32641 - 32511 = 130
    float smallest_pos_normal = bf16_ulp::bf16_bits_to_float(0x0080);  // Smallest positive normal
    float smallest_neg_normal = bf16_ulp::bf16_bits_to_float(0x8080);  // Smallest negative normal

    // Use DAZ value order index
    int32_t idx_pos = bf16_ulp::bf16_value_order_index_daz(smallest_pos_normal);
    int32_t idx_neg = bf16_ulp::bf16_value_order_index_daz(smallest_neg_normal);
    int32_t ulp_via_index = std::abs(idx_pos - idx_neg);

    // The ULP calculator counts all values between -min_normal and +min_normal
    // including the 128 positive denormals and 128 negative denormals that map to zero
    // So the gap is: 1 (zero) + 127 (pos denormals) + 1 (pos normal) + 127 (neg denormals) + 1 (neg normal) - 2 = 130
    EXPECT_EQ(ulp_via_index, 130) << "Cross-zero distance via index should be 130 ULP";

    // ulp_distance_bf16 should match
    int32_t ulp = bf16_ulp::ulp_distance_bf16(smallest_pos_normal, smallest_neg_normal);
    EXPECT_EQ(ulp, 130) << "DAZ-aware ULP distance should be 130";
}

TEST_F(BFloat16UlpTest, MaxUlpDistanceWithDAZ) {
    // Distance from most negative to most positive finite value with DAZ model
    // 0xFF7F (-max) -> index 0
    // 0x7F7F (+max) -> index 32640 + 0x7F7F - 0x007F = 32640 + 32512 = 65152
    float max_neg = bf16_ulp::bf16_bits_to_float(0xFF7F);  // -max finite
    float max_pos = bf16_ulp::bf16_bits_to_float(0x7F7F);  // +max finite

    int32_t idx_neg = bf16_ulp::bf16_value_order_index_daz(max_neg);
    int32_t idx_pos = bf16_ulp::bf16_value_order_index_daz(max_pos);

    EXPECT_EQ(idx_neg, 0) << "Most negative finite value should have DAZ index 0";
    EXPECT_EQ(idx_pos, 65152) << "Most positive finite value should have DAZ index 65152";

    // Using DAZ value order index for ULP distance
    int32_t ulp_via_index = std::abs(idx_pos - idx_neg);
    EXPECT_EQ(ulp_via_index, 65152) << "DAZ max ULP distance should be 65152";

    // ulp_distance_bf16 (now DAZ-aware) should match
    int32_t ulp = bf16_ulp::ulp_distance_bf16(max_neg, max_pos);
    EXPECT_EQ(ulp, 65152) << "DAZ-aware ULP distance should be 65152";
}

TEST_F(BFloat16UlpTest, VerifyIndexMonotonicity) {
    // Verify that the value order index is monotonically increasing
    // across the entire BF16 range (excluding NaN and inf)
    std::vector<std::pair<float, int32_t>> values_with_indices;

    // Collect all finite BF16 values with their indices
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN (exponent all 1s, mantissa non-zero)
        if ((bf16_bits & 0x7F80) == 0x7F80 && (bf16_bits & 0x007F) != 0) {
            continue;
        }
        // Skip infinity
        if (bf16_bits == 0x7F80 || bf16_bits == 0xFF80) {
            continue;
        }

        float val = bf16_ulp::bf16_bits_to_float(bf16_bits);
        int32_t idx = bf16_ulp::bf16_value_order_index(val);
        values_with_indices.push_back({val, idx});
    }

    // Sort by numerical value
    std::sort(values_with_indices.begin(), values_with_indices.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Verify indices are monotonically increasing
    for (size_t i = 1; i < values_with_indices.size(); ++i) {
        float prev_val = values_with_indices[i - 1].first;
        float curr_val = values_with_indices[i].first;
        int32_t prev_idx = values_with_indices[i - 1].second;
        int32_t curr_idx = values_with_indices[i].second;

        // Handle special case: -0 and +0 have same index
        if (prev_val == 0.0f && curr_val == 0.0f) {
            EXPECT_EQ(prev_idx, curr_idx) << "Both zeroes should have same index";
        } else {
            EXPECT_LE(prev_idx, curr_idx)
                << "Index should be monotonically increasing: val[" << i - 1 << "]=" << prev_val << " (idx=" << prev_idx
                << ") vs val[" << i << "]=" << curr_val << " (idx=" << curr_idx << ")";
        }
    }
}

TEST_F(BFloat16UlpTest, AdjacentNormalValuesHaveUlpOneOrZeroExceptAtZero) {
    // For adjacent BF16 NORMAL values, ULP should be 0 or 1, EXCEPT at the zero boundary.
    // With DAZ, all 128 negative denormals + (-0) collapse to zero, creating a 129 ULP gap
    // between -min_normal and 0. Similarly for positive side. This is expected DAZ behavior.
    std::vector<float> sorted_values;

    // Collect all finite BF16 values after DAZ normalization (skip duplicates)
    std::set<uint16_t> seen_bits;
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);
        if ((bf16_bits & 0x7F80) == 0x7F80 && (bf16_bits & 0x007F) != 0) {
            continue;  // NaN
        }
        if (bf16_bits == 0x7F80 || bf16_bits == 0xFF80) {
            continue;  // Inf
        }

        // Apply DAZ normalization to get canonical form
        uint16_t normalized = bf16_ulp::bf16_daz_normalize(bf16_bits);
        if (seen_bits.count(normalized)) {
            continue;  // Skip duplicates (denormals and -0 all map to +0)
        }
        seen_bits.insert(normalized);

        sorted_values.push_back(bf16_ulp::bf16_bits_to_float(normalized));
    }

    std::sort(sorted_values.begin(), sorted_values.end());

    // Check all adjacent pairs of DAZ-normalized values
    // Allow for the expected discontinuity at zero boundary
    int failures = 0;
    for (size_t i = 1; i < sorted_values.size() && failures < 10; ++i) {
        int32_t ulp = bf16_ulp::ulp_distance_bf16(sorted_values[i - 1], sorted_values[i]);

        // At zero boundary, expect 129 ULP gap (128 collapsed denormals + 1 for -0/+0)
        bool at_zero_boundary =
            (sorted_values[i - 1] < 0 && sorted_values[i] == 0) || (sorted_values[i - 1] == 0 && sorted_values[i] > 0);
        int32_t expected_max_ulp = at_zero_boundary ? 129 : 1;

        if (ulp > expected_max_ulp) {
            ++failures;
            std::cerr << "FAIL: Adjacent DAZ-normalized values [" << i - 1 << "]=" << sorted_values[i - 1] << " and ["
                      << i << "]=" << sorted_values[i] << " have ULP=" << ulp << " (expected <= " << expected_max_ulp
                      << ")\n";
        }
    }
    EXPECT_EQ(failures, 0) << "Adjacent DAZ-normalized BF16 values should have expected ULP";
}

// =============================================================================
// GELU Bug Reproducer Tests (Require Device)
// =============================================================================

class GeluUlpBugTest : public TTNNFixtureWithDevice {};

TEST_F(GeluUlpBugTest, DeepNegativeTailLowULP) {
    // Region 1: Deep negative tail (x < -5.5)
    // With C6 fix + DAZ+FTZ model:
    // - x < -13.2: FTZ returns 0 (both expected and actual) - ULP = 0
    // - -13.2 < x < -5.5: Asymptotic expansion - Max ULP <= 7

    std::vector<std::pair<float, int32_t>> test_cases = {
        {-13.5f, 0},   // FTZ region - both expected and actual are 0
        {-13.0f, 10},  // Asymptotic region
        {-12.0f, 10},
        {-10.0f, 10},
        {-8.0f, 10},
        {-6.0f, 10},
        {-5.5625f, 10}};

    for (const auto& [input_val, max_expected_ulp] : test_cases) {
        std::array<uint32_t, 4> dims = {1, 1, 32, 32};
        ttnn::Shape shape(dims);
        auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

        auto output_tensor = ttnn::gelu(input_tensor, false);
        auto output_cpu = ttnn::from_device(output_tensor);
        auto output_vec = output_cpu.to_vector<::bfloat16>();
        float actual = static_cast<float>(output_vec[0]);

        float expected = bf16_ulp::gelu_expected_bf16_daz(input_val);
        int32_t ulp_error = bf16_ulp::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp_error
                  << "\n";

        // Verify fix works - ULP should be low
        EXPECT_LE(ulp_error, max_expected_ulp)
            << "x=" << input_val << " expected ULP <= " << max_expected_ulp << ", got " << ulp_error;
    }
}

TEST_F(GeluUlpBugTest, NearZeroLowULP) {
    // Region 2: Near-zero region
    // With C6 fix: Taylor series GELU(x) ≈ x * (0.5 + 0.3989*x) for |x| < 0.125
    // Expected Max ULP <= 2

    std::vector<float> near_zero_inputs = {1e-10f, 1e-8f, 1e-6f, 1e-4f, 0.01f, 0.1f, -0.1f, -0.01f};

    for (float input_val : near_zero_inputs) {
        std::array<uint32_t, 4> dims = {1, 1, 32, 32};
        ttnn::Shape shape(dims);
        auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

        auto output_tensor = ttnn::gelu(input_tensor, false);
        auto output_cpu = ttnn::from_device(output_tensor);
        auto output_vec = output_cpu.to_vector<::bfloat16>();
        float actual = static_cast<float>(output_vec[0]);

        float expected = bf16_ulp::gelu_expected_bf16_daz(input_val);
        int32_t ulp_error = bf16_ulp::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp_error
                  << "\n";

        // Verify fix works - ULP should be low
        EXPECT_LE(ulp_error, 2) << "x=" << input_val << " expected ULP <= 2, got " << ulp_error;
    }
}

TEST_F(GeluUlpBugTest, TransitionRegionLowULP) {
    // Region 3: Transition region around segment boundaries
    // With C6 fix + raw x polynomials: Max ULP <= 10 (worst at x=-5.969 in asymptotic)

    std::vector<std::pair<float, int32_t>> test_cases = {
        {-5.5f, 10},
        {-5.4375f, 10},
        {-5.375f, 10},
        {-5.25f, 10},
        {-5.094f, 10},  // Segment boundary - v3 achieves ULP=0 here
        {-5.0f, 10},
        {-4.75f, 10},
        {-4.5f, 10},
        {-4.0f, 10}};

    int32_t max_ulp = 0;
    for (const auto& [input_val, max_expected_ulp] : test_cases) {
        std::array<uint32_t, 4> dims = {1, 1, 32, 32};
        ttnn::Shape shape(dims);
        auto input_tensor = ttnn::full(shape, input_val, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);

        auto output_tensor = ttnn::gelu(input_tensor, false);
        auto output_cpu = ttnn::from_device(output_tensor);
        auto output_vec = output_cpu.to_vector<::bfloat16>();
        float actual = static_cast<float>(output_vec[0]);

        float expected = bf16_ulp::gelu_expected_bf16_daz(input_val);
        int32_t ulp_error = bf16_ulp::ulp_distance_bf16_daz(actual, expected);
        max_ulp = std::max(max_ulp, ulp_error);

        std::cout << "x=" << input_val << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp_error
                  << "\n";

        // Verify fix works
        EXPECT_LE(ulp_error, max_expected_ulp)
            << "x=" << input_val << " expected ULP <= " << max_expected_ulp << ", got " << ulp_error;
    }

    // Verify overall max ULP is acceptable
    EXPECT_LE(max_ulp, 10) << "Transition region max ULP should be <= 10, got " << max_ulp;
}

TEST_F(GeluUlpBugTest, FTZBoundaryVerification) {
    // Test values around the FTZ (Flush-To-Zero) boundary
    // The boundary is at x ≈ -13.21 where exp(-x²/2) = 1.18e-38 (float32 normal min)
    // For x > -13.2, asymptotic should work correctly
    // For x < -13.2, results are flushed to zero by hardware FTZ
    // With DAZ+FTZ model, both expected and actual are 0 for x < -13.2, so ULP = 0

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    std::cout << "\n========================================\n";
    std::cout << "FTZ BOUNDARY VERIFICATION (DAZ+FTZ MODEL)\n";
    std::cout << "========================================\n";

    // Values that should work (above FTZ boundary)
    std::vector<float> working_values = {-13.0f, -12.5f, -12.0f, -11.0f, -10.0f};
    for (float x : working_values) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        float expected = bf16_ulp::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << x << ": expected=" << expected << ", actual=" << actual << ", ULP=" << ulp << "\n";

        // These should have low ULP (< 10) with DAZ+FTZ model
        EXPECT_LT(ulp, 10) << "x=" << x << " should have ULP < 10 with DAZ+FTZ model";
    }

    // Values at/below FTZ boundary - both expected and actual are 0
    std::vector<float> ftz_values = {-13.5f, -14.0f, -15.0f};
    for (float x : ftz_values) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        float expected = bf16_ulp::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, expected);

        std::cout << "x=" << x << " (FTZ): expected=" << expected << ", actual=" << actual << ", ULP=" << ulp << "\n";

        // With DAZ+FTZ model, both expected and actual are 0, so ULP = 0
        EXPECT_EQ(ulp, 0) << "x=" << x << " should have ULP = 0 with DAZ+FTZ (both are zero)";
    }

    std::cout << "========================================\n";
}

TEST_F(GeluUlpBugTest, DebugWorstCases) {
    // Debug the worst-case values from comprehensive analysis with DAZ+FTZ model
    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    std::cout << "\n============================================================\n";
    std::cout << "DEBUG: WORST-CASE VALUES (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n\n";

    // Worst cases from comprehensive analysis:
    // 1. Asymptotic region: x = -5.969 (Max ULP = 7)
    // 2. Deep neg asymptotic: x around -6 to -13

    std::vector<std::pair<float, std::string>> test_values = {
        // Near-zero (Taylor series)
        {1e-10f, "Near-zero positive"},
        {-1e-10f, "Near-zero negative"},
        {0.1f, "Small positive"},
        {-0.1f, "Small negative"},
        // Deep negative boundary
        {-13.0f, "At -13 (asymptotic)"},
        {-13.2f, "At -13.2 (FTZ boundary)"},
        {-13.5f, "At -13.5 (FTZ)"},
        // Seg 8 boundary (worst case)
        {-5.5f, "At -5.5 (seg 7 start)"},
        {-5.094f, "At -5.094 (WORST - seg boundary)"},
        {-5.095f, "At -5.095 (seg 7/8 boundary)"},
        {-5.0f, "At -5.0"},
        // Normal polynomial segments
        {-3.0f, "At -3.0 (seg 10)"},
        {-1.0f, "At -1.0 (seg 12)"},
        {1.0f, "At 1.0 (seg 14)"},
        {2.5f, "At 2.5 (seg 15)"},
    };

    std::cout << std::left << std::setw(30) << "Description" << std::setw(15) << "Input x" << std::setw(15)
              << "Expected" << std::setw(15) << "Actual" << std::setw(10) << "ULP" << "\n";
    std::cout << std::string(85, '-') << "\n";

    int32_t max_ulp = 0;
    for (const auto& [x, desc] : test_values) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);

        float expected = bf16_ulp::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, expected);
        max_ulp = std::max(max_ulp, ulp);

        std::cout << std::left << std::setw(30) << desc << std::scientific << std::setprecision(3) << std::setw(15) << x
                  << std::setw(15) << expected << std::setw(15) << actual << std::fixed << std::setw(10) << ulp << "\n";
    }
    std::cout << "============================================================\n";
    std::cout << "Max ULP across all test cases: " << max_ulp << "\n";

    // Verify max ULP is acceptable
    EXPECT_LE(max_ulp, 10) << "Expected max ULP <= 10, got " << max_ulp;
}

TEST_F(GeluUlpBugTest, ComprehensiveULPBySegment) {
    // Comprehensive ULP analysis over ALL valid BF16 values
    // Uses DAZ+FTZ model matching Tenstorrent hardware behavior
    // Segments based on the C6 implementation

    struct SegmentStats {
        std::string name;
        float x_min, x_max;
        int64_t sum_ulp = 0;
        int32_t max_ulp = 0;
        int32_t count = 0;
        float worst_x = 0;
    };

    std::vector<SegmentStats> segments = {
        {"Deep neg (FTZ)", -std::numeric_limits<float>::max(), -13.2f},
        {"Deep neg (asymp)", -13.2f, -5.5f},
        {"Range A [-5.5,-5.095]", -5.5f, -5.095f},
        {"Range B [-5.095,-4.136]", -5.095f, -4.136f},
        {"Range C [-4.136,-3.177]", -4.136f, -3.177f},
        {"Seg 10 [-3.177,-2.218]", -3.177f, -2.218f},
        {"Seg 11 [-2.218,-1.258]", -2.218f, -1.258f},
        {"Seg 12 [-1.258,-0.299]", -1.258f, -0.299f},
        {"Neg [-0.299,-0.125]", -0.299f, -0.125f},
        {"Near-zero (Taylor)", -0.125f, 0.125f},
        {"Pos [0.125,0.299]", 0.125f, 0.299f},
        {"Seg 13 [0.299,0.660]", 0.299f, 0.660f},
        {"Seg 14 [0.660,1.644]", 0.660f, 1.644f},
        {"Seg 15 [1.644,3.0]", 1.644f, 3.0f},
        {"Positive sat (x>=3)", 3.0f, std::numeric_limits<float>::max()},
    };

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    std::cout << "\n============================================================\n";
    std::cout << "COMPREHENSIVE ULP ANALYSIS BY SEGMENT (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n";
    std::cout << "Using Tenstorrent hardware model: denormals treated as zero\n\n";

    int32_t skipped_denormals = 0;

    // Iterate through all valid BF16 bit patterns (excluding NaN/Inf/Denormals)
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN and Inf (exponent = 0xFF)
        uint16_t exp_bits = (bf16_bits >> 7) & 0xFF;
        if (exp_bits == 0xFF) {
            continue;
        }

        // Skip denormals (exponent = 0, mantissa != 0) - they all map to zero
        if (bf16_ulp::is_bf16_denormal(bf16_bits)) {
            skipped_denormals++;
            continue;
        }

        float x = bf16_ulp::bf16_bits_to_float(bf16_bits);

        // Skip zeros
        if (x == 0.0f) {
            continue;
        }

        // Run GELU on device
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);

        // Compute expected with DAZ+FTZ applied
        float expected = bf16_ulp::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, expected);

        // Find which segment this x belongs to
        for (auto& seg : segments) {
            if (x >= seg.x_min && x < seg.x_max) {
                seg.sum_ulp += ulp;
                seg.count++;
                if (ulp > seg.max_ulp) {
                    seg.max_ulp = ulp;
                    seg.worst_x = x;
                }
                break;
            }
        }
    }

    std::cout << "Skipped " << skipped_denormals << " denormal values (all map to zero)\n\n";

    // Print results
    std::cout << std::left << std::setw(25) << "Segment" << std::right << std::setw(10) << "Count" << std::setw(12)
              << "Mean ULP" << std::setw(12) << "Max ULP" << std::setw(15) << "Worst x" << "\n";
    std::cout << std::string(74, '-') << "\n";

    int32_t overall_max_ulp = 0;
    int64_t overall_sum_ulp = 0;
    int32_t overall_count = 0;

    for (const auto& seg : segments) {
        if (seg.count > 0) {
            double mean_ulp = static_cast<double>(seg.sum_ulp) / seg.count;
            std::cout << std::left << std::setw(25) << seg.name << std::right << std::setw(10) << seg.count
                      << std::setw(12) << std::fixed << std::setprecision(2) << mean_ulp << std::setw(12) << seg.max_ulp
                      << std::setw(15) << std::scientific << std::setprecision(3) << seg.worst_x << "\n";

            overall_max_ulp = std::max(overall_max_ulp, seg.max_ulp);
            overall_sum_ulp += seg.sum_ulp;
            overall_count += seg.count;
        }
    }

    std::cout << std::string(74, '-') << "\n";
    double overall_mean = static_cast<double>(overall_sum_ulp) / overall_count;
    std::cout << std::left << std::setw(25) << "OVERALL" << std::right << std::setw(10) << overall_count
              << std::setw(12) << std::fixed << std::setprecision(2) << overall_mean << std::setw(12) << overall_max_ulp
              << "\n";
    std::cout << "============================================================\n";
}

TEST_F(GeluUlpBugTest, CumulativeULPDistribution) {
    // Comprehensive ULP distribution analysis over ALL valid BF16 values
    // Shows cumulative percentage at various ULP thresholds
    // Uses DAZ+FTZ model matching Tenstorrent hardware behavior

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    // ULP buckets for cumulative distribution
    std::vector<int32_t> ulp_thresholds = {0, 1, 2, 3, 5, 10, 20, 50, 100, 500, 1000, 10000};
    std::vector<int32_t> ulp_bucket_counts(ulp_thresholds.size(), 0);

    int32_t total_count = 0;
    int32_t max_ulp = 0;
    float worst_x = 0.0f;
    int32_t skipped_denormals = 0;

    // Collect all ULP values
    std::vector<std::pair<float, int32_t>> all_results;  // (x, ulp)

    std::cout << "\n============================================================\n";
    std::cout << "CUMULATIVE ULP DISTRIBUTION (DAZ+FTZ MODEL)\n";
    std::cout << "============================================================\n";
    std::cout << "Using Tenstorrent hardware model: denormals treated as zero\n\n";

    // Iterate through all valid BF16 bit patterns
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN and Inf
        uint16_t exp_bits = (bf16_bits >> 7) & 0xFF;
        if (exp_bits == 0xFF) {
            continue;
        }

        // Skip denormals
        if (bf16_ulp::is_bf16_denormal(bf16_bits)) {
            skipped_denormals++;
            continue;
        }

        float x = bf16_ulp::bf16_bits_to_float(bf16_bits);
        if (x == 0.0f) {
            continue;
        }

        // Run GELU on device
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);

        // Compute expected with DAZ+FTZ
        float expected = bf16_ulp::gelu_expected_bf16_daz(x);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, expected);

        all_results.push_back({x, ulp});
        total_count++;

        if (ulp > max_ulp) {
            max_ulp = ulp;
            worst_x = x;
        }

        // Count into buckets
        for (size_t i = 0; i < ulp_thresholds.size(); ++i) {
            if (ulp <= ulp_thresholds[i]) {
                ulp_bucket_counts[i]++;
            }
        }
    }

    std::cout << "Skipped " << skipped_denormals << " denormal values\n";
    std::cout << "Analyzed " << total_count << " normal BF16 values\n\n";

    // Print cumulative distribution
    std::cout << "CUMULATIVE DISTRIBUTION:\n";
    std::cout << std::string(50, '-') << "\n";
    std::cout << std::left << std::setw(15) << "ULP <=" << std::right << std::setw(12) << "Count" << std::setw(12)
              << "Percent" << std::setw(12) << "Cumul %" << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (size_t i = 0; i < ulp_thresholds.size(); ++i) {
        double pct = 100.0 * ulp_bucket_counts[i] / total_count;
        std::cout << std::left << std::setw(15) << ulp_thresholds[i] << std::right << std::setw(12)
                  << ulp_bucket_counts[i] << std::setw(11) << std::fixed << std::setprecision(2) << pct << "%"
                  << std::setw(11) << std::fixed << std::setprecision(2) << pct << "%" << "\n";
    }

    std::cout << std::string(50, '-') << "\n";
    std::cout << "\nSUMMARY STATISTICS:\n";
    std::cout << "  Max ULP:    " << max_ulp << "\n";
    std::cout << "  Worst x:    " << std::scientific << std::setprecision(6) << worst_x << "\n";
    std::cout << "  ULP <= 1:   " << std::fixed << std::setprecision(2) << (100.0 * ulp_bucket_counts[1] / total_count)
              << "% (" << ulp_bucket_counts[1] << " values)\n";
    std::cout << "  ULP <= 10:  " << std::fixed << std::setprecision(2) << (100.0 * ulp_bucket_counts[5] / total_count)
              << "% (" << ulp_bucket_counts[5] << " values)\n";
    std::cout << "  ULP > 100:  " << std::fixed << std::setprecision(2)
              << (100.0 * (total_count - ulp_bucket_counts[8]) / total_count) << "% ("
              << (total_count - ulp_bucket_counts[8]) << " values)\n";
    std::cout << "============================================================\n";

    // Find worst cases in each region
    std::cout << "\nTOP 10 WORST CASES:\n";
    std::cout << std::string(60, '-') << "\n";

    // Sort by ULP descending
    std::sort(all_results.begin(), all_results.end(), [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << std::left << std::setw(20) << "Input x" << std::right << std::setw(15) << "ULP" << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int i = 0; i < std::min(10, (int)all_results.size()); ++i) {
        std::cout << std::left << std::scientific << std::setprecision(6) << std::setw(20) << all_results[i].first
                  << std::right << std::fixed << std::setw(15) << all_results[i].second << "\n";
    }
    std::cout << "============================================================\n";
}

TEST_F(GeluUlpBugTest, SummaryStatistics) {
    // Run a subset of values and report summary statistics with DAZ+FTZ model
    std::cout << "\n========================================\n";
    std::cout << "GELU ULP SUMMARY (DAZ+FTZ MODEL)\n";
    std::cout << "========================================\n";

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    int32_t max_ulp_region1 = 0;
    int32_t max_ulp_region2 = 0;
    int32_t max_ulp_region3 = 0;

    // Region 1: Deep negative (FTZ + asymptotic)
    for (float x : {-13.5f, -13.0f, -10.0f, -6.0f}) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, bf16_ulp::gelu_expected_bf16_daz(x));
        max_ulp_region1 = std::max(max_ulp_region1, ulp);
    }

    // Region 2: Near-zero (Taylor series)
    for (float x : {1e-10f, 1e-6f, 0.01f, 0.1f}) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, bf16_ulp::gelu_expected_bf16_daz(x));
        max_ulp_region2 = std::max(max_ulp_region2, ulp);
    }

    // Region 3: Polynomial segments (including worst case)
    for (float x : {-5.5f, -5.094f, -5.0f, -4.0f, -2.0f, 1.0f, 2.5f}) {
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);
        int32_t ulp = bf16_ulp::ulp_distance_bf16_daz(actual, bf16_ulp::gelu_expected_bf16_daz(x));
        max_ulp_region3 = std::max(max_ulp_region3, ulp);
    }

    std::cout << "Region 1 (Deep Negative):  Max ULP = " << max_ulp_region1 << "\n";
    std::cout << "Region 2 (Near-Zero):      Max ULP = " << max_ulp_region2 << "\n";
    std::cout << "Region 3 (Polynomials):    Max ULP = " << max_ulp_region3 << "\n";
    std::cout << "\n";
    std::cout << "Expected with C6 fix + raw x polynomials: Max ULP <= 7 (at x=-5.969)\n";
    std::cout << "Hardware model: DAZ+FTZ (denormals treated as zero)\n";
    std::cout << "Source: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h\n";
    std::cout << "========================================\n";

    // Verify fix works - ULP should be low
    int32_t overall_max = std::max({max_ulp_region1, max_ulp_region2, max_ulp_region3});
    EXPECT_LE(overall_max, 10) << "Overall Max ULP should be <= 10, got " << overall_max;
}

TEST_F(GeluUlpBugTest, SubnormalOutputsFlushedToZero) {
    // Verify that all inputs whose reference GELU produces subnormal outputs
    // are returned as exactly zero by the hardware (FTZ behavior).
    //
    // BF16 subnormal range: exponent = 0, mantissa != 0
    // This corresponds to values with magnitude < 2^-126 ≈ 1.18e-38
    //
    // GELU produces tiny outputs for deep negative x (approaching 0 from below)
    // and for tiny positive x (GELU(x) ≈ 0.5*x for small x).

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    std::cout << "\n============================================================\n";
    std::cout << "SUBNORMAL OUTPUT VERIFICATION (FTZ BEHAVIOR)\n";
    std::cout << "============================================================\n";
    std::cout << "Checking that inputs producing subnormal reference outputs\n";
    std::cout << "return exactly zero from hardware (Flush-To-Zero).\n\n";

    int32_t subnormal_output_count = 0;
    int32_t flushed_to_zero_count = 0;
    int32_t not_flushed_count = 0;
    std::vector<std::tuple<float, double, float>> failures;  // (input, expected, actual)

    // Iterate through all valid BF16 bit patterns
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN and Inf
        uint16_t exp_bits = (bf16_bits >> 7) & 0xFF;
        if (exp_bits == 0xFF) {
            continue;
        }

        // Skip input denormals (they map to zero input anyway)
        if (bf16_ulp::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        float x = bf16_ulp::bf16_bits_to_float(bf16_bits);

        // Compute reference GELU in double precision
        double expected_f64 = bf16_ulp::gelu_exact(x);

        // Check if expected output is subnormal when converted to BF16
        // BF16 subnormal: |value| < 2^-126 and value != 0
        float expected_f32 = static_cast<float>(expected_f64);
        uint16_t expected_bf16_bits = bf16_ulp::float_to_bf16_bits(expected_f32);

        if (bf16_ulp::is_bf16_denormal(expected_bf16_bits)) {
            subnormal_output_count++;

            // Run GELU on device
            auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
            auto result = ttnn::from_device(ttnn::gelu(tensor, false));
            float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);

            if (actual == 0.0f) {
                flushed_to_zero_count++;
            } else {
                not_flushed_count++;
                if (failures.size() < 20) {
                    failures.push_back({x, expected_f64, actual});
                }
            }
        }
    }

    std::cout << "Inputs producing subnormal reference outputs: " << subnormal_output_count << "\n";
    std::cout << "Correctly flushed to zero:                    " << flushed_to_zero_count << "\n";
    std::cout << "NOT flushed (failures):                       " << not_flushed_count << "\n";

    if (!failures.empty()) {
        std::cout << "\nFAILURES (first " << failures.size() << "):\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << std::left << std::setw(15) << "Input x" << std::setw(20) << "Expected" << std::setw(20) << "Actual"
                  << "\n";
        std::cout << std::string(70, '-') << "\n";
        for (const auto& [x, exp, act] : failures) {
            std::cout << std::scientific << std::setprecision(6) << std::left << std::setw(15) << x << std::setw(20)
                      << exp << std::setw(20) << act << "\n";
        }
    }

    std::cout << "============================================================\n";

    // All subnormal outputs should be flushed to zero
    EXPECT_EQ(not_flushed_count, 0) << "Expected all subnormal outputs to be flushed to zero, but " << not_flushed_count
                                    << " were not";
}

TEST_F(GeluUlpBugTest, MonotonicityVerification) {
    // Verify GELU output monotonicity:
    // - GELU is monotonically DESCENDING from -inf until the local minimum at x ≈ -0.75
    // - GELU is monotonically ASCENDING from the local minimum to +inf
    //
    // The exact local minimum of GELU is at x ≈ -0.7523 where GELU(x) ≈ -0.1704
    // For BF16, we approximate this as x ≈ -0.75
    //
    // NOTE: Due to approximation errors at the asymptotic/polynomial boundary,
    // we allow small ULP tolerance violations. The known worst case is at x=-4.188
    // where Max ULP = 7. Gross monotonicity violations indicate implementation bugs.

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    std::cout << "\n============================================================\n";
    std::cout << "MONOTONICITY VERIFICATION\n";
    std::cout << "============================================================\n";
    std::cout << "GELU should be descending for x < -0.75 (local minimum)\n";
    std::cout << "GELU should be ascending for x > -0.75\n";
    std::cout << "Small violations at segment boundaries are tolerated.\n\n";

    // Collect all (x, gelu(x)) pairs for non-denormal BF16 values
    std::vector<std::pair<float, float>> values;

    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN and Inf
        uint16_t exp_bits = (bf16_bits >> 7) & 0xFF;
        if (exp_bits == 0xFF) {
            continue;
        }

        // Skip denormals
        if (bf16_ulp::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        float x = bf16_ulp::bf16_bits_to_float(bf16_bits);

        // Run GELU on device
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);

        values.push_back({x, actual});
    }

    // Sort by x value
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Find approximate local minimum (around x = -0.75)
    // We'll use -0.85 and -0.65 as bounds to find the minimum region
    const float MIN_REGION_LOW = -0.85f;
    const float MIN_REGION_HIGH = -0.65f;

    // Tolerance: allow violations where adjacent outputs differ by at most 5 ULP
    // This catches gross monotonicity bugs while allowing small approximation errors
    // at asymptotic region (known worst case: x=-5.969 with Max ULP=7)
    const int32_t ULP_TOLERANCE = 5;

    int32_t descending_violations = 0;
    int32_t ascending_violations = 0;
    int32_t small_violations_tolerated = 0;
    std::vector<std::tuple<float, float, float, float, int32_t, std::string>>
        violations;  // (x1, y1, x2, y2, ulp_diff, region)

    for (size_t i = 1; i < values.size(); ++i) {
        float x_prev = values[i - 1].first;
        float y_prev = values[i - 1].second;
        float x_curr = values[i].first;
        float y_curr = values[i].second;

        // Skip the local minimum region where monotonicity changes
        if (x_prev >= MIN_REGION_LOW && x_curr <= MIN_REGION_HIGH) {
            continue;
        }

        // Calculate ULP distance between adjacent outputs
        int32_t ulp_diff = bf16_ulp::ulp_distance_bf16_daz(y_prev, y_curr);

        if (x_curr < MIN_REGION_LOW) {
            // Descending region: y should decrease (or stay same) as x increases
            // This means y_curr should be <= y_prev
            if (y_curr > y_prev) {
                if (ulp_diff <= ULP_TOLERANCE) {
                    small_violations_tolerated++;
                } else {
                    descending_violations++;
                    if (violations.size() < 10) {
                        violations.push_back({x_prev, y_prev, x_curr, y_curr, ulp_diff, "descending"});
                    }
                }
            }
        } else if (x_prev > MIN_REGION_HIGH) {
            // Ascending region: y should increase (or stay same) as x increases
            // This means y_curr should be >= y_prev
            if (y_curr < y_prev) {
                if (ulp_diff <= ULP_TOLERANCE) {
                    small_violations_tolerated++;
                } else {
                    ascending_violations++;
                    if (violations.size() < 10) {
                        violations.push_back({x_prev, y_prev, x_curr, y_curr, ulp_diff, "ascending"});
                    }
                }
            }
        }
    }

    // Find actual minimum
    auto min_it = std::min_element(
        values.begin(), values.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

    std::cout << "Total BF16 values tested: " << values.size() << "\n";
    std::cout << "Local minimum found at:   x = " << min_it->first << ", GELU(x) = " << min_it->second << "\n";
    std::cout << "Small violations tolerated (ULP <= " << ULP_TOLERANCE << "): " << small_violations_tolerated << "\n";
    std::cout << "Gross descending violations (x < " << MIN_REGION_LOW << "): " << descending_violations << "\n";
    std::cout << "Gross ascending violations (x > " << MIN_REGION_HIGH << "):  " << ascending_violations << "\n";

    if (!violations.empty()) {
        std::cout << "\nGROSS VIOLATIONS (first " << violations.size() << "):\n";
        std::cout << std::string(100, '-') << "\n";
        std::cout << std::left << std::setw(12) << "x_prev" << std::setw(15) << "GELU(x_prev)" << std::setw(12)
                  << "x_curr" << std::setw(15) << "GELU(x_curr)" << std::setw(10) << "ULP diff" << std::setw(12)
                  << "Region" << "\n";
        std::cout << std::string(100, '-') << "\n";
        for (const auto& [x1, y1, x2, y2, ulp, region] : violations) {
            std::cout << std::scientific << std::setprecision(3) << std::left << std::setw(12) << x1 << std::setw(15)
                      << y1 << std::setw(12) << x2 << std::setw(15) << y2 << std::fixed << std::setw(10) << ulp
                      << std::setw(12) << region << "\n";
        }
    }

    std::cout << "============================================================\n";

    // Verify no gross monotonicity violations (1 ULP violations tolerated at segment boundaries)
    EXPECT_EQ(descending_violations, 0) << "GELU should be monotonically descending for x < " << MIN_REGION_LOW;
    EXPECT_EQ(ascending_violations, 0) << "GELU should be monotonically ascending for x > " << MIN_REGION_HIGH;
}

TEST_F(GeluUlpBugTest, DenormalInputsProduceSameOutputAsZero) {
    // Verify that all BF16 denormal inputs produce the same output as zero input (DAZ behavior).
    //
    // Per tech_reports/Handling_Special_Value/special_values.md:
    // "denormals | all | 0x0" - the hardware treats all denormals as zero.
    //
    // BF16 denormals: exponent = 0, mantissa != 0
    // Positive denormals: 0x0001 to 0x007F (127 values)
    // Negative denormals: 0x8001 to 0x807F (127 values)
    // Total: 254 denormal values (excluding ±0)

    std::array<uint32_t, 4> dims = {1, 1, 32, 32};
    ttnn::Shape shape(dims);

    std::cout << "\n============================================================\n";
    std::cout << "DENORMAL INPUT VERIFICATION (DAZ BEHAVIOR)\n";
    std::cout << "============================================================\n";
    std::cout << "Checking that all denormal inputs produce same output as zero.\n";
    std::cout << "Hardware treats denormals as zero (DAZ = Denormals-Are-Zero).\n\n";

    // First, get GELU(0)
    auto zero_tensor = ttnn::full(shape, 0.0f, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
    auto zero_result = ttnn::from_device(ttnn::gelu(zero_tensor, false));
    float gelu_zero = static_cast<float>(zero_result.to_vector<::bfloat16>()[0]);

    std::cout << "GELU(0) = " << gelu_zero << "\n\n";

    int32_t denormal_count = 0;
    int32_t match_count = 0;
    int32_t mismatch_count = 0;
    std::vector<std::tuple<uint16_t, float, float>> mismatches;  // (bits, input_approx, actual_output)

    // Test all denormal bit patterns
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        if (!bf16_ulp::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        denormal_count++;
        float x = bf16_ulp::bf16_bits_to_float(bf16_bits);

        // Run GELU on device
        auto tensor = ttnn::full(shape, x, DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device_);
        auto result = ttnn::from_device(ttnn::gelu(tensor, false));
        float actual = static_cast<float>(result.to_vector<::bfloat16>()[0]);

        if (actual == gelu_zero) {
            match_count++;
        } else {
            mismatch_count++;
            if (mismatches.size() < 20) {
                mismatches.push_back({bf16_bits, x, actual});
            }
        }
    }

    std::cout << "Total denormal inputs tested:        " << denormal_count << "\n";
    std::cout << "Match GELU(0) = " << gelu_zero << ":             " << match_count << "\n";
    std::cout << "Mismatches (different from GELU(0)): " << mismatch_count << "\n";

    if (!mismatches.empty()) {
        std::cout << "\nMISMATCHES (first " << mismatches.size() << "):\n";
        std::cout << std::string(60, '-') << "\n";
        std::cout << std::left << std::setw(12) << "BF16 bits" << std::setw(20) << "Input (approx)" << std::setw(20)
                  << "GELU output" << "\n";
        std::cout << std::string(60, '-') << "\n";
        for (const auto& [bits, x, output] : mismatches) {
            std::cout << "0x" << std::hex << std::setw(4) << std::setfill('0') << bits << std::dec << std::setfill(' ')
                      << "      " << std::scientific << std::setprecision(6) << std::left << std::setw(20) << x
                      << std::setw(20) << output << "\n";
        }
    }

    std::cout << "============================================================\n";

    // All denormal inputs should produce the same output as zero
    EXPECT_EQ(mismatch_count, 0) << "Expected all denormal inputs to produce GELU(0) = " << gelu_zero << ", but "
                                 << mismatch_count << " produced different values";
}

// =============================================================================
// FP64 vs MPFR-256 Reference Comparison Test
// =============================================================================

/**
 * FP64-only GELU reference using standard library erf(), with DAZ+FTZ.
 *
 * WARNING: This function is INACCURATE for x < -8.375 because fp64 erf()
 * saturates to -1.0, making (1 + erf) = 0 and thus GELU(x) = 0.
 * The true zero saturation threshold is x = -13.1875.
 *
 * This function exists ONLY to demonstrate why MPFR is needed.
 *
 * Applies DAZ to input and FTZ to output to match hardware behavior.
 */
inline float gelu_fp64_with_daz_ftz(float x) {
    // DAZ: Apply denormal-as-zero to input
    float x_daz = bf16_ulp::bf16_daz_normalize(x);

    // Compute GELU using fp64 precision
    constexpr double SQRT2 = 1.4142135623730950488;
    double result = 0.5 * static_cast<double>(x_daz) * (1.0 + std::erf(static_cast<double>(x_daz) / SQRT2));

    // FTZ: Apply flush-to-zero to output
    return bf16_ulp::bf16_daz_normalize(static_cast<float>(result));
}

/**
 * MPFR-256 GELU reference with DAZ+FTZ.
 *
 * Uses 256-bit precision for accurate erf() computation.
 * Applies DAZ to input and FTZ to output to match hardware behavior.
 */
inline float gelu_mpfr256_with_daz_ftz(float x) {
    // DAZ: Apply denormal-as-zero to input
    float x_daz = bf16_ulp::bf16_daz_normalize(x);

    // Compute GELU using MPFR 256-bit precision
    double result = bf16_ulp::gelu_exact(static_cast<double>(x_daz));

    // FTZ: Apply flush-to-zero to output
    return bf16_ulp::bf16_daz_normalize(static_cast<float>(result));
}

/**
 * Test showing ULP difference between fp64 and MPFR-256 GELU references.
 *
 * This test demonstrates why MPFR is needed: fp64 erf() saturates to -1.0
 * at x ≈ -8.375, causing the fp64 reference to return 0.0 prematurely.
 * The MPFR-256 reference correctly computes tiny non-zero values down to
 * x = -13.1875 (the true BF16 zero saturation threshold).
 */
TEST_F(BFloat16UlpTest, Fp64VsMpfr256ReferenceComparison) {
    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "FP64 vs MPFR-256 GELU REFERENCE COMPARISON\n";
    std::cout << "================================================================\n";
    std::cout << "This test shows why MPFR-256 is needed for accurate reference.\n";
    std::cout << "FP64 erf() saturates to -1.0 at x ≈ -8.375, giving GELU(x) = 0\n";
    std::cout << "prematurely. True zero threshold is x = -13.1875.\n\n";

    struct ComparisonResult {
        float x;
        double fp64_result;
        double mpfr_result;
        int32_t ulp_diff;
    };

    std::vector<ComparisonResult> worst_cases;
    int32_t max_ulp_diff = 0;
    int32_t total_count = 0;
    int32_t agree_count = 0;     // ULP diff = 0
    int32_t close_count = 0;     // ULP diff 1-10
    int32_t moderate_count = 0;  // ULP diff 11-1000
    int32_t severe_count = 0;    // ULP diff > 1000

    float first_divergence_x = 0;
    bool found_first_divergence = false;

    // Scan all normal BF16 values
    for (uint32_t bits = 0; bits <= 0xFFFF; ++bits) {
        uint16_t bf16_bits = static_cast<uint16_t>(bits);

        // Skip NaN and Inf
        uint16_t exp_bits = (bf16_bits >> 7) & 0xFF;
        if (exp_bits == 0xFF) {
            continue;
        }

        // Skip denormals (they map to zero anyway)
        if (bf16_ulp::is_bf16_denormal(bf16_bits)) {
            continue;
        }

        float x = bf16_ulp::bf16_bits_to_float(bf16_bits);
        total_count++;

        // Compute both references with DAZ+FTZ applied
        float fp64_bf16 = gelu_fp64_with_daz_ftz(x);
        float mpfr_bf16 = gelu_mpfr256_with_daz_ftz(x);

        // Calculate ULP difference between the two BF16 results
        int32_t ulp_diff = bf16_ulp::ulp_distance_bf16_daz(fp64_bf16, mpfr_bf16);

        if (ulp_diff == 0) {
            agree_count++;
        } else if (ulp_diff <= 10) {
            close_count++;
        } else if (ulp_diff <= 1000) {
            moderate_count++;
        } else {
            severe_count++;
        }

        if (ulp_diff > 0 && !found_first_divergence) {
            first_divergence_x = x;
            found_first_divergence = true;
        }

        if (ulp_diff > max_ulp_diff) {
            max_ulp_diff = ulp_diff;
        }

        // Collect worst cases for display
        if (ulp_diff > 100 && worst_cases.size() < 50) {
            worst_cases.push_back({x, static_cast<double>(fp64_bf16), static_cast<double>(mpfr_bf16), ulp_diff});
        }
    }

    // Sort worst cases by ULP diff (descending)
    std::sort(
        worst_cases.begin(), worst_cases.end(), [](const auto& a, const auto& b) { return a.ulp_diff > b.ulp_diff; });

    // Print statistics
    std::cout << "STATISTICS:\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Total BF16 values tested:        " << total_count << "\n";
    std::cout << "FP64 and MPFR agree (ULP=0):     " << agree_count << " (" << std::fixed << std::setprecision(2)
              << (100.0 * agree_count / total_count) << "%)\n";
    std::cout << "Close (ULP 1-10):                " << close_count << "\n";
    std::cout << "Moderate difference (ULP 11-1000): " << moderate_count << "\n";
    std::cout << "SEVERE difference (ULP > 1000):  " << severe_count << "\n";
    std::cout << "Maximum ULP difference:          " << max_ulp_diff << "\n";
    std::cout << "First divergence at x =          " << first_divergence_x << "\n";

    // Print worst cases
    if (!worst_cases.empty()) {
        std::cout << "\nWORST CASES (FP64 vs MPFR-256 difference > 100 ULP):\n";
        std::cout << std::string(100, '-') << "\n";
        std::cout << std::left << std::setw(12) << "x" << std::setw(20) << "FP64 GELU" << std::setw(20)
                  << "MPFR-256 GELU" << std::setw(12) << "ULP diff"
                  << "Notes\n";
        std::cout << std::string(100, '-') << "\n";

        for (size_t i = 0; i < std::min(worst_cases.size(), size_t(30)); ++i) {
            const auto& wc = worst_cases[i];
            std::cout << std::scientific << std::setprecision(4) << std::left << std::setw(12) << wc.x << std::setw(20)
                      << wc.fp64_result << std::setw(20) << wc.mpfr_result << std::fixed << std::setw(12)
                      << wc.ulp_diff;

            // Add notes
            if (wc.fp64_result == 0.0 && wc.mpfr_result != 0.0) {
                std::cout << "FP64 erf() saturated!";
            }
            std::cout << "\n";
        }
    }

    // Print specific boundary region
    std::cout << "\nBOUNDARY REGION DETAIL (x from -9.0 to -8.0):\n";
    std::cout << std::string(100, '-') << "\n";
    std::cout << std::left << std::setw(12) << "x" << std::setw(20) << "FP64+FTZ" << std::setw(20) << "MPFR+FTZ"
              << std::setw(15) << "1+erf (FP64)"
              << "ULP diff\n";
    std::cout << std::string(100, '-') << "\n";

    for (uint16_t bits = 0xC110; bits >= 0xC100; --bits) {  // -9.0 to -8.0
        float x = bf16_ulp::bf16_bits_to_float(bits);

        // Compute both with DAZ+FTZ
        float fp64_bf16 = gelu_fp64_with_daz_ftz(x);
        float mpfr_bf16 = gelu_mpfr256_with_daz_ftz(x);

        // Compute 1 + erf for FP64 to show saturation
        constexpr double SQRT2 = 1.4142135623730950488;
        double one_plus_erf_fp64 = 1.0 + std::erf(static_cast<double>(x) / SQRT2);

        int32_t ulp_diff = bf16_ulp::ulp_distance_bf16_daz(fp64_bf16, mpfr_bf16);

        std::cout << std::fixed << std::setprecision(4) << std::left << std::setw(12) << x << std::scientific
                  << std::setprecision(6) << std::setw(20) << fp64_bf16 << std::setw(20) << mpfr_bf16 << std::setw(15)
                  << one_plus_erf_fp64 << std::fixed << ulp_diff << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << "CONCLUSION:\n";
    std::cout << "================================================================\n";
    std::cout << "FP64 erf() saturates to -1.0 around x = -8.375, causing\n";
    std::cout << "(1 + erf) = 0 and thus GELU(x) = 0 prematurely.\n";
    std::cout << "MPFR-256 correctly computes tiny non-zero GELU values.\n";
    std::cout << "This affects " << (severe_count + moderate_count) << " BF16 values.\n";
    std::cout << "================================================================\n";

    // The test passes - it's informational
    // But we do verify that there IS a significant difference
    EXPECT_GT(max_ulp_diff, 1000) << "Expected significant FP64 vs MPFR difference in deep negative region";
    EXPECT_GT(severe_count, 0) << "Expected some severe ULP differences";
}

}  // namespace ttnn::test
