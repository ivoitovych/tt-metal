// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "bf16_ulp.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

// ============================================================================
// BFloat16 ULP Calculator Test Suite
// ============================================================================
//
// Comprehensive tests for the bf16_ulp module which provides precise ULP
// (Units in Last Place) distance calculation for bfloat16 floating-point values.
//
// Test Coverage:
// 1. Order table construction and properties
// 2. Special value handling (NaN, Inf, zeros, subnormals)
// 3. ULP distance calculations at key boundaries
// 4. Symmetry and identity properties
// 5. Large-scale validation
//
// ============================================================================

class BF16ULPTest : public ::testing::Test {
protected:
    const std::array<uint16_t, bf16_ulp::kTableSize>& order = bf16_ulp::get_order_table();
};

// ============================================================================
// Order Table Construction Tests
// ============================================================================

TEST_F(BF16ULPTest, OrderTableSize) {
    EXPECT_EQ(order.size(), bf16_ulp::kTableSize);
}

TEST_F(BF16ULPTest, NaNPatternsCounted) {
    // Count NaN patterns: exp=0xFF && frac!=0
    // 2 signs * (2^7 - 1) mantissa patterns = 254
    int nan_count = 0;
    int error_count = 0;

    for (uint32_t u = 0; u < bf16_ulp::kTableSize; ++u) {
        uint16_t bits = static_cast<uint16_t>(u);
        if (bf16_ulp::bf16_is_nan(bits)) {
            nan_count++;
        }
        if (order[bits] == bf16_ulp::kError) {
            error_count++;
        }
    }

    EXPECT_EQ(nan_count, bf16_ulp::kNumNaNPatterns);
    EXPECT_EQ(error_count, bf16_ulp::kNumNaNPatterns);
}

TEST_F(BF16ULPTest, NumericPatternsCounted) {
    // Numeric patterns = 65536 - 254 NaNs = 65282
    int numeric_count = 0;
    uint16_t max_order = 0;

    for (uint32_t u = 0; u < bf16_ulp::kTableSize; ++u) {
        uint16_t bits = static_cast<uint16_t>(u);
        if (order[bits] != bf16_ulp::kError) {
            numeric_count++;
            if (order[bits] > max_order) {
                max_order = order[bits];
            }
        }
    }

    EXPECT_EQ(numeric_count, bf16_ulp::kNumNumericPatterns);
    // 65281 distinct values (65282 patterns - 1 for +0/-0 sharing) => max index = 65280
    EXPECT_EQ(max_order, bf16_ulp::kMaxOrderIndex);
}

TEST_F(BF16ULPTest, NaNMapsToError) {
    // All NaN patterns should map to kError
    for (uint32_t u = 0; u < bf16_ulp::kTableSize; ++u) {
        uint16_t bits = static_cast<uint16_t>(u);
        bool is_nan = bf16_ulp::bf16_is_nan(bits);
        EXPECT_EQ(order[bits] == bf16_ulp::kError, is_nan) << "bits=0x" << std::hex << bits << " is_nan=" << is_nan;
    }
}

// ============================================================================
// Special Value Order Tests
// ============================================================================

TEST_F(BF16ULPTest, InfinityOrdering) {
    // -inf should have order 0 (smallest)
    EXPECT_EQ(order[bf16_ulp::kNegInf], 0);

    // +inf should have max order
    EXPECT_EQ(order[bf16_ulp::kPosInf], bf16_ulp::kMaxOrderIndex);

    // Both are valid (not NaN)
    EXPECT_NE(order[bf16_ulp::kNegInf], bf16_ulp::kError);
    EXPECT_NE(order[bf16_ulp::kPosInf], bf16_ulp::kError);
}

TEST_F(BF16ULPTest, ZeroOrdering) {
    // +0 and -0 should have the same order (they are equal)
    EXPECT_EQ(order[bf16_ulp::kPosZero], order[bf16_ulp::kNegZero]);

    // Both should be valid
    EXPECT_NE(order[bf16_ulp::kPosZero], bf16_ulp::kError);
    EXPECT_NE(order[bf16_ulp::kNegZero], bf16_ulp::kError);
}

TEST_F(BF16ULPTest, OneOrdering) {
    // -1 < 0 < +1
    EXPECT_LT(order[bf16_ulp::kNegOne], order[bf16_ulp::kPosZero]);
    EXPECT_LT(order[bf16_ulp::kPosZero], order[bf16_ulp::kPosOne]);
}

TEST_F(BF16ULPTest, FiniteBoundaryOrdering) {
    // -inf < largest magnitude negative finite
    EXPECT_LT(order[bf16_ulp::kNegInf], order[bf16_ulp::kLargestNegFinite]);
    EXPECT_EQ(order[bf16_ulp::kLargestNegFinite], 1);

    // largest positive finite < +inf
    EXPECT_LT(order[bf16_ulp::kLargestPosFinite], order[bf16_ulp::kPosInf]);
    EXPECT_EQ(order[bf16_ulp::kLargestPosFinite], bf16_ulp::kMaxOrderIndex - 1);
}

// ============================================================================
// ULP Distance Basic Tests
// ============================================================================

TEST_F(BF16ULPTest, ULPDistanceIdentity) {
    // ulp_distance(x, x) = 0 for any non-NaN value
    for (uint32_t u = 0; u < bf16_ulp::kTableSize; u += 17) {  // Sample every 17th
        uint16_t bits = static_cast<uint16_t>(u);
        if (bf16_ulp::bf16_is_nan(bits)) {
            EXPECT_EQ(bf16_ulp::ulp_distance_bits(bits, bits), bf16_ulp::kError);
        } else {
            EXPECT_EQ(bf16_ulp::ulp_distance_bits(bits, bits), 0) << "bits=0x" << std::hex << bits;
        }
    }
}

TEST_F(BF16ULPTest, ULPDistanceSymmetry) {
    // ulp_distance(a, b) = ulp_distance(b, a)
    uint32_t seed = 0xDEADBEEF;
    auto xorshift = [&seed]() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return seed;
    };

    for (int i = 0; i < 10000; ++i) {
        uint16_t a = static_cast<uint16_t>(xorshift());
        uint16_t b = static_cast<uint16_t>(xorshift());

        uint16_t dab = bf16_ulp::ulp_distance_bits(a, b);
        uint16_t dba = bf16_ulp::ulp_distance_bits(b, a);
        EXPECT_EQ(dab, dba) << "a=0x" << std::hex << a << " b=0x" << b;
    }
}

TEST_F(BF16ULPTest, ULPDistanceZeros) {
    // +0 and -0 should have ULP distance 0
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kPosZero, bf16_ulp::kNegZero), 0);
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kNegZero, bf16_ulp::kPosZero), 0);
}

TEST_F(BF16ULPTest, ULPDistanceNaN) {
    // Any distance involving NaN should return kError
    uint16_t qnan = bf16_ulp::kQNaN;
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(qnan, bf16_ulp::kPosOne), bf16_ulp::kError);
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kPosOne, qnan), bf16_ulp::kError);
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(qnan, qnan), bf16_ulp::kError);

    // Different NaN patterns
    uint16_t snan = 0x7F81;
    EXPECT_TRUE(bf16_ulp::bf16_is_nan(snan));
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(qnan, snan), bf16_ulp::kError);
}

// ============================================================================
// ULP Distance Neighbor Tests
// ============================================================================

TEST_F(BF16ULPTest, ULPDistanceConsecutivePositive) {
    // Consecutive positive normal values should have ULP distance 1
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0x3F80, 0x3F81), 1);  // 1.0 to next
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0x4000, 0x4001), 1);  // 2.0 to next
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0x0001, 0x0002), 1);  // Smallest subnormals
}

TEST_F(BF16ULPTest, ULPDistanceConsecutiveNegative) {
    // Consecutive negative values
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0x8001, 0x8002), 1);  // Negative subnormals
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0xBF80, 0xBF81), 1);  // -1.0 to next magnitude
}

TEST_F(BF16ULPTest, ULPDistanceInfToFinite) {
    // -inf to largest magnitude negative: 1 ULP
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kNegInf, bf16_ulp::kLargestNegFinite), 1);

    // Largest positive finite to +inf: 1 ULP
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kLargestPosFinite, bf16_ulp::kPosInf), 1);
}

TEST_F(BF16ULPTest, ULPDistanceSubnormalToNormal) {
    // Positive subnormal to normal transition: 0x007F to 0x0080
    // 0x007F = largest positive subnormal
    // 0x0080 = smallest positive normal
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0x007F, 0x0080), 1);

    // Negative side: 0x807F to 0x8080
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(0x807F, 0x8080), 1);
}

TEST_F(BF16ULPTest, ULPDistanceAcrossZero) {
    // Smallest negative subnormal to zero: 1 ULP
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kSmallestNegSubnormal, bf16_ulp::kPosZero), 1);

    // Zero to smallest positive subnormal: 1 ULP
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kPosZero, bf16_ulp::kSmallestPosSubnormal), 1);

    // Smallest negative to smallest positive through zero: 2 ULPs
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kSmallestNegSubnormal, bf16_ulp::kSmallestPosSubnormal), 2);
}

// ============================================================================
// ULP Distance Large Range Tests
// ============================================================================

TEST_F(BF16ULPTest, ULPDistanceFullRange) {
    // -inf to +inf should be max_order
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kNegInf, bf16_ulp::kPosInf), bf16_ulp::kMaxOrderIndex);
}

TEST_F(BF16ULPTest, ULPDistanceLargeFinite) {
    // Largest magnitude negative finite to largest magnitude positive finite
    // Should be max_order - 2 (excluding the two infinities)
    EXPECT_EQ(
        bf16_ulp::ulp_distance_bits(bf16_ulp::kLargestNegFinite, bf16_ulp::kLargestPosFinite),
        bf16_ulp::kMaxOrderIndex - 2);
}

TEST_F(BF16ULPTest, ULPDistanceInfToSmallestMagnitude) {
    // -inf to smallest magnitude negative subnormal
    // order(-inf) = 0, order(-smallest subnormal) = order(zero) - 1
    uint16_t zero_order = order[bf16_ulp::kPosZero];
    uint16_t neg_smallest_order = order[bf16_ulp::kSmallestNegSubnormal];
    EXPECT_EQ(neg_smallest_order, zero_order - 1);

    // Distance should be zero_order - 1
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kNegInf, bf16_ulp::kSmallestNegSubnormal), zero_order - 1);

    // +inf to smallest positive subnormal
    uint16_t pos_smallest_order = order[bf16_ulp::kSmallestPosSubnormal];
    EXPECT_EQ(pos_smallest_order, zero_order + 1);

    uint16_t expected_dist = bf16_ulp::kMaxOrderIndex - (zero_order + 1);
    EXPECT_EQ(bf16_ulp::ulp_distance_bits(bf16_ulp::kSmallestPosSubnormal, bf16_ulp::kPosInf), expected_dist);
}

// ============================================================================
// Order Table Monotonicity Test
// ============================================================================

TEST_F(BF16ULPTest, OrderTableMonotonicity) {
    // Verify that the order table is monotonic with respect to float values
    // Build sorted list of numeric values and verify order increments correctly

    struct Entry {
        uint16_t bits;
        float value;
    };

    std::vector<Entry> numeric;
    numeric.reserve(bf16_ulp::kTableSize);

    for (uint32_t u = 0; u < bf16_ulp::kTableSize; ++u) {
        uint16_t bits = static_cast<uint16_t>(u);
        if (!bf16_ulp::bf16_is_nan(bits)) {
            numeric.push_back({bits, bf16_ulp::bf16_bits_to_float32(bits)});
        }
    }

    // Sort by value (with tie-break by bits for +0/-0)
    std::sort(numeric.begin(), numeric.end(), [](const Entry& a, const Entry& b) {
        if (a.value < b.value) {
            return true;
        }
        if (b.value < a.value) {
            return false;
        }
        return a.bits < b.bits;
    });

    EXPECT_EQ(numeric.size(), bf16_ulp::kNumNumericPatterns);

    // First should be -inf, last should be +inf
    EXPECT_EQ(numeric.front().bits, bf16_ulp::kNegInf);
    EXPECT_EQ(numeric.back().bits, bf16_ulp::kPosInf);

    // Verify monotonicity: order should never decrease
    uint16_t prev_order = order[numeric[0].bits];
    float prev_value = numeric[0].value;

    for (size_t i = 1; i < numeric.size(); ++i) {
        uint16_t cur_order = order[numeric[i].bits];
        float cur_value = numeric[i].value;

        EXPECT_GE(cur_order, prev_order) << "Order decreased at index " << i << " bits=0x" << std::hex
                                         << numeric[i].bits;

        // Order should increase exactly when value strictly increases
        if (prev_value < cur_value) {
            EXPECT_EQ(cur_order, static_cast<uint16_t>(prev_order + 1))
                << "Order didn't increment when value increased at index " << i;
        } else {
            // Equal values should have same order (only +0/-0)
            EXPECT_EQ(cur_order, prev_order);
        }

        prev_order = cur_order;
        prev_value = cur_value;
    }
}

// ============================================================================
// Order Count Per Index Test
// ============================================================================

TEST_F(BF16ULPTest, OrderCountPerIndex) {
    // Each order index should have exactly one pattern, except for the
    // zero index which has two (+0 and -0)

    std::vector<uint32_t> counts(bf16_ulp::kMaxOrderIndex + 1, 0);

    for (uint32_t u = 0; u < bf16_ulp::kTableSize; ++u) {
        uint16_t bits = static_cast<uint16_t>(u);
        uint16_t o = order[bits];
        if (o != bf16_ulp::kError) {
            counts[o]++;
        }
    }

    uint32_t total = 0;
    uint32_t multi_count = 0;
    uint16_t multi_order = 0;

    for (uint16_t o = 0; o <= bf16_ulp::kMaxOrderIndex; ++o) {
        total += counts[o];
        if (counts[o] > 1) {
            multi_count++;
            multi_order = o;
        }
        // Each index should have at least one pattern
        EXPECT_GE(counts[o], 1u) << "Order index " << o << " has no patterns";
        // Each index should have at most 2 patterns
        EXPECT_LE(counts[o], 2u) << "Order index " << o << " has too many patterns";
    }

    EXPECT_EQ(total, bf16_ulp::kNumNumericPatterns);
    // Only one order index should have multiple patterns (the zeros)
    EXPECT_EQ(multi_count, 1u);
    EXPECT_EQ(counts[multi_order], 2u);

    // The multi_order should be the zero order
    EXPECT_EQ(order[bf16_ulp::kPosZero], multi_order);
    EXPECT_EQ(order[bf16_ulp::kNegZero], multi_order);
}

// ============================================================================
// Float32 Interface Test
// ============================================================================

TEST_F(BF16ULPTest, ULPDistanceFloat32Interface) {
    // Test the float32 interface
    EXPECT_EQ(bf16_ulp::ulp_distance(1.0f, 1.0f), 0);
    EXPECT_EQ(bf16_ulp::ulp_distance(0.0f, -0.0f), 0);

    // 1.0f truncates to bf16 0x3F80
    float one = bf16_ulp::bf16_bits_to_float32(0x3F80);
    float next = bf16_ulp::bf16_bits_to_float32(0x3F81);
    EXPECT_EQ(bf16_ulp::ulp_distance(one, next), 1);

    // NaN should return error
    float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_EQ(bf16_ulp::ulp_distance(nan, 1.0f), bf16_ulp::kError);
}

// ============================================================================
// Bit Conversion Utilities Test
// ============================================================================

TEST_F(BF16ULPTest, BitConversionRoundTrip) {
    // For all bf16 patterns, converting to float and back should give same bits
    // (except for NaN which may canonicalize)
    for (uint32_t u = 0; u < bf16_ulp::kTableSize; ++u) {
        uint16_t bits = static_cast<uint16_t>(u);
        if (bf16_ulp::bf16_is_nan(bits)) {
            continue;  // NaN may not round-trip exactly
        }

        float f = bf16_ulp::bf16_bits_to_float32(bits);
        uint16_t back = bf16_ulp::float32_to_bf16_bits(f);
        EXPECT_EQ(back, bits) << "Round-trip failed for bits=0x" << std::hex << bits;
    }
}

TEST_F(BF16ULPTest, SpecialValueConversion) {
    // Test conversion of special values
    EXPECT_TRUE(std::isinf(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kPosInf)));
    EXPECT_GT(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kPosInf), 0);

    EXPECT_TRUE(std::isinf(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kNegInf)));
    EXPECT_LT(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kNegInf), 0);

    EXPECT_TRUE(std::isnan(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kQNaN)));

    EXPECT_EQ(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kPosZero), 0.0f);
    EXPECT_EQ(bf16_ulp::bf16_bits_to_float32(bf16_ulp::kNegZero), 0.0f);
    // Distinguish +0/-0 by reciprocal
    EXPECT_GT(1.0f / bf16_ulp::bf16_bits_to_float32(bf16_ulp::kPosZero), 0);
    EXPECT_LT(1.0f / bf16_ulp::bf16_bits_to_float32(bf16_ulp::kNegZero), 0);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(BF16ULPTest, IsNaNDetection) {
    // Test NaN detection
    EXPECT_TRUE(bf16_ulp::bf16_is_nan(0x7FC0));  // Canonical qNaN
    EXPECT_TRUE(bf16_ulp::bf16_is_nan(0xFFC0));  // Negative qNaN
    EXPECT_TRUE(bf16_ulp::bf16_is_nan(0x7F81));  // sNaN
    EXPECT_TRUE(bf16_ulp::bf16_is_nan(0x7FFF));  // All mantissa bits set

    EXPECT_FALSE(bf16_ulp::bf16_is_nan(0x7F80));  // +inf (not NaN)
    EXPECT_FALSE(bf16_ulp::bf16_is_nan(0xFF80));  // -inf (not NaN)
    EXPECT_FALSE(bf16_ulp::bf16_is_nan(0x0000));  // +0
    EXPECT_FALSE(bf16_ulp::bf16_is_nan(0x3F80));  // 1.0
}

TEST_F(BF16ULPTest, IsInfDetection) {
    EXPECT_TRUE(bf16_ulp::bf16_is_inf(0x7F80));  // +inf
    EXPECT_TRUE(bf16_ulp::bf16_is_inf(0xFF80));  // -inf

    EXPECT_FALSE(bf16_ulp::bf16_is_inf(0x7FC0));  // qNaN
    EXPECT_FALSE(bf16_ulp::bf16_is_inf(0x0000));  // +0
    EXPECT_FALSE(bf16_ulp::bf16_is_inf(0x7F7F));  // Largest finite
}

TEST_F(BF16ULPTest, IsSubnormalDetection) {
    EXPECT_TRUE(bf16_ulp::bf16_is_subnormal(0x0001));  // Smallest positive subnormal
    EXPECT_TRUE(bf16_ulp::bf16_is_subnormal(0x007F));  // Largest positive subnormal
    EXPECT_TRUE(bf16_ulp::bf16_is_subnormal(0x8001));  // Smallest magnitude negative subnormal

    EXPECT_FALSE(bf16_ulp::bf16_is_subnormal(0x0000));  // +0 (not subnormal)
    EXPECT_FALSE(bf16_ulp::bf16_is_subnormal(0x0080));  // Smallest positive normal
    EXPECT_FALSE(bf16_ulp::bf16_is_subnormal(0x3F80));  // 1.0
}

TEST_F(BF16ULPTest, IsZeroDetection) {
    EXPECT_TRUE(bf16_ulp::bf16_is_zero(0x0000));  // +0
    EXPECT_TRUE(bf16_ulp::bf16_is_zero(0x8000));  // -0

    EXPECT_FALSE(bf16_ulp::bf16_is_zero(0x0001));  // Smallest subnormal
    EXPECT_FALSE(bf16_ulp::bf16_is_zero(0x3F80));  // 1.0
}
