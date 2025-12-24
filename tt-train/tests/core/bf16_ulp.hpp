// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ============================================================================
// BFloat16 ULP (Units in Last Place) Distance Calculator
// ============================================================================
//
// This module provides precise ULP distance calculation for bfloat16 values.
// ULP distance measures how many representable floating-point values exist
// between two numbers, which is the gold standard for floating-point precision
// comparison.
//
// Key features:
// - Precomputed order table for O(1) ULP distance lookup
// - Handles all bf16 special cases: NaN, Inf, subnormals, signed zeros
// - +0 and -0 are treated as equal (same order index)
// - NaN returns error sentinel (kError = 0xFFFF)
//
// BFloat16 format: sign(1) | exponent(8) | mantissa(7)
// - Same exponent range as float32 (bias 127)
// - Reduced precision (7 bits vs 23 bits mantissa)
//
// Usage:
//   #include "bf16_ulp.hpp"
//   uint16_t dist = bf16_ulp::ulp_distance(0x3F80, 0x3F81);  // 1.0 to next
//
// ============================================================================

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

namespace bf16_ulp {

// Error sentinel for NaN or invalid inputs
constexpr uint16_t kError = 0xFFFF;

// Total number of bf16 bit patterns
constexpr size_t kTableSize = 1u << 16;

// ============================================================================
// BFloat16 Bit Manipulation Utilities
// ============================================================================

/**
 * Convert bf16 bit pattern to float32 value.
 * BFloat16 is the upper 16 bits of a float32.
 */
inline float bf16_bits_to_float32(uint16_t bf16_bits) {
    uint32_t u = static_cast<uint32_t>(bf16_bits) << 16;
    return std::bit_cast<float>(u);
}

/**
 * Convert float32 to bf16 bit pattern (truncation, no rounding).
 */
inline uint16_t float32_to_bf16_bits(float f) {
    uint32_t u = std::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>(u >> 16);
}

/**
 * Check if bf16 bit pattern represents NaN.
 * NaN: exponent = 0xFF (all 1s) AND mantissa != 0
 */
inline bool bf16_is_nan(uint16_t bf16_bits) {
    const uint16_t exp = static_cast<uint16_t>((bf16_bits >> 7) & 0xFFu);
    const uint16_t frac = static_cast<uint16_t>(bf16_bits & 0x7Fu);
    return (exp == 0xFFu) && (frac != 0);
}

/**
 * Check if bf16 bit pattern represents infinity (+inf or -inf).
 * Inf: exponent = 0xFF (all 1s) AND mantissa = 0
 */
inline bool bf16_is_inf(uint16_t bf16_bits) {
    const uint16_t exp = static_cast<uint16_t>((bf16_bits >> 7) & 0xFFu);
    const uint16_t frac = static_cast<uint16_t>(bf16_bits & 0x7Fu);
    return (exp == 0xFFu) && (frac == 0);
}

/**
 * Check if bf16 bit pattern represents a subnormal (denormalized) number.
 * Subnormal: exponent = 0 AND mantissa != 0
 */
inline bool bf16_is_subnormal(uint16_t bf16_bits) {
    const uint16_t exp = static_cast<uint16_t>((bf16_bits >> 7) & 0xFFu);
    const uint16_t frac = static_cast<uint16_t>(bf16_bits & 0x7Fu);
    return (exp == 0) && (frac != 0);
}

/**
 * Check if bf16 bit pattern represents zero (+0 or -0).
 */
inline bool bf16_is_zero(uint16_t bf16_bits) {
    return (bf16_bits & 0x7FFFu) == 0;
}

// ============================================================================
// Order Table Construction
// ============================================================================

namespace detail {

struct Entry {
    uint16_t bits;
    float value;
};

// Sort by float value, with deterministic tie-breaking by bits for +0/-0
inline bool entry_less(const Entry& a, const Entry& b) {
    if (a.value < b.value) {
        return true;
    }
    if (b.value < a.value) {
        return false;
    }
    // Equal values (only +0/-0 case for non-NaN): tie-break by bits
    return a.bits < b.bits;
}

}  // namespace detail

/**
 * Build the order table mapping bf16 bit patterns to monotonic order indices.
 *
 * The order table assigns each numeric bf16 value a unique index from 0 to N-1
 * (where N is the count of distinct numeric values), sorted by numeric value.
 *
 * Properties:
 * - NaN patterns map to kError (0xFFFF)
 * - +0 and -0 share the same order index (they are equal)
 * - -inf has order 0, +inf has the maximum order
 * - Consecutive representable values have consecutive order indices
 *
 * This is computed once and cached for O(1) ULP distance lookups.
 */
inline std::array<uint16_t, kTableSize> build_order_table() {
    std::array<uint16_t, kTableSize> order{};
    order.fill(kError);

    // Collect all numeric (non-NaN) bf16 values
    std::vector<detail::Entry> numeric;
    numeric.reserve(kTableSize);

    for (uint32_t u = 0; u < kTableSize; ++u) {
        const uint16_t bits = static_cast<uint16_t>(u);
        if (bf16_is_nan(bits)) {
            continue;  // Keep as kError
        }
        numeric.push_back(detail::Entry{bits, bf16_bits_to_float32(bits)});
    }

    // Sort by numeric value
    std::sort(numeric.begin(), numeric.end(), detail::entry_less);

    if (numeric.empty()) {
        return order;  // Should never happen
    }

    // Assign order indices: increment only when value strictly increases
    uint16_t current = 0;
    float prev = numeric[0].value;
    order[numeric[0].bits] = current;

    for (size_t i = 1; i < numeric.size(); ++i) {
        float cur = numeric[i].value;

        // Increase index only if strictly greater
        // (handles +0/-0 equality correctly)
        if (prev < cur) {
            ++current;
        }

        order[numeric[i].bits] = current;
        prev = cur;
    }

    return order;
}

/**
 * Get the singleton order table (computed once on first use).
 */
inline const std::array<uint16_t, kTableSize>& get_order_table() {
    static const auto order = build_order_table();
    return order;
}

// ============================================================================
// ULP Distance Calculation
// ============================================================================

/**
 * Compute ULP distance between two bf16 bit patterns.
 *
 * @param a_bits First bf16 value as bit pattern
 * @param b_bits Second bf16 value as bit pattern
 * @return ULP distance, or kError (0xFFFF) if either input is NaN
 *
 * The ULP distance is the number of representable bf16 values between a and b,
 * inclusive of the endpoints minus 1. Specifically:
 * - ulp_distance(x, x) = 0 for any non-NaN x
 * - ulp_distance(x, next_representable(x)) = 1
 * - ulp_distance(+0, -0) = 0 (they are the same value)
 */
inline uint16_t ulp_distance_bits(uint16_t a_bits, uint16_t b_bits) {
    const auto& order = get_order_table();
    const uint16_t oa = order[a_bits];
    const uint16_t ob = order[b_bits];

    if (oa == kError || ob == kError) {
        return kError;
    }

    return (oa >= ob) ? static_cast<uint16_t>(oa - ob) : static_cast<uint16_t>(ob - oa);
}

/**
 * Compute ULP distance between two float32 values interpreted as bf16.
 *
 * @param a First value (will be truncated to bf16)
 * @param b Second value (will be truncated to bf16)
 * @return ULP distance, or kError if either value becomes NaN in bf16
 *
 * Note: This truncates float32 to bf16 without rounding. For precise control,
 * use ulp_distance_bits() with explicit bf16 bit patterns.
 */
inline uint16_t ulp_distance(float a, float b) {
    return ulp_distance_bits(float32_to_bf16_bits(a), float32_to_bf16_bits(b));
}

/**
 * Compute ULP distance using custom order table (for testing).
 */
inline uint16_t ulp_distance_bits(const std::array<uint16_t, kTableSize>& order, uint16_t a_bits, uint16_t b_bits) {
    const uint16_t oa = order[a_bits];
    const uint16_t ob = order[b_bits];

    if (oa == kError || ob == kError) {
        return kError;
    }

    return (oa >= ob) ? static_cast<uint16_t>(oa - ob) : static_cast<uint16_t>(ob - oa);
}

// ============================================================================
// Constants for Key BFloat16 Values
// ============================================================================

// Special values
constexpr uint16_t kPosZero = 0x0000;  // +0
constexpr uint16_t kNegZero = 0x8000;  // -0
constexpr uint16_t kPosInf = 0x7F80;   // +inf
constexpr uint16_t kNegInf = 0xFF80;   // -inf
constexpr uint16_t kQNaN = 0x7FC0;     // Quiet NaN (canonical)

// Common values
constexpr uint16_t kPosOne = 0x3F80;  // +1.0
constexpr uint16_t kNegOne = 0xBF80;  // -1.0

// Smallest/largest finite values
constexpr uint16_t kSmallestPosSubnormal = 0x0001;  // Smallest positive subnormal
constexpr uint16_t kLargestPosFinite = 0x7F7F;      // Largest positive finite
constexpr uint16_t kSmallestNegSubnormal = 0x8001;  // Smallest magnitude negative subnormal
constexpr uint16_t kLargestNegFinite = 0xFF7F;      // Largest magnitude negative finite

// Statistics about bf16 numeric values
constexpr uint32_t kNumNaNPatterns = 254;        // 2 signs * (2^7 - 1) mantissa patterns
constexpr uint32_t kNumNumericPatterns = 65282;  // 65536 - 254
constexpr uint16_t kMaxOrderIndex = 65280;       // 65281 distinct values - 1 (0-indexed)

}  // namespace bf16_ulp
