# GELU BF16 Approximation: Range [-13.2, -5.5]
## Using MPFR 256-bit Precision Reference

---

## Executive Summary

This document describes a high-accuracy implementation of the GELU activation function for BF16 (bfloat16) inputs in the range [-13.2, -5.5], verified against MPFR 256-bit precision reference values.

**Key Results:**
| Metric | Value |
|--------|-------|
| **Maximum ULP Error** | **1** |
| Points with 0 ULP | 161 (98.2%) |
| Points with 1 ULP | 3 (1.8%) |
| Total bf16 points | 164 |

---

## 1. Background

### 1.1 The GELU Function

The GELU (Gaussian Error Linear Unit) activation function is defined as:

$$\text{GELU}(x) = x \cdot \frac{1}{2} \cdot \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

For large negative values of x, the erf function approaches -1, making (1 + erf) approach 0, and thus GELU(x) approaches 0.

### 1.2 BF16 Format

BF16 (bfloat16) is a 16-bit floating-point format with:
- 1 sign bit
- 8 exponent bits (same as fp32)
- 7 mantissa bits

Key characteristics:
- **Smallest normal value**: ~1.175494 × 10⁻³⁸ (`0x0080`)
- **Subnormal values**: With FTZ (flush-to-zero), subnormals are treated as zero

### 1.3 Why MPFR is Required

The standard fp64 `erf()` function saturates to exactly -1.0 for x ≤ -8.375, which causes incorrect GELU calculations:

| x | erf(x/√2) [fp64] | True erf(x/√2) | Issue |
|---|------------------|----------------|-------|
| -8.3125 | -0.9999999999999999 | -0.9999999999999999 | OK |
| **-8.375** | **-1.0** | -0.99999999999999999... | **Saturated!** |
| -9.0 | -1.0 | -0.99999999999999999... | Saturated |
| -13.0 | -1.0 | -0.99999999999999999... | Saturated |

This premature saturation causes fp64-based analysis to incorrectly identify -8.375 as the zero threshold, when the true threshold is **-13.1875**.

---

## 2. Zero Saturation Threshold Analysis

### 2.1 MPFR Reference Implementation

```cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <stdfloat>
#include <bit>
#include <mpfr.h>

// Flush subnormals to zero for bf16
inline std::bfloat16_t ftz_bf16(std::bfloat16_t x) {
    uint16_t bits = std::bit_cast<uint16_t>(x);
    uint16_t exponent = (bits >> 7) & 0xFF;
    if (exponent == 0) {
        return std::bit_cast<std::bfloat16_t>(static_cast<uint16_t>(bits & 0x8000));
    }
    return x;
}

// High-precision GELU using MPFR (256-bit precision)
double gelu_mpfr(float xf, mpfr_prec_t precision = 256) {
    mpfr_t x, sqrt2, x_div_sqrt2, erf_result, one, half, one_plus_erf, result;

    mpfr_init2(x, precision);
    mpfr_init2(sqrt2, precision);
    mpfr_init2(x_div_sqrt2, precision);
    mpfr_init2(erf_result, precision);
    mpfr_init2(one, precision);
    mpfr_init2(half, precision);
    mpfr_init2(one_plus_erf, precision);
    mpfr_init2(result, precision);

    mpfr_set_flt(x, xf, MPFR_RNDN);
    mpfr_set_ui(one, 1, MPFR_RNDN);
    mpfr_set_d(half, 0.5, MPFR_RNDN);
    mpfr_sqrt_ui(sqrt2, 2, MPFR_RNDN);
    mpfr_div(x_div_sqrt2, x, sqrt2, MPFR_RNDN);
    mpfr_erf(erf_result, x_div_sqrt2, MPFR_RNDN);
    mpfr_add(one_plus_erf, one, erf_result, MPFR_RNDN);
    mpfr_mul(result, x, half, MPFR_RNDN);
    mpfr_mul(result, result, one_plus_erf, MPFR_RNDN);

    double gelu_double = mpfr_get_d(result, MPFR_RNDN);

    mpfr_clear(x); mpfr_clear(sqrt2); mpfr_clear(x_div_sqrt2);
    mpfr_clear(erf_result); mpfr_clear(one); mpfr_clear(half);
    mpfr_clear(one_plus_erf); mpfr_clear(result);

    return gelu_double;
}

std::bfloat16_t gelu_mpfr_bf16(std::bfloat16_t input, mpfr_prec_t precision = 256) {
    input = ftz_bf16(input);
    float xf = static_cast<float>(input);
    double gelu = gelu_mpfr(xf, precision);
    std::bfloat16_t result = static_cast<std::bfloat16_t>(static_cast<float>(gelu));
    return ftz_bf16(result);
}
```

### 2.2 Threshold Discovery

Using MPFR 256-bit precision, the exact zero saturation boundary is:

| x | |GELU| (MPFR 256-bit) | bf16 min normal | Result |
|---|----------------------|-----------------|--------|
| -13.0 | 7.952×10⁻³⁸ | 1.175×10⁻³⁸ | **NORMAL** (nonzero) |
| -13.0625 | 3.522×10⁻³⁸ | 1.175×10⁻³⁸ | **NORMAL** (nonzero) |
| **-13.125** | **1.554×10⁻³⁸** | 1.175×10⁻³⁸ | **NORMAL** (last nonzero) |
| **-13.1875** | **6.829×10⁻³⁹** | 1.175×10⁻³⁸ | **SUBNORMAL** → FTZ to 0 |
| -13.25 | 2.989×10⁻³⁹ | 1.175×10⁻³⁸ | SUBNORMAL → FTZ to 0 |

**Conclusion**: The true zero threshold is **x = -13.1875** (bf16: `0xC153`)

---

## 3. Implementation Strategy

### 3.1 Three-Region Approach

```
┌─────────────────────────────────────────────────────────────────┐
│  x ≤ -13.1875          │  Return 0 (zero saturation)           │
├─────────────────────────────────────────────────────────────────┤
│  -13.125 < x ≤ -8.125  │  Lookup table (81 entries, 324 bytes) │
├─────────────────────────────────────────────────────────────────┤
│  -8.125 < x ≤ -5.5     │  Segmented order-4 polynomials        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Lookup Table for Far-Negative Region?

The GELU values in range (-13.125, -8.125] span 23 orders of magnitude (10⁻³⁸ to 10⁻¹⁵). Polynomial approximation cannot achieve acceptable ULP error across such extreme dynamic range. A lookup table of 81 entries (324 bytes) achieves **0 ULP** (perfect accuracy).

---

## 4. Complete Implementation

### 4.1 Header File: `gelu_bf16_neg_mpfr.hpp`

```cpp
// ============================================================================
// GELU BF16 Approximation: Range [-13.2, -5.5]
// Generated using MPFR 256-bit precision reference
// Maximum ULP error: 1 (essentially perfect accuracy)
// ============================================================================

#pragma once

#include <cstdint>
#include <bit>
#include <stdfloat>

namespace gelu_bf16_neg {

// ============================================================================
// BF16 Utilities
// ============================================================================

inline std::bfloat16_t ftz_bf16(std::bfloat16_t x) {
    uint16_t bits = std::bit_cast<uint16_t>(x);
    uint16_t exponent = (bits >> 7) & 0xFF;
    if (exponent == 0) {
        return std::bit_cast<std::bfloat16_t>(static_cast<uint16_t>(bits & 0x8000));
    }
    return x;
}

// ============================================================================
// Zero Saturation Threshold
// For x <= -13.1875, GELU(x) = 0 in bf16 (verified with MPFR 256-bit)
// ============================================================================

static constexpr float ZERO_THRESHOLD = -13.1875f;
static constexpr uint16_t ZERO_THRESHOLD_BITS = 0xC153;

// ============================================================================
// Lookup Table for Far Negative Region: x in (-13.125, -8.125]
// 81 entries, 324 bytes
// Achieves 0 ULP (perfect accuracy)
// ============================================================================

static constexpr uint16_t FAR_NEG_LUT_START = 0xC102;  // -8.125
static constexpr uint16_t FAR_NEG_LUT_END = 0xC152;    // -13.125

// Format: {input_bf16_bits, output_bf16_bits}
static const uint16_t FAR_NEG_LUT[][2] = {
    {0xc152, 0x80a9}, {0xc151, 0x8140}, {0xc150, 0x81d8}, {0xc14f, 0x8273},
    {0xc14e, 0x8308}, {0xc14d, 0x8398}, {0xc14c, 0x8429}, {0xc14b, 0x84bb},
    {0xc14a, 0x854e}, {0xc149, 0x85e3}, {0xc148, 0x8678}, {0xc147, 0x8707},
    {0xc146, 0x8793}, {0xc145, 0x881f}, {0xc144, 0x88ab}, {0xc143, 0x8937},
    {0xc142, 0x89c4}, {0xc141, 0x8a51}, {0xc140, 0x8add}, {0xc13f, 0x8b6a},
    {0xc13e, 0x8bf6}, {0xc13d, 0x8c81}, {0xc13c, 0x8d07}, {0xc13b, 0x8d8c},
    {0xc13a, 0x8e11}, {0xc139, 0x8e96}, {0xc138, 0x8f1a}, {0xc137, 0x8f9e},
    {0xc136, 0x9021}, {0xc135, 0x90a3}, {0xc134, 0x9125}, {0xc133, 0x91a6},
    {0xc132, 0x9227}, {0xc131, 0x92a7}, {0xc130, 0x9327}, {0xc12f, 0x93a5},
    {0xc12e, 0x9423}, {0xc12d, 0x94a1}, {0xc12c, 0x951e}, {0xc12b, 0x959a},
    {0xc12a, 0x9616}, {0xc129, 0x9691}, {0xc128, 0x970c}, {0xc127, 0x9787},
    {0xc126, 0x9801}, {0xc125, 0x9877}, {0xc124, 0x98eb}, {0xc123, 0x995e},
    {0xc122, 0x99d2}, {0xc121, 0x9a45}, {0xc120, 0x9ab8}, {0xc11f, 0x9b2c},
    {0xc11e, 0x9b9f}, {0xc11d, 0x9c14}, {0xc11c, 0x9c88}, {0xc11b, 0x9cf9},
    {0xc11a, 0x9d64}, {0xc119, 0x9dd0}, {0xc118, 0x9e3c}, {0xc117, 0x9eaa},
    {0xc116, 0x9f19}, {0xc115, 0x9f89}, {0xc114, 0x9ff5}, {0xc113, 0xa05a},
    {0xc112, 0xa0c1}, {0xc111, 0xa12b}, {0xc110, 0xa196}, {0xc10f, 0xa203},
    {0xc10e, 0xa265}, {0xc10d, 0xa2c7}, {0xc10c, 0xa32c}, {0xc10b, 0xa394},
    {0xc10a, 0xa3ff}, {0xc109, 0xa45a}, {0xc108, 0xa4ba}, {0xc107, 0xa51e},
    {0xc106, 0xa585}, {0xc105, 0xa5e1}, {0xc104, 0xa63c}, {0xc103, 0xa69d},
    {0xc102, 0xa703}
};

static constexpr int FAR_NEG_LUT_SIZE = sizeof(FAR_NEG_LUT) / sizeof(FAR_NEG_LUT[0]);

// Lookup function using direct indexing (inputs are contiguous in this range)
inline std::bfloat16_t lookup_far_neg(uint16_t x_bits) {
    // FAR_NEG_LUT is indexed from 0xC152 down to 0xC102
    // Index = 0xC152 - x_bits
    int idx = 0xC152 - x_bits;
    if (idx >= 0 && idx < FAR_NEG_LUT_SIZE) {
        return std::bit_cast<std::bfloat16_t>(FAR_NEG_LUT[idx][1]);
    }
    return std::bfloat16_t(0);
}

// ============================================================================
// Polynomial Coefficients for x in (-8.125, -5.5]
// 11 segments, 8 points each (last segment has 2 points)
// Maximum ULP: 1
// ============================================================================

// Polynomial evaluation: p(x) = c0 + (x-xmid)*(c1 + (x-xmid)*(c2 + (x-xmid)*(c3 + (x-xmid)*c4)))
inline double eval_poly4_shifted(double x, double x_mid,
                                  double c0, double c1, double c2, double c3, double c4) {
    double dx = x - x_mid;
    return c0 + dx * (c1 + dx * (c2 + dx * (c3 + dx * c4)));
}

// Segment boundaries (x values)
static constexpr float SEG_BOUNDS[] = {
    -8.125f,   // Start of polynomial region
    -7.78125f,
    -7.53125f,
    -7.28125f,
    -7.03125f,
    -6.78125f,
    -6.53125f,
    -6.28125f,
    -6.03125f,
    -5.78125f,
    -5.53125f,
    -5.5f      // End
};

// Segment 0: [-8.0625, -7.8125], 8 pts, Max ULP=0
static constexpr double SEG0_XMID = -7.9375;
static constexpr double SEG0_C0 = -8.180560468869243e-15;
static constexpr double SEG0_C1 = -6.484898859467758e-14;
static constexpr double SEG0_C2 = -2.523291125856310e-13;
static constexpr double SEG0_C3 = -6.843968240039732e-13;
static constexpr double SEG0_C4 = -1.331450237814045e-12;

// Segment 1: [-7.7812, -7.5625], 8 pts, Max ULP=1
static constexpr double SEG1_XMID = -7.671875;
static constexpr double SEG1_C0 = -6.500450697402105e-14;
static constexpr double SEG1_C1 = -4.978690624902730e-13;
static constexpr double SEG1_C2 = -1.878765233332207e-12;
static constexpr double SEG1_C3 = -4.824423368625657e-12;
static constexpr double SEG1_C4 = -8.662829108288652e-12;

// Segment 2: [-7.5312, -7.3125], 8 pts, Max ULP=0
static constexpr double SEG2_XMID = -7.421875;
static constexpr double SEG2_C0 = -4.285191246178417e-13;
static constexpr double SEG2_C1 = -3.175470320990865e-12;
static constexpr double SEG2_C2 = -1.156774563451189e-11;
static constexpr double SEG2_C3 = -2.858713670532853e-11;
static constexpr double SEG2_C4 = -4.952127380651460e-11;

// Segment 3: [-7.2812, -7.0625], 8 pts, Max ULP=0
static constexpr double SEG3_XMID = -7.171875;
static constexpr double SEG3_C0 = -2.652801618819745e-12;
static constexpr double SEG3_C1 = -1.899694266694665e-11;
static constexpr double SEG3_C2 = -6.677345595430689e-11;
static constexpr double SEG3_C3 = -1.585393708783509e-10;
static constexpr double SEG3_C4 = -2.643424840341101e-10;

// Segment 4: [-7.0312, -6.8125], 8 pts, Max ULP=0
static constexpr double SEG4_XMID = -6.921875;
static constexpr double SEG4_C0 = -1.542562398592692e-11;
static constexpr double SEG4_C1 = -1.066158090094188e-10;
static constexpr double SEG4_C2 = -3.610972901594848e-10;
static constexpr double SEG4_C3 = -8.224960005734471e-10;
static constexpr double SEG4_C4 = -1.317609742095899e-09;

// Segment 5: [-6.7812, -6.5625], 8 pts, Max ULP=0
static constexpr double SEG5_XMID = -6.671875;
static constexpr double SEG5_C0 = -8.425128748225761e-11;
static constexpr double SEG5_C1 = -5.612732063080472e-10;
static constexpr double SEG5_C2 = -1.828990740492901e-09;
static constexpr double SEG5_C3 = -3.990225435968794e-09;
static constexpr double SEG5_C4 = -6.129209359219904e-09;

// Segment 6: [-6.5312, -6.3125], 8 pts, Max ULP=0
static constexpr double SEG6_XMID = -6.421875;
static constexpr double SEG6_C0 = -4.322556608200194e-10;
static constexpr double SEG6_C1 = -2.771329408357892e-09;
static constexpr double SEG6_C2 = -8.674692233269561e-09;
static constexpr double SEG6_C3 = -1.809427712337921e-08;
static constexpr double SEG6_C4 = -2.659096659167986e-08;

// Segment 7: [-6.2812, -6.0625], 8 pts, Max ULP=0
static constexpr double SEG7_XMID = -6.171875;
static constexpr double SEG7_C0 = -2.082544510068984e-09;
static constexpr double SEG7_C1 = -1.283222811647506e-08;
static constexpr double SEG7_C2 = -3.851452122516594e-08;
static constexpr double SEG7_C3 = -7.665664854777053e-08;
static constexpr double SEG7_C4 = -1.075082558237318e-07;

// Segment 8: [-6.0312, -5.8125], 8 pts, Max ULP=0
static constexpr double SEG8_XMID = -5.921875;
static constexpr double SEG8_C0 = -9.424427685609932e-09;
static constexpr double SEG8_C1 = -5.571167607358211e-08;
static constexpr double SEG8_C2 = -1.600207559154601e-07;
static constexpr double SEG8_C3 = -3.032304568148156e-07;
static constexpr double SEG8_C4 = -4.046999703345027e-07;

// Segment 9: [-5.7812, -5.5625], 8 pts, Max ULP=0
static constexpr double SEG9_XMID = -5.671875;
static constexpr double SEG9_C0 = -4.005568370868782e-08;
static constexpr double SEG9_C1 = -2.267462914073533e-07;
static constexpr double SEG9_C2 = -6.219276984480050e-07;
static constexpr double SEG9_C3 = -1.119222455258123e-06;
static constexpr double SEG9_C4 = -1.416892594988746e-06;

// Segment 10: [-5.5312, -5.5000], 2 pts, Max ULP=0
static constexpr double SEG10_XMID = -5.515625;
static constexpr double SEG10_C0 = -9.618875791587677e-08;
static constexpr double SEG10_C1 = -5.282454813763634e-07;
static constexpr double SEG10_C2 = 0.0;
static constexpr double SEG10_C3 = 0.0;
static constexpr double SEG10_C4 = 0.0;

// ============================================================================
// Main Evaluation Function
// ============================================================================

inline std::bfloat16_t gelu_neg_range(std::bfloat16_t input) {
    input = ftz_bf16(input);
    float x = static_cast<float>(input);
    uint16_t x_bits = std::bit_cast<uint16_t>(input);

    // 1. Zero saturation region: x <= -13.1875
    if (x <= ZERO_THRESHOLD) {
        return std::bfloat16_t(0.0f);
    }

    // 2. Far negative region: x in (-13.125, -8.125] - use lookup table
    if (x_bits >= FAR_NEG_LUT_START && x_bits <= FAR_NEG_LUT_END) {
        return lookup_far_neg(x_bits);
    }

    // 3. Polynomial region: x in (-8.125, -5.5]
    double result;

    if (x <= SEG_BOUNDS[1]) {
        result = eval_poly4_shifted(x, SEG0_XMID, SEG0_C0, SEG0_C1, SEG0_C2, SEG0_C3, SEG0_C4);
    } else if (x <= SEG_BOUNDS[2]) {
        result = eval_poly4_shifted(x, SEG1_XMID, SEG1_C0, SEG1_C1, SEG1_C2, SEG1_C3, SEG1_C4);
    } else if (x <= SEG_BOUNDS[3]) {
        result = eval_poly4_shifted(x, SEG2_XMID, SEG2_C0, SEG2_C1, SEG2_C2, SEG2_C3, SEG2_C4);
    } else if (x <= SEG_BOUNDS[4]) {
        result = eval_poly4_shifted(x, SEG3_XMID, SEG3_C0, SEG3_C1, SEG3_C2, SEG3_C3, SEG3_C4);
    } else if (x <= SEG_BOUNDS[5]) {
        result = eval_poly4_shifted(x, SEG4_XMID, SEG4_C0, SEG4_C1, SEG4_C2, SEG4_C3, SEG4_C4);
    } else if (x <= SEG_BOUNDS[6]) {
        result = eval_poly4_shifted(x, SEG5_XMID, SEG5_C0, SEG5_C1, SEG5_C2, SEG5_C3, SEG5_C4);
    } else if (x <= SEG_BOUNDS[7]) {
        result = eval_poly4_shifted(x, SEG6_XMID, SEG6_C0, SEG6_C1, SEG6_C2, SEG6_C3, SEG6_C4);
    } else if (x <= SEG_BOUNDS[8]) {
        result = eval_poly4_shifted(x, SEG7_XMID, SEG7_C0, SEG7_C1, SEG7_C2, SEG7_C3, SEG7_C4);
    } else if (x <= SEG_BOUNDS[9]) {
        result = eval_poly4_shifted(x, SEG8_XMID, SEG8_C0, SEG8_C1, SEG8_C2, SEG8_C3, SEG8_C4);
    } else if (x <= SEG_BOUNDS[10]) {
        result = eval_poly4_shifted(x, SEG9_XMID, SEG9_C0, SEG9_C1, SEG9_C2, SEG9_C3, SEG9_C4);
    } else {
        result = eval_poly4_shifted(x, SEG10_XMID, SEG10_C0, SEG10_C1, SEG10_C2, SEG10_C3, SEG10_C4);
    }

    std::bfloat16_t bf16_result = static_cast<std::bfloat16_t>(static_cast<float>(result));
    return ftz_bf16(bf16_result);
}

} // namespace gelu_bf16_neg
```

---

## 5. Verification Test

### 5.1 Test Program: `verify_gelu.cpp`

```cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <stdfloat>
#include <bit>
#include <mpfr.h>

// Include the generated header (copy content from above or use #include)
#include "gelu_bf16_neg_mpfr.hpp"

// MPFR reference for verification
std::bfloat16_t gelu_mpfr_reference(float xf, mpfr_prec_t precision = 256) {
    mpfr_t x, sqrt2, x_div_sqrt2, erf_result, one, half, one_plus_erf, result;
    mpfr_init2(x, precision); mpfr_init2(sqrt2, precision);
    mpfr_init2(x_div_sqrt2, precision); mpfr_init2(erf_result, precision);
    mpfr_init2(one, precision); mpfr_init2(half, precision);
    mpfr_init2(one_plus_erf, precision); mpfr_init2(result, precision);

    mpfr_set_flt(x, xf, MPFR_RNDN);
    mpfr_set_ui(one, 1, MPFR_RNDN);
    mpfr_set_d(half, 0.5, MPFR_RNDN);
    mpfr_sqrt_ui(sqrt2, 2, MPFR_RNDN);
    mpfr_div(x_div_sqrt2, x, sqrt2, MPFR_RNDN);
    mpfr_erf(erf_result, x_div_sqrt2, MPFR_RNDN);
    mpfr_add(one_plus_erf, one, erf_result, MPFR_RNDN);
    mpfr_mul(result, x, half, MPFR_RNDN);
    mpfr_mul(result, result, one_plus_erf, MPFR_RNDN);

    float gelu_float = mpfr_get_flt(result, MPFR_RNDN);

    mpfr_clear(x); mpfr_clear(sqrt2); mpfr_clear(x_div_sqrt2);
    mpfr_clear(erf_result); mpfr_clear(one); mpfr_clear(half);
    mpfr_clear(one_plus_erf); mpfr_clear(result);

    std::bfloat16_t bf16_result = static_cast<std::bfloat16_t>(gelu_float);
    return gelu_bf16_neg::ftz_bf16(bf16_result);
}

int calculate_ulp(std::bfloat16_t computed, std::bfloat16_t reference) {
    uint16_t comp_bits = std::bit_cast<uint16_t>(computed);
    uint16_t ref_bits = std::bit_cast<uint16_t>(reference);

    if ((comp_bits == 0x0000 || comp_bits == 0x8000) &&
        (ref_bits == 0x0000 || ref_bits == 0x8000)) return 0;

    auto to_signed = [](uint16_t bits) -> int32_t {
        if (bits & 0x8000) return -static_cast<int32_t>(bits & 0x7FFF);
        return static_cast<int32_t>(bits);
    };

    return std::abs(to_signed(comp_bits) - to_signed(ref_bits));
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  VERIFICATION: gelu_bf16_neg_mpfr.hpp" << std::endl;
    std::cout << "  Testing against MPFR 256-bit reference" << std::endl;
    std::cout << "================================================================" << std::endl;

    int total_points = 0;
    int max_ulp = 0;
    int ulp_histogram[10] = {0};

    // Test all bf16 points in range [-13.2, -5.5]
    for (uint16_t bits = 0x8080; bits <= 0xFF7F; ++bits) {
        std::bfloat16_t x = std::bit_cast<std::bfloat16_t>(bits);
        float xf = static_cast<float>(x);

        if (std::isinf(xf) || std::isnan(xf)) continue;
        if (xf < -13.2f || xf > -5.5f) continue;

        total_points++;

        std::bfloat16_t computed = gelu_bf16_neg::gelu_neg_range(x);
        std::bfloat16_t reference = gelu_mpfr_reference(xf);

        int ulp = calculate_ulp(computed, reference);
        max_ulp = std::max(max_ulp, ulp);

        if (ulp < 9) ulp_histogram[ulp]++;
        else ulp_histogram[9]++;

        if (ulp > 1) {
            std::cout << "WARNING: x=" << xf << " ULP=" << ulp << std::endl;
        }
    }

    std::cout << "\nTotal points tested: " << total_points << std::endl;
    std::cout << "Maximum ULP error: " << max_ulp << std::endl;

    std::cout << "\nULP Distribution:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        if (ulp_histogram[i] > 0) {
            std::cout << "  ULP " << (i < 9 ? std::to_string(i) : "9+") << ": "
                      << ulp_histogram[i] << " points ("
                      << std::fixed << std::setprecision(1)
                      << 100.0 * ulp_histogram[i] / total_points << "%)" << std::endl;
        }
    }

    std::cout << "\n================================================================" << std::endl;
    if (max_ulp <= 1) {
        std::cout << "✓ VERIFICATION PASSED: Maximum ULP = " << max_ulp << std::endl;
    } else {
        std::cout << "✗ VERIFICATION FAILED: Maximum ULP = " << max_ulp << std::endl;
    }
    std::cout << "================================================================" << std::endl;

    mpfr_free_cache();
    return (max_ulp <= 1) ? 0 : 1;
}
```

### 5.2 Build and Run

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install libmpfr-dev libgmp-dev

# Compile
g++ -std=c++23 -O3 -o verify_gelu verify_gelu.cpp -lmpfr -lgmp

# Run
./verify_gelu
```

### 5.3 Expected Output

```
================================================================
  VERIFICATION: gelu_bf16_neg_mpfr.hpp
  Testing against MPFR 256-bit reference
================================================================

Total points tested: 164
Maximum ULP error: 1

ULP Distribution:
  ULP 0: 161 points (98.2%)
  ULP 1: 3 points (1.8%)

================================================================
✓ VERIFICATION PASSED: Maximum ULP = 1
================================================================
```

---

## 6. Polynomial Segment Details

| Segment | x Range | x_mid | Max ULP | Points |
|---------|---------|-------|---------|--------|
| 0 | [-8.0625, -7.8125] | -7.9375 | 0 | 8 |
| 1 | [-7.7812, -7.5625] | -7.6719 | 1 | 8 |
| 2 | [-7.5312, -7.3125] | -7.4219 | 0 | 8 |
| 3 | [-7.2812, -7.0625] | -7.1719 | 0 | 8 |
| 4 | [-7.0312, -6.8125] | -6.9219 | 0 | 8 |
| 5 | [-6.7812, -6.5625] | -6.6719 | 0 | 8 |
| 6 | [-6.5312, -6.3125] | -6.4219 | 0 | 8 |
| 7 | [-6.2812, -6.0625] | -6.1719 | 0 | 8 |
| 8 | [-6.0312, -5.8125] | -5.9219 | 0 | 8 |
| 9 | [-5.7812, -5.5625] | -5.6719 | 0 | 8 |
| 10 | [-5.5312, -5.5000] | -5.5156 | 0 | 2 |

---

## 7. Memory Requirements

| Component | Size |
|-----------|------|
| Lookup table | 324 bytes (81 × 4 bytes) |
| Polynomial coefficients | ~600 bytes (11 segments × ~55 bytes) |
| Segment boundaries | 48 bytes (12 floats) |
| **Total** | **~1 KB** |

---

## 8. Key Findings

1. **fp64 erf() is insufficient** for computing reference values in the far-negative region. The function saturates to -1.0 for x ≤ -8.375, giving incorrect zero threshold. MPFR or equivalent arbitrary-precision libraries are required.

2. **The true zero threshold is -13.1875**, not -8.375 as fp64-based analysis suggested. This is where |GELU(x)| falls below the smallest normal bf16 value.

3. **Lookup tables are essential** for the far-negative region (x < -8.125) where GELU values span 23 orders of magnitude. Polynomial approximation cannot handle this dynamic range.

4. **Segmented polynomials achieve ≤1 ULP** for x > -8.125 when using shifted coordinates (x - x_mid) and MPFR-derived coefficients.

---

## 9. Usage Example

```cpp
#include "gelu_bf16_neg_mpfr.hpp"

std::bfloat16_t gelu_bf16_full(std::bfloat16_t x) {
    float xf = static_cast<float>(x);

    // Negative tail: x in [-13.2, -5.5]
    if (xf >= -13.2f && xf <= -5.5f) {
        return gelu_bf16_neg::gelu_neg_range(x);
    }

    // Very negative: x < -13.2
    if (xf < -13.2f) {
        return std::bfloat16_t(0.0f);
    }

    // Other regions: implement separately
    // ...
}
```

---

## Appendix: BF16 Format Reference

```
BF16 Bit Layout: [S][EEEEEEEE][MMMMMMM]
                  1    8         7     = 16 bits

Special Values:
  +0:         0x0000
  -0:         0x8000
  +∞:         0x7F80
  -∞:         0xFF80
  Smallest normal (+): 0x0080 = 2^-126 ≈ 1.175494e-38
  Smallest normal (-): 0x8080 = -2^-126 ≈ -1.175494e-38

Zero Saturation Threshold: 0xC153 = -13.1875
```

---

*Maximum ULP: 1 across all 164 bf16 points in range [-13.2, -5.5].*
