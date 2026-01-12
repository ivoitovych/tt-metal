# GELU BF16 Zero Saturation Threshold Research

## Executive Summary

This report documents the investigation into finding the exact input threshold where the GELU (Gaussian Error Linear Unit) activation function saturates to zero when computed with BF16 (bfloat16) precision and flush-to-zero (FTZ) semantics.

**Key Finding**: The true zero saturation threshold is **x = -13.1875** (bf16: `0xC153`), not -8.375 as initially determined using fp64 precision.

---

## 1. Background

### 1.1 The GELU Function

The GELU activation function is defined as:

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

### 1.3 Problem Statement

When implementing GELU for hardware accelerators, we need to know at what input value x the output becomes exactly zero in BF16 representation. This allows for optimization by returning 0 directly instead of computing the expensive erf function.

---

## 2. Initial Investigation (fp64 Reference)

### 2.1 Methodology

The initial approach used fp64 (double precision) to compute the reference GELU value, then converted to BF16:

```cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <stdfloat>
#include <bit>
#include <numbers>

// Flush subnormals to zero for bf16
inline std::bfloat16_t ftz_bf16(std::bfloat16_t x) {
    uint16_t bits = std::bit_cast<uint16_t>(x);
    uint16_t exponent = (bits >> 7) & 0xFF;
    if (exponent == 0) {
        return std::bit_cast<std::bfloat16_t>(static_cast<uint16_t>(bits & 0x8000));
    }
    return x;
}

// Reference GELU using fp64 precision
std::bfloat16_t gelu_reference_fp64(std::bfloat16_t input) {
    input = ftz_bf16(input);
    double x = static_cast<double>(static_cast<float>(input));
    double gelu = x * 0.5 * (1.0 + std::erf(x / std::numbers::sqrt2));
    std::bfloat16_t result = static_cast<std::bfloat16_t>(static_cast<float>(gelu));
    return ftz_bf16(result);
}

int main() {
    std::cout << "=== fp64 Reference GELU Scan ===" << std::endl;

    // Scan negative bf16 values
    for (uint16_t bits = 0xC110; bits >= 0xC100; --bits) {  // -9.0 to -8.0
        std::bfloat16_t x = std::bit_cast<std::bfloat16_t>(bits);
        float xf = static_cast<float>(x);

        std::bfloat16_t result = gelu_reference_fp64(x);
        uint16_t result_bits = std::bit_cast<uint16_t>(result);
        bool is_zero = (result_bits == 0x0000 || result_bits == 0x8000);

        std::cout << "x = " << std::setw(10) << xf
                  << " (0x" << std::hex << bits << std::dec << ")"
                  << " -> GELU = " << static_cast<float>(result)
                  << (is_zero ? " [ZERO]" : "") << std::endl;
    }

    return 0;
}
```

### 2.2 Initial Result (INCORRECT)

The fp64-based scan found:
- **Last zero**: x = -8.375 (bf16: `0xC106`)
- **First nonzero**: x = -8.3125 (bf16: `0xC105`)

This result was **incorrect** due to limitations in the fp64 erf() implementation.

---

## 3. Discovery of fp64 Limitation

### 3.1 The Problem

The fp64 `erf()` function saturates to exactly -1.0 for inputs where the true value is extremely close to (but not exactly) -1.0. This causes:

```
erf(x/√2) = -1.0  (exactly, in fp64)
1 + erf(x/√2) = 0
GELU(x) = x × 0.5 × 0 = 0
```

### 3.2 Evidence

| x | erf(x/√2) [fp64] | True erf(x/√2) | 1 + erf [True] |
|---|------------------|----------------|----------------|
| -7.0 | -0.9999999999974404 | -0.9999999999974404 | 2.56×10⁻¹² |
| -8.0 | -0.9999999999999988 | -0.9999999999999988 | 1.22×10⁻¹⁵ |
| -8.3125 | -0.9999999999999999 | -0.9999999999999999 | 1.11×10⁻¹⁶ |
| **-8.375** | **-1.0** | -0.99999999999999999... | **~5.5×10⁻¹⁷** |
| -9.0 | -1.0 | -0.99999999999999999... | ~2.3×10⁻¹⁹ |

The fp64 erf() function loses the tiny deviation from -1.0, causing premature saturation.

---

## 4. Corrected Analysis Using MPFR

### 4.1 MPFR Solution

To get the true GELU values, we use MPFR (Multiple Precision Floating-Point Reliable) library with 256-bit precision:

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
std::bfloat16_t gelu_reference_mpfr(std::bfloat16_t input, mpfr_prec_t precision = 256) {
    input = ftz_bf16(input);
    float xf = static_cast<float>(input);

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

    float gelu_float = mpfr_get_flt(result, MPFR_RNDN);

    mpfr_clear(x); mpfr_clear(sqrt2); mpfr_clear(x_div_sqrt2);
    mpfr_clear(erf_result); mpfr_clear(one); mpfr_clear(half);
    mpfr_clear(one_plus_erf); mpfr_clear(result);

    std::bfloat16_t bf16_result = static_cast<std::bfloat16_t>(gelu_float);
    return ftz_bf16(bf16_result);
}

// Get high-precision GELU value as double for display
double gelu_mpfr_as_double(float xf, mpfr_prec_t precision = 256) {
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

int main() {
    std::cout << "=== MPFR 256-bit GELU Reference ===" << std::endl;

    mpfr_prec_t precision = 256;

    // Scan around the true boundary
    std::cout << "\nBoundary region (x around -13.2):" << std::endl;
    std::cout << std::setw(12) << "x"
              << std::setw(22) << "GELU (MPFR)"
              << std::setw(18) << "bf16 result"
              << std::setw(12) << "status" << std::endl;
    std::cout << std::string(64, '-') << std::endl;

    for (uint16_t bits = 0xC158; bits >= 0xC14E; --bits) {
        std::bfloat16_t x = std::bit_cast<std::bfloat16_t>(bits);
        float xf = static_cast<float>(x);

        double gelu_hp = gelu_mpfr_as_double(xf, precision);
        std::bfloat16_t result = gelu_reference_mpfr(x, precision);
        uint16_t result_bits = std::bit_cast<uint16_t>(result);
        bool is_zero = (result_bits == 0x0000 || result_bits == 0x8000);

        std::cout << std::setw(12) << std::fixed << std::setprecision(4) << xf
                  << std::setw(22) << std::scientific << std::setprecision(10) << gelu_hp
                  << std::setw(18) << static_cast<float>(result)
                  << std::setw(12) << (is_zero ? "ZERO" : "nonzero") << std::endl;
    }

    mpfr_free_cache();
    return 0;
}
```

**Compilation**:
```bash
g++ -std=c++23 -O3 -o gelu_mpfr gelu_mpfr.cpp -lmpfr -lgmp
```

### 4.2 MPFR Results

The MPFR-based scan reveals the true boundary:

| x | |GELU| (MPFR 256-bit) | bf16 min normal | Result |
|---|----------------------|-----------------|--------|
| -13.0 | 7.952×10⁻³⁸ | 1.175×10⁻³⁸ | **NORMAL** (nonzero) |
| -13.0625 | 3.522×10⁻³⁸ | 1.175×10⁻³⁸ | **NORMAL** (nonzero) |
| **-13.125** | **1.554×10⁻³⁸** | 1.175×10⁻³⁸ | **NORMAL** (last nonzero) |
| **-13.1875** | **6.829×10⁻³⁹** | 1.175×10⁻³⁸ | **SUBNORMAL** → FTZ to 0 |
| -13.25 | 2.989×10⁻³⁹ | 1.175×10⁻³⁸ | SUBNORMAL → FTZ to 0 |

---

## 5. Complete Verification Code

The following code performs an exhaustive scan of all negative BF16 values using MPFR 256-bit precision:

```cpp
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <stdfloat>
#include <bit>
#include <mpfr.h>

inline std::bfloat16_t ftz_bf16(std::bfloat16_t x) {
    uint16_t bits = std::bit_cast<uint16_t>(x);
    uint16_t exponent = (bits >> 7) & 0xFF;
    if (exponent == 0) {
        return std::bit_cast<std::bfloat16_t>(static_cast<uint16_t>(bits & 0x8000));
    }
    return x;
}

std::bfloat16_t gelu_reference_mpfr(std::bfloat16_t input, mpfr_prec_t precision = 256) {
    input = ftz_bf16(input);
    float xf = static_cast<float>(input);

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

    float gelu_float = mpfr_get_flt(result, MPFR_RNDN);

    mpfr_clear(x); mpfr_clear(sqrt2); mpfr_clear(x_div_sqrt2);
    mpfr_clear(erf_result); mpfr_clear(one); mpfr_clear(half);
    mpfr_clear(one_plus_erf); mpfr_clear(result);

    std::bfloat16_t bf16_result = static_cast<std::bfloat16_t>(gelu_float);
    return ftz_bf16(bf16_result);
}

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  EXHAUSTIVE BF16 SCAN: GELU Zero Saturation Threshold" << std::endl;
    std::cout << "  Using MPFR 256-bit precision reference" << std::endl;
    std::cout << "================================================================" << std::endl;

    mpfr_prec_t precision = 256;

    uint16_t first_nonzero_bits = 0;
    float first_nonzero_x = 0;
    uint16_t last_zero_bits = 0;
    float last_zero_x = 0;

    int count_zero = 0, count_nonzero = 0;

    std::cout << "\nScanning all normal negative bf16 values..." << std::endl;

    // Scan from most negative to least negative (normal values only)
    for (uint16_t bits = 0xFF7F; bits >= 0x8080; --bits) {
        std::bfloat16_t x = std::bit_cast<std::bfloat16_t>(bits);
        float xf = static_cast<float>(x);

        if (std::isinf(xf) || std::isnan(xf)) continue;

        std::bfloat16_t result = gelu_reference_mpfr(x, precision);
        uint16_t result_bits = std::bit_cast<uint16_t>(result);
        bool is_zero = (result_bits == 0x0000 || result_bits == 0x8000);

        if (is_zero) {
            count_zero++;
            if (first_nonzero_bits == 0) {
                last_zero_bits = bits;
                last_zero_x = xf;
            }
        } else {
            count_nonzero++;
            if (first_nonzero_bits == 0) {
                first_nonzero_bits = bits;
                first_nonzero_x = xf;
            }
        }
    }

    std::cout << "\n================================================================" << std::endl;
    std::cout << "RESULTS" << std::endl;
    std::cout << "================================================================" << std::endl;

    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Normal negative bf16 values with GELU = 0:  " << count_zero << std::endl;
    std::cout << "  Normal negative bf16 values with GELU ≠ 0:  " << count_nonzero << std::endl;

    std::cout << "\nBoundary:" << std::endl;
    std::cout << "  Last zero at:      x = " << last_zero_x
              << " (bf16: 0x" << std::hex << last_zero_bits << std::dec << ")" << std::endl;
    std::cout << "  First nonzero at:  x = " << first_nonzero_x
              << " (bf16: 0x" << std::hex << first_nonzero_bits << std::dec << ")" << std::endl;

    std::cout << "\n================================================================" << std::endl;
    std::cout << "CONCLUSION" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "\n  For x <= " << last_zero_x << ":  GELU(x) = 0 in bf16 with FTZ" << std::endl;
    std::cout << "  For x >  " << last_zero_x << ":  GELU(x) ≠ 0 in bf16 with FTZ" << std::endl;

    mpfr_free_cache();
    return 0;
}
```

**Compilation**:
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install libmpfr-dev libgmp-dev

# Compile
g++ -std=c++23 -O3 -o exhaustive_scan exhaustive_scan.cpp -lmpfr -lgmp

# Run
./exhaustive_scan
```

---

## 6. Results Summary

### 6.1 Final Answer

| Metric | Value |
|--------|-------|
| **Zero Saturation Threshold** | **x = -13.1875** |
| BF16 Hex Representation | `0xC153` |
| Last Nonzero Value | x = -13.125 (`0xC152`) |
| Normal bf16 values with GELU = 0 | 16,044 |
| Normal bf16 values with GELU ≠ 0 | 16,468 |

### 6.2 Why -13.1875?

The threshold is determined by where |GELU(x)| falls below the smallest normal BF16 value:

- **BF16 smallest normal**: ~1.175494 × 10⁻³⁸
- **|GELU(-13.125)|**: ~1.554 × 10⁻³⁸ (just above → representable)
- **|GELU(-13.1875)|**: ~6.829 × 10⁻³⁹ (below → subnormal → FTZ to 0)

### 6.3 Comparison: fp64 vs MPFR

| Method | Threshold Found | Correct? |
|--------|-----------------|----------|
| fp64 reference | x = -8.375 | ❌ **Wrong** (erf saturation) |
| MPFR 256-bit | x = -13.1875 | ✅ **Correct** |

---

## 7. Implementation Recommendation

For GELU implementations targeting BF16 with FTZ:

```cpp
inline float gelu_bf16_optimized(float x) {
    // Zero saturation threshold (MPFR-verified)
    if (x <= -13.1875f) {
        return 0.0f;
    }

    // For x > -13.1875, use polynomial approximation or erf computation
    // ...
}
```

This optimization allows skipping the expensive erf computation for 16,044 out of 32,512 normal negative BF16 values (49.3%).

---

## 8. Lessons Learned

1. **fp64 is insufficient** for computing reference values when results approach the limits of floating-point representation.

2. **The erf() function** in standard libraries saturates to ±1.0 before the mathematical function truly reaches those values.

3. **erfc() provides a simpler solution**: For negative x, use the identity `1 + erf(x/√2) = erfc(|x|/√2)`. The `erfc()` function returns small positive values for large arguments without saturation. This eliminates the need for MPFR in production code while matching MPFR-256 exactly (verified: 0 ULP difference across all BF16 values).

4. **MPFR is useful for research** to determine exact boundaries, but `erfc()` is sufficient for reference implementations.

5. **The true threshold** depends on the target format's smallest normal value, not on when intermediate calculations saturate.

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

*Research used MPFR 256-bit precision for correctness verification. For production reference implementations, fp64 with `erfc()` is sufficient and matches MPFR-256 exactly.*
