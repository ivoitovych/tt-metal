# GELU Implementation Reference for Tenstorrent Hardware

This document provides a complete, fact-based reference to the GELU activation function implementation in tt-metal. All information is traceable to source code with exact file paths and line numbers.

**Document Version:** 2025-12-27
**Hardware:** Wormhole n150 L (Blackhole implementations are identical)
**Branch:** `ivoitovych/bert-model-for-ttml-pr-gelu-test-suite-amendment-ulp-diagnostic-04`

**Architecture Note:** Wormhole B0 and Blackhole share identical GELU implementations. All file paths shown are for Wormhole B0; Blackhole equivalents exist at the same relative paths under `blackhole/` instead of `wormhole_b0/`.

---

## Table of Contents

1. [Summary: What tt-train Actually Uses](#1-summary-what-tt-train-actually-uses)
2. [Implementation Count](#2-implementation-count)
3. [Forward Pass: Accurate Mode (Chebyshev)](#3-forward-pass-accurate-mode-chebyshev)
4. [Forward Pass: Fast Mode (LUT)](#4-forward-pass-fast-mode-lut)
5. [Backward Pass: Exact Mode (erf-based)](#5-backward-pass-exact-mode-erf-based)
6. [Backward Pass: Approximate Mode (tanh-based)](#6-backward-pass-approximate-mode-tanh-based)
7. [Reproducible Reference Implementations](#7-reproducible-reference-implementations)
8. [Known Precision Characteristics](#8-known-precision-characteristics)
9. [File Reference Table](#9-file-reference-table)

---

## 1. Summary: What tt-train Actually Uses

**tt-train/TTML uses exactly TWO implementations:**

| Pass | Mode | Implementation | Default? |
|------|------|----------------|----------|
| Forward | Accurate | 15th-degree Chebyshev polynomial | **YES** |
| Backward | Exact | erf-based with exp() | **YES** |

**Evidence from source code:**

```cpp
// File: tt-train/sources/ttml/ops/unary_ops.cpp:37-49

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value()));  // Default: parameter=false
    autograd::GradFunction grad = [tensor, out]() {
        static const std::string approx_mode = "none";  // Uses exact erf-based mode
        auto dL_dt = ttnn::experimental::gelu_bw(out->get_grad(), tensor->get_value(), approx_mode);
        tensor->add_grad(dL_dt);
    };
    // ...
}
```

**Default parameter verification:**

```cpp
// File: ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp:26-33

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFastAndApproximateMode {
    static Tensor invoke(
        const Tensor& input_tensor,
        bool parameter = false,  // <-- DEFAULT IS FALSE (accurate mode)
        // ...
    );
};
```

---

## 2. Implementation Count

**Total: 4 distinct implementations**

| # | Pass | Mode Name | Algorithm | Used by tt-train? |
|---|------|-----------|-----------|-------------------|
| 1 | Forward | Accurate | Chebyshev polynomial (15th degree) | **YES** |
| 2 | Forward | Fast | 6-piece piecewise linear LUT | No |
| 3 | Backward | Exact | erf() + exp() based | **YES** |
| 4 | Backward | Approximate | tanh() based | No |

**Note:** There is also a CDF-polynomial implementation in TT-LLK (`_calculate_gelu_accurate_()`) but it is NOT used - the Metal kernel overrides it with Chebyshev for accurate mode.

---

## 3. Forward Pass: Accurate Mode (Chebyshev)

**This is what tt-train uses for forward GELU.**

### Source File

`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h`

### Code Path

```
ttnn::gelu(tensor)                                    // unary.cpp:128-140
  -> UnaryWithParam{GELU, 0.0f}                       // parameter=false (accurate mode)
    -> gelu_tile<false>(idst)                         // gelu.h:38-41
      -> calculate_gelu<false, 8>()                   // ckernel_sfpu_gelu.h:73-89
        -> for each element in tile:
             if (in == 0.0f): result = 0.0f
             else if (in < 3.0f): result = calculate_gelu_chebyshev(in)
             else: result = in  // identity for x >= 3.0
```

### Complete Algorithm (lines 73-89)

```cpp
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        _calculate_gelu_<APPROXIMATION_MODE, ITERATIONS>();
    } else {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat result = in;
            v_if(in == 0.0f) { result = 0.0f; }
            v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); }
            v_endif;
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}
```

### Chebyshev Polynomial (lines 33-61)

```cpp
inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {
        result = POLYVAL15(
            -1.81205228163e-09,   // c15
            -4.59055119276e-08,   // c14
            -3.74540617693e-07,   // c13
            -2.29754133825e-07,   // c12
            1.19076782913e-05,    // c11
            4.25116466215e-05,    // c10
            -0.000138391838381,   // c9
            -0.000862052441087,   // c8
            0.000768340223025,    // c7
            0.0092074331601,      // c6
            -0.00208478037614,    // c5
            -0.0656369476513,     // c4
            0.00244542739174,     // c3
            0.398579460781,       // c2
            0.499174645395,       // c1
            2.98325768482e-05,    // c0 <-- Floor value for tiny inputs
            val);
        result = setsgn(result, val);  // Preserve sign of input
    }
    v_endif;
    return result;
}
```

### POLYVAL15 Macro (Horner's Method)

```cpp
// File: ckernel_sfpu_gelu.h:13-31
#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x) \
    (((((((((((((((c15) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) \
    * (x) + (c10)) * (x) + (c9)) * (x) + (c8)) * (x) + (c7)) * (x) + (c6)) \
    * (x) + (c5)) * (x) + (c4)) * (x) + (c3)) * (x) + (c2)) * (x) + (c1)) * (x) + (c0))
```

### Algorithm Summary

For input `x`:
1. If `x == 0.0`: return `0.0`
2. If `x >= 3.0`: return `x` (GELU saturates to identity)
3. If `x < -5.5`: return `0.0` (GELU saturates to zero)
4. Otherwise: return `sign(x) * POLYVAL15(coefficients, x)`

---

## 4. Forward Pass: Fast Mode (LUT)

**NOT used by tt-train (but available via `ttnn::gelu(tensor, true)`).**

### Source File

`tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h`

### Code Path

```
ttnn::gelu(tensor, true)                              // fast_and_approximate=true
  -> UnaryWithParam{GELU, 1.0f}                       // parameter=true
    -> gelu_tile<true>(idst)                          // gelu.h:38-41
      -> calculate_gelu<true, 8>()                    // metal ckernel_sfpu_gelu.h:75-76
        -> _calculate_gelu_<true, 8>()                // tt_llk ckernel_sfpu_gelu.h:100-111
          -> _calculate_gelu_appx_<8>()               // tt_llk ckernel_sfpu_gelu.h:38-84
```

### Algorithm (lines 38-84)

```cpp
template <int ITERATIONS>
inline void _calculate_gelu_appx_() {
    // Load LUT coefficients from L-registers (initialized by _init_gelu_())
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    // ... l1, l2, l4, l5, l6

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in      = sfpi::dst_reg[0];
        sfpi::vFloat half    = sfpi::vConstFloatPrgm0;  // 0.5
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result  = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result               = half_in + result;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

### LUT Coefficients (lines 182-213)

The 6-piece piecewise linear LUT is initialized with these coefficients:

| Segment (|x|) | Slope (A) | Intercept (B) | Hex (A) | Hex (B) |
|---------------|-----------|---------------|---------|---------|
| [0.0, 0.5) | 0.1928 | -0.000104* | 0x322B | 0x86D8 |
| [0.5, 1.0) | 0.4939 | -0.1605 | 0x37E7 | 0xB122 |
| [1.0, 1.5) | 0.6189 | -0.2797 | 0x38F3 | 0xB479 |
| [1.5, 2.0) | 0.6099 | -0.2635 | 0x38E1 | 0xB437 |
| [2.0, 3.0) | 0.5402 | -0.1194 | 0x3852 | 0xAFA4 |
| [3.0, inf) | 0.5000 | 0.0 | 0x3800 | 0x7C00** |

*Note: Source code comment claims B=-0.0150 (hex 0xA3AE) for segment [0.0, 0.5), but the actual loaded value is 0x86D8 ≈ -0.000104. This table reports what the code actually loads.

**Note: 0x7C00 is +inf in IEEE 754 half-precision; hardware may interpret this specially.

### Formula

```
GELU_fast(x) = 0.5 * x + lut2_sign(x)

where lut2_sign(x) = sign(x) * (A[segment] * |x| + B[segment])
```

---

## 5. Backward Pass: Exact Mode (erf-based)

**This is what tt-train uses for backward GELU.**

### Source File

`ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp`

### Mathematical Formula

```
GELU'(x) = CDF(x) + x * PDF(x)

where:
  CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
  PDF(x) = exp(-0.5 * x^2) / sqrt(2 * pi)

grad_input = grad_output * GELU'(x)
```

### Constants (lines 30-31)

```cpp
constexpr float kAlpha = 0.70710678118654752440f;  // 1 / sqrt(2)
constexpr float kBeta = 0.3989422804014327f;       // 1 / sqrt(2 * pi)
```

### Algorithm (lines 50-91)

```cpp
// Step 1: CDF term = 0.5 * (1 + erf(x / sqrt(2)))
tile[1] = x * kAlpha;              // x / sqrt(2)
tile[1] = erf(tile[1]);            // erf(x / sqrt(2))
tile[1] = (tile[1] + 1.0) * 0.5;   // 0.5 * (1 + erf(...))

// Step 2: PDF term = x * (1 / sqrt(2*pi)) * exp(-x^2 / 2)
tile[2] = x^2;                     // square
tile[2] = tile[2] * -0.5;          // -0.5 * x^2
tile[2] = exp(tile[2]);            // exp(-0.5 * x^2)
tile[2] = tile[2] * kBeta;         // * (1 / sqrt(2*pi))
tile[2] = tile[2] * x;             // * x

// Step 3: Result = grad * (CDF + PDF)
result = grad * (tile[1] + tile[2]);
```

---

## 6. Backward Pass: Approximate Mode (tanh-based)

**NOT used by tt-train (but available via `approx_mode="tanh"`).**

### Source File

`ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp`

### Mathematical Formula

Uses the tanh approximation of GELU:
```
GELU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

The derivative is:
```
GELU'_approx(x) = CDF_term + x * PDF_term

where:
  beta = sqrt(2/pi) = 0.7978845608...
  kappa = 0.044715
  inner = beta * (x + kappa * x^3)

  CDF_term = 0.5 * (1 + tanh(inner))
  PDF_term = 0.5 * beta * (1 + 3*kappa*x^2) * (1 - tanh^2(inner))
```

### Constants (lines 19-32)

```cpp
#define M_SQRT2 1.41421356237309504880f    // sqrt(2)
#define M_2_SQRTPI 1.12837916709551257390f // 2/sqrt(pi)

constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;  // sqrt(2/pi)
constexpr float kKappa = 0.044715;
```

---

## 7. Reproducible Reference Implementations

### Forward GELU (Accurate Mode) - Python/NumPy

```python
import numpy as np

# Chebyshev coefficients from ckernel_sfpu_gelu.h:36-52
CHEBYSHEV_COEFFS = [
    -1.81205228163e-09,   # c15
    -4.59055119276e-08,   # c14
    -3.74540617693e-07,   # c13
    -2.29754133825e-07,   # c12
    1.19076782913e-05,    # c11
    4.25116466215e-05,    # c10
    -0.000138391838381,   # c9
    -0.000862052441087,   # c8
    0.000768340223025,    # c7
    0.0092074331601,      # c6
    -0.00208478037614,    # c5
    -0.0656369476513,     # c4
    0.00244542739174,     # c3
    0.398579460781,       # c2
    0.499174645395,       # c1
    2.98325768482e-05,    # c0
]

def gelu_chebyshev(x: np.ndarray) -> np.ndarray:
    """
    Reproduces the Tenstorrent hardware GELU (accurate mode).

    Algorithm from: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
    """
    result = np.zeros_like(x)

    # x == 0: result = 0 (already initialized)

    # x >= 3.0: result = x (identity)
    mask_large = x >= 3.0
    result[mask_large] = x[mask_large]

    # -5.5 <= x < 3.0: use Chebyshev polynomial
    mask_poly = (x >= -5.5) & (x < 3.0) & (x != 0.0)
    x_poly = x[mask_poly]

    # Horner's method for polynomial evaluation
    poly_result = np.zeros_like(x_poly)
    for coeff in CHEBYSHEV_COEFFS:
        poly_result = poly_result * x_poly + coeff

    # Apply sign correction: result = sign(x) * |poly_result|
    result[mask_poly] = np.sign(x_poly) * np.abs(poly_result)

    # x < -5.5: result = 0 (already initialized)

    return result

def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Standard GELU formula for comparison."""
    return 0.5 * x * (1 + np.erf(x / np.sqrt(2)))
```

### Forward GELU (Fast Mode) - Python/NumPy

```python
import numpy as np

# LUT coefficients from ckernel_sfpu_gelu.h:205-212
# Note: First segment intercept is 0x86D8 ≈ -0.000104 (not -0.0150 as comment claims)
LUT_SEGMENTS = [
    # (max_x, slope_A, intercept_B)
    (0.5, 0.1928, -0.000104),  # B = 0x86D8 (actual loaded value)
    (1.0, 0.4939, -0.1605),
    (1.5, 0.6189, -0.2797),
    (2.0, 0.6099, -0.2635),
    (3.0, 0.5402, -0.1194),
    (np.inf, 0.5000, 0.0),
]

def gelu_lut(x: np.ndarray) -> np.ndarray:
    """
    Reproduces the Tenstorrent hardware GELU (fast/LUT mode).

    Algorithm from: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
    Formula: GELU(x) = 0.5*x + sign(x) * (A*|x| + B)
    """
    abs_x = np.abs(x)
    sign_x = np.sign(x)

    # Initialize with last segment values
    A = np.full_like(x, 0.5)
    B = np.full_like(x, 0.0)

    # Apply segments in reverse order (so smaller thresholds override)
    for max_val, slope, intercept in reversed(LUT_SEGMENTS):
        mask = abs_x < max_val
        A[mask] = slope
        B[mask] = intercept

    # lut2_sign(x) = sign(x) * (A * |x| + B)
    lut_result = sign_x * (A * abs_x + B)

    # result = 0.5 * x + lut_result
    return 0.5 * x + lut_result
```

### Backward GELU (Exact Mode) - Python/NumPy

```python
import numpy as np

def gelu_backward_exact(grad_output: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Reproduces the Tenstorrent hardware GELU backward (exact/erf mode).

    Algorithm from: ttnn/.../eltwise_bw_gelu_approx_none.cpp
    """
    kAlpha = 0.70710678118654752440  # 1 / sqrt(2)
    kBeta = 0.3989422804014327       # 1 / sqrt(2 * pi)

    # CDF term: 0.5 * (1 + erf(x / sqrt(2)))
    cdf_term = 0.5 * (1 + np.erf(x * kAlpha))

    # PDF term: x * (1 / sqrt(2*pi)) * exp(-x^2 / 2)
    pdf_term = x * kBeta * np.exp(-0.5 * x * x)

    # grad_input = grad_output * (CDF + PDF)
    return grad_output * (cdf_term + pdf_term)
```

### Forward GELU (Accurate Mode) - C++ Reference

```cpp
#include <cmath>
#include <vector>

// Chebyshev coefficients from ckernel_sfpu_gelu.h:36-52
constexpr float CHEBYSHEV_COEFFS[16] = {
    -1.81205228163e-09f,   // c15
    -4.59055119276e-08f,   // c14
    -3.74540617693e-07f,   // c13
    -2.29754133825e-07f,   // c12
    1.19076782913e-05f,    // c11
    4.25116466215e-05f,    // c10
    -0.000138391838381f,   // c9
    -0.000862052441087f,   // c8
    0.000768340223025f,    // c7
    0.0092074331601f,      // c6
    -0.00208478037614f,    // c5
    -0.0656369476513f,     // c4
    0.00244542739174f,     // c3
    0.398579460781f,       // c2
    0.499174645395f,       // c1
    2.98325768482e-05f,    // c0
};

// Horner's method polynomial evaluation (matches POLYVAL15 macro)
inline float polyval15(float x) {
    float result = 0.0f;
    for (int i = 0; i < 16; ++i) {
        result = result * x + CHEBYSHEV_COEFFS[i];
    }
    return result;
}

/**
 * Reproduces the Tenstorrent hardware GELU (accurate mode).
 *
 * Algorithm from: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
 */
float gelu_chebyshev(float x) {
    if (x == 0.0f) {
        return 0.0f;
    }
    if (x >= 3.0f) {
        return x;  // Identity for large positive values
    }
    if (x < -5.5f) {
        return 0.0f;  // Zero for large negative values
    }
    // Apply Chebyshev polynomial with sign correction
    float poly = polyval15(x);
    return std::copysign(std::abs(poly), x);
}

// Exact GELU for comparison
float gelu_exact(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}
```

### Forward GELU (Fast Mode) - C++ Reference

```cpp
#include <cmath>

// LUT coefficients from ckernel_sfpu_gelu.h:205-212
// Format: {max_abs_x, slope_A, intercept_B}
// Note: First segment intercept is 0x86D8 ≈ -0.000104 (not -0.0150 as source comment claims)
constexpr float LUT_SEGMENTS[6][3] = {
    {0.5f, 0.1928f, -0.000104f},  // B = 0x86D8 (actual loaded value)
    {1.0f, 0.4939f, -0.1605f},
    {1.5f, 0.6189f, -0.2797f},
    {2.0f, 0.6099f, -0.2635f},
    {3.0f, 0.5402f, -0.1194f},
    {INFINITY, 0.5000f, 0.0f},
};

/**
 * Reproduces the Tenstorrent hardware GELU (fast/LUT mode).
 *
 * Algorithm from: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
 * Formula: GELU(x) = 0.5*x + sign(x) * (A*|x| + B)
 */
float gelu_lut(float x) {
    float abs_x = std::abs(x);
    float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;

    // Find appropriate LUT segment
    float A = 0.5f, B = 0.0f;
    for (const auto& seg : LUT_SEGMENTS) {
        if (abs_x < seg[0]) {
            A = seg[1];
            B = seg[2];
            break;
        }
    }

    // lut2_sign(x) = sign(x) * (A * |x| + B)
    float lut_result = sign_x * (A * abs_x + B);

    // result = 0.5 * x + lut_result
    return 0.5f * x + lut_result;
}
```

### Backward GELU (Exact Mode) - C++ Reference

```cpp
#include <cmath>

/**
 * Reproduces the Tenstorrent hardware GELU backward (exact/erf mode).
 *
 * Algorithm from: ttnn/.../eltwise_bw_gelu_approx_none.cpp
 *
 * @param grad_output Upstream gradient (dL/dy)
 * @param x           Original input to forward GELU
 * @return            Input gradient (dL/dx)
 */
float gelu_backward_exact(float grad_output, float x) {
    constexpr float kAlpha = 0.70710678118654752440f;  // 1 / sqrt(2)
    constexpr float kBeta = 0.3989422804014327f;       // 1 / sqrt(2 * pi)

    // CDF term: 0.5 * (1 + erf(x / sqrt(2)))
    float cdf_term = 0.5f * (1.0f + std::erf(x * kAlpha));

    // PDF term: x * (1 / sqrt(2*pi)) * exp(-x^2 / 2)
    float pdf_term = x * kBeta * std::exp(-0.5f * x * x);

    // grad_input = grad_output * (CDF + PDF)
    return grad_output * (cdf_term + pdf_term);
}
```

### Backward GELU (Approximate Mode) - C++ Reference

```cpp
#include <cmath>

/**
 * Reproduces the Tenstorrent hardware GELU backward (approximate/tanh mode).
 *
 * Algorithm from: ttnn/.../eltwise_bw_gelu_approx_tanh.cpp
 *
 * NOT used by tt-train, provided for completeness.
 */
float gelu_backward_tanh(float grad_output, float x) {
    constexpr float kBeta = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float kKappa = 0.044715f;

    // inner = beta * (x + kappa * x^3)
    float x_cubed = x * x * x;
    float inner = kBeta * (x + kKappa * x_cubed);

    // tanh_inner = tanh(inner)
    float tanh_inner = std::tanh(inner);

    // CDF_term = 0.5 * (1 + tanh(inner))
    float cdf_term = 0.5f * (1.0f + tanh_inner);

    // sech^2(inner) = 1 - tanh^2(inner)
    float sech_sq = 1.0f - tanh_inner * tanh_inner;

    // PDF_term = 0.5 * beta * (1 + 3*kappa*x^2) * sech^2(inner)
    float pdf_term = 0.5f * kBeta * (1.0f + 3.0f * kKappa * x * x) * sech_sq;

    // grad_input = grad_output * (CDF_term + x * PDF_term)
    return grad_output * (cdf_term + x * pdf_term);
}
```

---

## 8. Known Precision Characteristics

### Floor Value Bug (Accurate Mode)

For tiny positive inputs (~1e-38 to ~1e-10), the Chebyshev polynomial produces a constant floor value instead of the expected near-linear response.

| Input Region | Expected | Actual | Root Cause |
|--------------|----------|--------|------------|
| 1e-38 to 1e-10 | GELU(x) ≈ 0.5*x | Constant 2.98e-05 | Polynomial c0 term dominates |
| -2 to 2 (active region) | Standard GELU | Excellent (0-4 ULP) | Polynomial well-fitted |
| > 3 | GELU(x) = x | Exact | Uses identity |
| < -3 | GELU(x) = 0 | Correct | Sign correction |

**Root Cause:** The constant term c0 = 2.98325768482e-05 dominates when x is tiny because all higher-order terms (c1*x, c2*x^2, ...) become negligible.

**BFloat16 representation:** 0x37F9 = 2.980232e-05

See `gelu_precision_analysis.md` for detailed ULP analysis.

---

## 9. File Reference Table

### Forward Pass Implementation Files

| File | Lines | Description |
|------|-------|-------------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | 33-89 | Metal GELU: Chebyshev + dispatcher |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h` | 38-111, 182-213 | TT-LLK GELU: LUT approx + init |
| `tt_metal/include/compute_kernel_api/eltwise_unary/gelu.h` | 18-41 | Compute API: gelu_tile() |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` | 127-145 | TTNN operation wrapper |
| `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | 26-33 | Default parameter (false) |

### Backward Pass Implementation Files

| File | Lines | Description |
|------|-------|-------------|
| `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp` | 30-91 | Exact erf-based backward |
| `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp` | 19-117 | Approximate tanh-based backward |

### tt-train Usage

| File | Lines | Description |
|------|-------|-------------|
| `tt-train/sources/ttml/ops/unary_ops.cpp` | 37-49 | GELU forward + backward call |

### Related Files

| File | Description |
|------|-------------|
| `tt-train/tests/ops/gelu_op_test.cpp` | GELU test suite |
| `tt-train/tests/ops/gelu_precision_analysis.md` | ULP precision analysis |
| `tt-train/tests/core/bf16_ulp.hpp` | BFloat16 ULP calculator |

---

## Appendix: Mathematical Definitions

### Exact GELU (erf-based)

```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

where erf(x) = (2/sqrt(pi)) * integral(0, x, exp(-t^2) dt)
```

### GELU Derivative

```
GELU'(x) = CDF(x) + x * PDF(x)

where:
  CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))     [Cumulative distribution function]
  PDF(x) = exp(-0.5 * x^2) / sqrt(2 * pi)   [Probability density function]
```

### Tanh Approximation (not used by tt-train)

```
GELU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```
