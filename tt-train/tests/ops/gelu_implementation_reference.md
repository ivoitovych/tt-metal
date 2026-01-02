# GELU Implementation Reference for Tenstorrent Hardware

This document provides a complete, fact-based reference to the GELU activation function implementation in tt-metal. All information is traceable to source code with exact file paths and line numbers.

**Document Version:** 2025-12-28
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
10. [SFPU Operations Reference (Implementation Constraints)](#10-sfpu-operations-reference-implementation-constraints)

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

where lut2_sign(x) = A[segment] * |x| + B[segment]  (always positive due to SGN_UPDATE)
```

**Note on lut2_sign:** Unlike what the function name suggests, `lut2_sign()` does NOT multiply by `sign(x)`. It uses `SFPLUTFP32_MOD0_SGN_UPDATE` which means the sign comes from the computation result. Since A > 0 and |x| >= 0, and B is typically small, the result is always positive.

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

    IMPORTANT: lut2_sign() uses SFPLUTFP32_MOD0_SGN_UPDATE, meaning the sign
    comes from the computation result, NOT from the input. The LUT computes
    A*|x| + B which is always positive (for A>0, B>=0).

    Formula: GELU(x) = 0.5*x + (A*|x| + B)
    """
    abs_x = np.abs(x)

    # Initialize with last segment values
    A = np.full_like(x, 0.5)
    B = np.full_like(x, 0.0)

    # Apply segments in reverse order (so smaller thresholds override)
    for max_val, slope, intercept in reversed(LUT_SEGMENTS):
        mask = abs_x < max_val
        A[mask] = slope
        B[mask] = intercept

    # lut2_sign(x) = A * |x| + B  (always positive due to SGN_UPDATE)
    lut_result = A * abs_x + B

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
 *
 * IMPORTANT: lut2_sign() uses SFPLUTFP32_MOD0_SGN_UPDATE, meaning the sign
 * comes from the computation result, NOT from the input. The LUT computes
 * A*|x| + B which is always positive (for A>0, B>=0).
 *
 * Formula: GELU(x) = 0.5*x + (A*|x| + B)
 */
float gelu_lut(float x) {
    float abs_x = std::abs(x);

    // Find appropriate LUT segment
    float A = 0.5f, B = 0.0f;
    for (const auto& seg : LUT_SEGMENTS) {
        if (abs_x < seg[0]) {
            A = seg[1];
            B = seg[2];
            break;
        }
    }

    // lut2_sign(x) = A * |x| + B  (always positive due to SGN_UPDATE)
    float lut_result = A * abs_x + B;

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

### Large Input Value Behavior (Fast/LUT Mode)

**Correction (2025-12-30):** Hardware testing confirmed that the fast/LUT mode correctly handles large inputs. The previously documented "bug" was based on an incorrect understanding of the `lut2_sign` function.

| Input x | GELU_LUT (actual) | GELU_exact (expected) | Error |
|---------|-------------------|----------------------|-------|
| -10.0 | ~0.0 | ~0 | ~0 ✓ |
| -5.0 | ~0.0 | ~0 | ~0 ✓ |
| -3.0 | ~0.0 | -0.004 | ~0.004 ✓ |
| +3.0 | +3.0 | +2.996 | 0.004 ✓ |
| +10.0 | +10.0 | +10.0 | 0 ✓ |

**Correct Algorithm:** The `lut2_sign()` function uses `SFPLUTFP32_MOD0_SGN_UPDATE`, meaning the sign comes from the computation result, NOT from the input. For the last LUT segment (|x| >= 3.0), the coefficients are A=0.5, B=0.0. The formula:

```
lut2_sign(x) = A*|x| + B = 0.5*|x|  (always positive)
result = 0.5*x + lut2_sign(x) = 0.5*x + 0.5*|x|
```

For negative x (e.g., x=-10): `0.5*(-10) + 0.5*10 = -5 + 5 = 0` ✓
For positive x (e.g., x=+10): `0.5*(+10) + 0.5*10 = 5 + 5 = 10` ✓

This correctly implements the asymmetric GELU behavior (GELU → 0 for x → -∞, GELU → x for x → +∞).

**Minor Bug at x=0:** Fast mode returns ~-0.000104 for x=0 instead of exactly 0. This is due to the LUT intercept B=-0.000104 in the first segment [0, 0.5).

**Source:** `runtime/sfpi/include/wormhole/sfpi_lib.h` - `lut2_sign()` with `SFPLUTFP32_MOD0_SGN_UPDATE`

---

## 9. File Reference Table

### Two-Layer File Architecture

**Important:** There are two files named `ckernel_sfpu_gelu.h` with different content:

| Layer | Path | Size | MD5 |
|-------|------|------|-----|
| **Metal API** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | 96 lines | `381945844ca0...` |
| **TT-LLK Base** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h` | 265 lines | `689b032d7813...` |

**Relationship:**

```
tt_metal/include/compute_kernel_api/eltwise_unary/gelu.h
    │
    │  #include "ckernel_sfpu_gelu.h"
    ▼
Metal API layer: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
    │
    ├── #include "ckernel.h"  ───────────────────────────────────────────────┐
    ├── calculate_gelu_chebyshev()      ← Chebyshev polynomial (accurate)    │
    ├── calculate_gelu<APPROX_MODE>()   ← Dispatcher                         │
    │       ├── APPROX_MODE=true:  calls _calculate_gelu_<>()  ──────────────┼──┐
    │       └── APPROX_MODE=false: uses calculate_gelu_chebyshev()           │  │
    └── gelu_init() → calls _init_gelu_<>()  ────────────────────────────────┼──┤
                                                                             │  │
                                 ┌───────────────────────────────────────────┘  │
                                 ▼                                              │
TT-LLK Base layer: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h
    │
    ├── _init_gelu_()                    ← LUT coefficient initialization  ◄────┤
    ├── _calculate_gelu_appx_()          ← LUT implementation (fast mode)  ◄────┘
    ├── _calculate_gelu_accurate_()      ← CDF-based (DEAD CODE - Metal overrides)
    ├── _calculate_gelu_()               ← Internal dispatcher
    └── _calculate_gelu_derivative_()    ← Derivative implementations
```

**Key insight:** The Metal layer **overrides** the TT-LLK accurate mode with Chebyshev polynomial, but still uses TT-LLK's LUT for fast/approximate mode. The `_calculate_gelu_accurate_()` function in TT-LLK is effectively dead code.

### Complete Call Stack

This section traces the complete call path from `ttnn::gelu()` down to the kernel implementations.

#### TTNN Layer → Device Operation

```
ttnn::gelu(tensor, fast_and_approximate=false)   // Python/C++ API entry point
    │
    │  File: ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp:127-140
    ▼
ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::GELU>::invoke(
    tensor,
    parameter=false,  // default: accurate mode
    ...)
    │
    │  Creates: UnaryWithParam{UnaryOpType::GELU, 0.0f}  // 0.0f = accurate, 1.0f = fast
    ▼
detail::unary_impl() → prim::unary()
    │
    │  Generates compute kernel that calls:
    ▼
gelu_tile_init<fast_and_approx>()  // Once per tile batch
gelu_tile<fast_and_approx>(idst)   // Once per tile
```

#### Compute Kernel API → SFPU Macros

```
gelu_tile_init<APPROX>()
    │
    │  File: tt_metal/include/compute_kernel_api/eltwise_unary/gelu.h:18-21
    ▼
MATH(SFPU_INIT_KERNEL_CALL(gelu, sfpu::gelu_init, APPROX))
    │
    │  File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h:14-15
    │  Expands to:
    ▼
llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROX>(sfpu::gelu_init<APPROX>)
    │
    │  File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h:17-21
    ▼
_llk_math_eltwise_unary_sfpu_init_<SfpuType::gelu>()  // Hardware SFPU setup
sfpu::gelu_init<APPROX>()                              // Software initialization
```

```
gelu_tile<APPROX>(idst)
    │
    │  File: tt_metal/include/compute_kernel_api/eltwise_unary/gelu.h:38-41
    ▼
MATH(SFPU_UNARY_NO_PARAM_KERNEL(gelu, RC, APPROX, idst))
    │
    │  File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h:75-77
    │  Expands to:
    ▼
_llk_math_eltwise_unary_sfpu_params_<APPROX>(
    ckernel::sfpu::calculate_gelu<APPROX>,
    idst,
    (int)VectorMode::RC)
    │
    │  File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h:11-68
    │  Sets up DST registers, calls the SFPU function for each tile face (4 faces x 8 rows = 32 elements)
    ▼
ckernel::sfpu::calculate_gelu<APPROX>()
```

#### SFPU Function Dispatch (Metal API Layer)

```
sfpu::gelu_init<APPROX>()
    │
    │  File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h:63-66
    ▼
_init_gelu_<APPROX>()   // Defined in TT-LLK Base layer
                        // Loads LUT coefficients to L-registers
```

```
ckernel::sfpu::calculate_gelu<APPROX>()
    │
    │  File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h:73-89
    │
    ├── if APPROX == true (fast mode):
    │       │
    │       ▼
    │   _calculate_gelu_<true, ITERATIONS>()   // TT-LLK Base layer
    │       │
    │       │  File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h:100-111
    │       ▼
    │   _calculate_gelu_appx_<ITERATIONS>()    // LUT-based piecewise linear
    │       │
    │       │  File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h:38-84
    │       ▼
    │   result = 0.5*x + lut2_sign(x, l0..l6)  // Formula with bug for x<-3
    │
    └── if APPROX == false (accurate mode):
            │
            │  Metal layer handles this directly (does NOT call TT-LLK)
            ▼
        for each element in tile:
            if (x == 0.0f): result = 0.0f
            else if (x < 3.0f): result = calculate_gelu_chebyshev(x)
            else: result = x  // identity for x >= 3.0
                │
                │  File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h:33-61
                ▼
            calculate_gelu_chebyshev(x)
                │
                ▼
            POLYVAL15(c15..c0, x)  // 15th-degree Chebyshev polynomial
            result = setsgn(result, x)  // Preserve input sign
```

#### Dead Code Path (TT-LLK Accurate Mode)

The following function exists in TT-LLK but is **never called** because Metal API overrides accurate mode:

```
_calculate_gelu_accurate_()
    │
    │  File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h:118-180
    │
    │  DEAD CODE - Metal layer's calculate_gelu<false>() uses Chebyshev instead
    ▼
    [CDF-based polynomial implementation - never executed]
```

### Forward Pass Implementation Files

| File | Lines | Description |
|------|-------|-------------|
| `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | 33-89 | Metal API: Chebyshev polynomial + dispatcher |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h` | 38-111, 182-213 | TT-LLK Base: LUT approximation + init functions |
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

## 10. SFPU Operations Reference (Implementation Constraints)

This section documents the primitive operations available at the GELU implementation point (SFPU/SFPI layer). These operations define the building blocks for implementing alternative GELU algorithms.

### Source Files

| File | Description |
|------|-------------|
| `runtime/sfpi/include/sfpi.h` | Core SFPI C++ wrapper classes (vFloat, vInt, vUInt) |
| `runtime/sfpi/include/wormhole/sfpi_hw.h` | Hardware builtin definitions |
| `runtime/sfpi/include/wormhole/sfpi_lib.h` | Library functions (lut, exexp, setexp, etc.) |
| `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/*.h` | Higher-level SFPU operations |

### Vector Register Types

The SFPU operates on 32-element vectors (one per SFPU lane per tile row):

| Type | Description |
|------|-------------|
| `vFloat` | 32-bit floating-point vector register |
| `vInt` | 32-bit signed integer vector register |
| `vUInt` | 32-bit unsigned integer vector register |
| `dst_reg[]` | DST register array (8-16 tiles depending on FP16/FP32 mode) |
| `l_reg[]` | L-register array (8 registers for LUT coefficients, constants) |

### Primitive Arithmetic Operations

**vFloat Arithmetic** (`sfpi.h:344-397`):
```cpp
// Binary operators (return new vFloat)
vFloat operator+(const vFloat b) const;  // Addition
vFloat operator-(const vFloat b) const;  // Subtraction
vFloat operator*(const vFloat b) const;  // Multiplication
// NOTE: No native division - use sfpu_reciprocal() + multiply

// Compound assignment
vFloat operator+=(const vFloat);
vFloat operator-=(const vFloat);
vFloat operator*=(const vFloat);

// Unary operators
vFloat operator-() const;    // Negation
vFloat operator++();         // Increment
vFloat operator--();         // Decrement
```

**vInt Arithmetic** (`sfpi.h:460-548`):
```cpp
// Arithmetic
vInt operator+(const vInt) const;
vInt operator-(const vInt) const;
vInt operator+=(const vInt);
vInt operator-=(const vInt);

// Bitwise
vInt operator&(const vInt) const;   // AND
vInt operator|(const vInt) const;   // OR
vInt operator^(const vInt) const;   // XOR
vInt operator~() const;             // NOT
vInt operator<<(int) const;         // Left shift
```

### Comparison Operations

**vFloat Comparisons** (`sfpi.h:390-396`):
```cpp
// Returns __vCond for use with v_if/v_elseif
__vCond operator==(const float x) const;
__vCond operator!=(const float x) const;
__vCond operator<(const float x) const;
__vCond operator<=(const float x) const;
__vCond operator>(const float x) const;
__vCond operator>=(const float x) const;

// Also supports vFloat comparisons
__vCond operator<(const vFloat x) const;
// etc.
```

**vInt Comparisons** (`sfpi.h:520-548`):
```cpp
__vCond operator==(const vInt) const;
__vCond operator!=(const vInt) const;
__vCond operator<(const vInt) const;
__vCond operator<=(const vInt) const;
__vCond operator>(const vInt) const;
__vCond operator>=(const vInt) const;
```

### Predicated Execution (Vectorized Conditionals)

```cpp
// Usage pattern:
v_if (condition) {
    // Executed for lanes where condition is true
}
v_elseif (condition2) {
    // Executed for lanes where condition is false AND condition2 is true
}
v_else {
    // Executed for lanes where both conditions are false
}
v_endif;
```

**Hardware behavior:** Predication masks vector lanes; all lanes still execute the same instruction but results are only written for active lanes.

### Exponent/Mantissa Manipulation (`sfpi_lib.h`)

```cpp
// Extract exponent (debiased: returns actual exponent, not biased value)
vInt exexp(const vFloat v);

// Extract exponent (raw biased value)
vInt exexp_nodebias(const vFloat v);

// Extract mantissa (normalized with 8/9 leading bits)
vInt exman8(const vFloat v);
vInt exman9(const vFloat v);

// Set exponent (from immediate or vector)
vFloat setexp(const vFloat v, const uint32_t exp);
vFloat setexp(const vFloat v, const __vIntBase exp);

// Set mantissa
vFloat setman(const vFloat v, const uint32_t man);
vFloat setman(const vFloat v, const __vIntBase man);

// Add to exponent (multiply by 2^exp)
vFloat addexp(const vFloat in, const int32_t exp);

// Set sign
vType setsgn(const vType v, const int32_t sgn);
vType setsgn(const vType v, const vType sgn);
```

### LUT-Based Function Approximation

```cpp
// 3-entry LUT (sign retained)
vFloat lut(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2);

// 3-entry LUT (sign updated from LUT)
vFloat lut_sign(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2);

// 6-entry LUT variants (FP16 or FP32 coefficients)
vFloat lut2(const vFloat v, const vUInt l0, const vUInt l1, const vUInt l2,
            const vUInt b01, const vUInt b23, const vUInt b45, const int mode = 1);
vFloat lut2_sign(/* same params */);

// FP32 LUT with 3 entries
vFloat lut2(const vFloat v,
            const vFloat a0, const vFloat a1, const vFloat a2,
            const vFloat b0, const vFloat b1, const vFloat b2);
```

**LUT operation:** Selects coefficients based on |x| magnitude, computes `A*|x| + B` or similar.

### Other Utility Functions

```cpp
// Absolute value
vFloat abs(const vFloat v);
vInt abs(const vInt v);

// Leading zeros count
vInt lz(const vType v);
vInt lz_nosgn(const vType v);  // Ignores sign bit

// Shift operation
vUInt shft(const vUInt v, const vInt amt);
vUInt shft(const vUInt v, int amt);

// Type reinterpret (bit-preserving cast)
vType reinterpret<vType>(const __vBase v);

// Min/max (in-place swap: after call, dst = min, src = max)
void vec_min_max(__vBase& dst, __vBase& src);
void vec_swap(__vBase& dst, __vBase& src);
```

### Type Conversion Functions

```cpp
// Integer to float
vFloat int32_to_float(vInt in, int round_mode = 1);

// Float to half-precision
vUInt float_to_fp16a(vFloat in, int round_mode = 1);
vUInt float_to_fp16b(vFloat in, int round_mode = 1);

// Float to integer
vUInt float_to_uint8(vFloat in, int round_mode = 1);
vUInt float_to_int8(vFloat in, int round_mode = 1);
vUInt float_to_uint16(vFloat in, int round_mode = 1);
vUInt float_to_int16(vFloat in, int round_mode = 1);

// int16 conversion (used for floor/round operations)
int16 float_to_int16(vFloat in, int round_mode);
```

### Constants

```cpp
// Built-in constants
vFloat vConst0;      // 0.0f
vFloat vConst1;      // 1.0f
vFloat vConstNeg1;   // -1.0f

// Programmable constants (user-settable, persist across calls)
vFloat vConstFloatPrgm0;
vFloat vConstFloatPrgm1;
vFloat vConstFloatPrgm2;
```

### Higher-Level Composite Functions

These functions are built from primitives and available in `ckernel_sfpu_*.h`:

| Function | Location | Algorithm | Notes |
|----------|----------|-----------|-------|
| `_sfpu_exp_21f_()` | `ckernel_sfpu_exp.h` | Moroz et al. polynomial | ~5 ULP for FP32 |
| `_sfpu_exp_61f_()` | `ckernel_sfpu_exp.h` | 6th-degree polynomial | More accurate |
| `_sfpu_exp_f32_accurate_()` | `ckernel_sfpu_exp.h` | Cody-Waite + Taylor | <1 ULP for FP32 |
| `_sfpu_reciprocal_<N>()` | `ckernel_sfpu_recip.h` | Newton-Raphson (N iterations) | N=1 for BF16, N=2 for FP32 |
| `_sfpu_tanh_polynomial_()` | `ckernel_sfpu_tanh.h` | 6th-degree polynomial | For BF16 mode |
| `_sfpu_tanh_continued_fraction_()` | `ckernel_sfpu_tanh.h` | Lambert's continued fraction | For FP32 mode |
| `calculate_erf_body()` | `ckernel_sfpu_erf_erfc.h` | Piecewise polynomial | 5th-degree per segment |
| `sfpu_sinpi()` / `sfpu_tan()` | `ckernel_sfpu_trigonometry.h` | Polynomial approximations | Various degrees |

### Performance Considerations

**Relative Operation Costs (estimated, lower is faster):**

| Category | Operations | Relative Cost |
|----------|-----------|---------------|
| Single-cycle | `+`, `-`, `*`, abs, setsgn, exexp, setexp | 1x |
| LUT | lut(), lut2(), lut2_sign() | 1-2x |
| Predication | v_if/v_elseif/v_endif | 1x per block |
| Newton-Raphson 1-iter | reciprocal (BF16 precision) | 5-10x |
| Newton-Raphson 2-iter | reciprocal (FP32 precision) | 10-15x |
| exp() | exponential function | 15-25x |
| tanh() (polynomial) | hyperbolic tangent | 10-20x |
| erf() | error function | 20-30x |

**Key observations:**
1. **No native division** - must use reciprocal (Newton-Raphson) + multiply
2. **LUT operations are fast** - 6-piece piecewise linear is efficient
3. **Polynomial evaluation** uses Horner's method - O(n) multiply-adds for degree n
4. **Predication** is efficient but adds conditional overhead per block
5. **exp/tanh/erf are expensive** - composed from many primitives

### Implications for Alternative GELU Implementations

The current Chebyshev implementation uses:
- 15 multiply-add operations (Horner's method)
- 2 comparisons + 1 predicated block
- 1 setsgn operation

Potential alternatives:
1. **Lower-degree polynomial**: Fewer operations, lower precision
2. **Piecewise linear (LUT)**: Already implemented as fast mode
3. **Rational approximation (Padé)**: Requires reciprocal (expensive)
4. **Lookup table with interpolation**: Limited by LUT granularity

For GELU backward, the bottleneck is exp() and erf():
- Consider tanh-based approximation (uses cheaper tanh instead of erf+exp)
- Or pre-computed lookup tables for common input ranges

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
