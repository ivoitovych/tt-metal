# GELU Implementation Reference for Tenstorrent Hardware

This document provides a comprehensive reference to the GELU activation function implementation in tt-metal, including file locations, code paths, mathematical formulas, and known precision characteristics.

## Mathematical Definition

**Exact GELU:**
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

**Tanh Approximation:**
```
GELU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

**Derivative (Backward):**
```
GELU'(x) = CDF(x) + x * PDF(x)
where:
  CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
  PDF(x) = exp(-0.5 * x^2) / sqrt(2 * pi)
```

---

## Implementation File Locations

### Core SFPU Kernel (Hardware Level)

| Architecture | File Path | Key Lines |
|--------------|-----------|-----------|
| Wormhole B0 (Metal) | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | 33-89 |
| Wormhole B0 (TT-LLK) | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h` | 17-213 |
| Blackhole (Metal) | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h` | 33-89 |
| Blackhole (TT-LLK) | `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h` | 17-213 |

### Supporting Functions

| Function | File Path | Key Lines |
|----------|-----------|-----------|
| CDF Approximation | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_cdf.h` | 15-72 |
| ERF Approximation | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erf_erfc.h` | 19-32 |

### TTNN Operation Interface

| Component | File Path | Key Lines |
|-----------|-----------|-----------|
| Operation Type Enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | 23 |
| Compute API | `tt_metal/include/compute_kernel_api/eltwise_unary/gelu.h` | 18-41 |

### Backward Pass Kernels

| Mode | File Path |
|------|-----------|
| Exact (erf-based) | `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_none.cpp` |
| Approximate (tanh-based) | `ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/kernels/compute/eltwise_bw_gelu_approx_tanh.cpp` |

---

## Implementation Modes

The GELU implementation supports two modes controlled by the `APPROXIMATION_MODE` template parameter:

### Mode 1: Fast Approximation (`APPROXIMATION_MODE=true`)

Uses a 6-piece piecewise linear LUT (Look-Up Table) for maximum speed.

**Formula:** `GELU_fast(x) = 0.5*x + LUT(x)`

**LUT Coefficients** (from `_init_gelu_()` in `ckernel_sfpu_gelu.h:183-213`):
```cpp
// LUT segments: slope (A) and intercept (B) pairs
// x in [0.0, 0.5): A=0.1928, B=-0.0150
// x in [0.5, 1.0): A=0.4939, B=-0.1605
// x in [1.0, 1.5): A=0.6189, B=-0.2797
// x in [1.5, 2.0): A=0.6099, B=-0.2635
// x in [2.0, 3.0): A=0.5402, B=-0.1194
// x >= 3.0:        A=0.50,   B=0.0
```

**Code Path:**
```
gelu_tile(fast_and_approx=true)
  -> calculate_gelu<true>()
    -> _calculate_gelu_<true>()
      -> _calculate_gelu_appx_<ITERATIONS>()
        -> lut2_sign() for 6-piece LUT lookup
```

### Mode 2: Accurate Mode (`APPROXIMATION_MODE=false`)

Uses polynomial approximations for higher accuracy.

**Implementation:** Uses CDF approximation with 5th-degree polynomial.

**Code Path (TT-LLK):**
```
gelu_tile(fast_and_approx=false)
  -> calculate_gelu<false>()
    -> _calculate_gelu_<false>()
      -> _calculate_gelu_accurate_<ITERATIONS>()
        -> _calculate_cdf_appx_(in, scaled=true)
          -> _calculate_pos_cdf_appx_(val)
```

**Code Path (Metal - Chebyshev):**
```
gelu_tile(fast_and_approx=false)
  -> calculate_gelu<false>()
    -> calculate_gelu_chebyshev(in)  // 15th-degree polynomial
```

---

## Core Implementation Code

### Chebyshev Polynomial (15th degree) - Metal Implementation

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h:33-61`

```cpp
inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    v_if(val >= -5.5f) {
        result = POLYVAL15(
            -1.81205228163e-09,
            -4.59055119276e-08,
            -3.74540617693e-07,
            -2.29754133825e-07,
            1.19076782913e-05,
            4.25116466215e-05,
            -0.000138391838381,
            -0.000862052441087,
            0.000768340223025,
            0.0092074331601,
            -0.00208478037614,
            -0.0656369476513,
            0.00244542739174,
            0.398579460781,
            0.499174645395,
            2.98325768482e-05,  // <-- NOTE: This is the floor value!
            val);
        result = setsgn(result, val);
    }
    v_endif;
    return result;
}
```

**Key Observation:** The constant term `2.98325768482e-05` in the Chebyshev polynomial corresponds to the floor value `0x37F9` discovered in precision analysis. For inputs close to zero, the polynomial evaluates to approximately this constant.

### CDF Approximation - TT-LLK Implementation

**File:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_cdf.h:15-72`

```cpp
inline sfpi::vFloat _calculate_pos_cdf_appx_(sfpi::vFloat val) {
    // Polynomial coefficients for x in [0, 2.5):
    // [0.0122792, -0.05281024, -0.03048313, 0.41314081, 0.49866379]
    sfpi::vFloat result;
    v_if (val < 2.5f) {
        result = POLYVAL5(0.0122792f, -0.05281024f, -0.03048313f,
                          0.41314081f, 0.49866379f, val);
    }
    v_else {
        // Linear approximation for x >= 2.5
        result = 0.44656975f * val + 0.58216001f;
    }
    v_endif;

    // Clamp to [0, 1]
    v_if (result > 1.0f) { result = 1.0f; }
    v_endif;
    return result;
}

inline sfpi::vFloat _calculate_cdf_appx_(sfpi::vFloat val, bool scaled = false) {
    sfpi::vFloat result = 0.0f;
    v_if (val < 0.0f) {
        result = 1.0f - _calculate_pos_cdf_appx_(-val);
    }
    v_else {
        result = _calculate_pos_cdf_appx_(val);
    }
    v_endif;

    if (scaled) {
        result *= val;  // GELU = x * CDF(x/sqrt(2))
    }
    return result;
}
```

### ERF Approximation

**File:** `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erf_erfc.h:19-32`

```cpp
template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_erf_body(vFloat x) {
    // Piecewise polynomial approximation for erf(x), x >= 0
    vFloat result = 1.0f;
    v_if(x >= 3.0f) {
        result = 1.0f;  // erf saturates to 1
    }
    v_elseif(x >= 1.0f) {
        // 5th-degree polynomial for x in [1, 3)
        result = POLYVAL5(-0.03170029f, 0.31310241f, -1.1603072f,
                          1.91684792f, -0.19469693f, x);
    }
    v_elseif(x >= 0.0f) {
        // 5th-degree polynomial for x in [0, 1)
        result = POLYVAL5(0.166342190f, -0.476685015f, 0.0275416549,
                          1.12544048f, 0.0000661338118f, x);
    }
    v_else {
        result = 0.0f;
    }
    v_endif;
    return result;
}
```

### Fast LUT Approximation

**File:** `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_gelu.h:38-84`

```cpp
template <int ITERATIONS>
inline void _calculate_gelu_appx_() {
    // Load LUT coefficients from L-registers
    sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vUInt l2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vUInt l4 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vUInt l5 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vUInt l6 = sfpi::l_reg[sfpi::LRegs::LReg6];

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in      = sfpi::dst_reg[0];
        sfpi::vFloat half    = sfpi::vConstFloatPrgm0;  // 0.5
        sfpi::vFloat half_in = in * half;

        // 6-piece piecewise linear LUT with sign handling
        sfpi::vFloat result  = lut2_sign(in, l0, l1, l2, l4, l5, l6);
        result               = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}
```

---

## Known Precision Characteristics

### Floor Value Bug (Tiny Inputs)

**Discovery:** For tiny normal inputs (~1e-38 to ~1e-10), GELU outputs a constant floor value of `2.980232e-05` (bf16 bits: `0x37F9`).

| Input Region | Expected Behavior | Actual Behavior | Max ULP Error |
|--------------|-------------------|-----------------|---------------|
| Tiny normal (1e-38 to 1e-10) | GELU(x) ≈ 0.5*x | Constant 2.98e-05 | 14,266 |
| GELU active region (-2 to 2) | Standard GELU curve | Excellent precision | 0-4 |
| Large positive (>3) | GELU(x) ≈ x | Correct | 0-2 |
| Large negative (<-3) | GELU(x) ≈ 0 | Correct | 0-2 |

**Root Cause:** The Chebyshev polynomial's constant term `2.98325768482e-05` dominates for near-zero inputs because the higher-order terms become negligible. This creates a floor effect where the output cannot go below this minimum value.

**Impact:** For BERT and other transformer models, this is generally acceptable because:
1. Values in the tiny normal range rarely occur in practice
2. The GELU active region (-2 to 2) has excellent precision (0-4 ULP)
3. Post-LayerNorm activations typically have mean~0, stddev~1

See `gelu_precision_analysis.md` for detailed analysis and visualization.

---

## Usage in tt-train

### Forward Pass (from tests)

```cpp
// Reference implementation for comparison
float reference_gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

// Hardware operation via TTNN
auto output = ttml::ops::gelu(input);
```

### Backward Pass (from tests)

```cpp
// Reference implementation
float reference_gelu_backward(float x) {
    float cdf = 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
    float pdf = std::exp(-0.5f * x * x) / std::sqrt(2.0f * M_PI);
    return cdf + x * pdf;
}

// Hardware operation via TTNN
auto grad_input = ttml::ops::gelu_bw(grad_output, input);
```

---

## Related Files

| File | Description |
|------|-------------|
| `tt-train/tests/ops/gelu_op_test.cpp` | GELU test suite with ULP validation |
| `tt-train/tests/ops/gelu_precision_analysis.md` | Detailed precision analysis |
| `tt-train/tests/core/bf16_ulp.hpp` | BFloat16 ULP distance calculator |
| `gelu_ulp_plot_wh_n150_bf16_ulp_module.png` | ULP error visualization |

---

## Version Information

- **Hardware:** Wormhole n150 L
- **Branch:** `ivoitovych/bert-model-for-ttml-pr-gelu-test-suite-amendment-ulp-diagnostic-04`
- **Date:** 2025-12-27
- **tt-metal commit:** See current HEAD
