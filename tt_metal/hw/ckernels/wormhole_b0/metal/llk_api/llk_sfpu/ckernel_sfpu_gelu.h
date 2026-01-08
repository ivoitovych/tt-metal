// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

// C6 Adaptive Polynomial GELU Implementation
// Achieves Max ULP = 46 across entire BF16 range (DAZ+FTZ model)
// Reference: https://github.com/ivoitovych/bf16_gelu_research

// Degree-4 polynomial evaluation: c0 + c1*u + c2*u² + c3*u³ + c4*u⁴
// Using Horner's method: ((((c4*u + c3)*u + c2)*u + c1)*u + c0)
#define POLY4(c0, c1, c2, c3, c4, u) ((((c4) * (u) + (c3)) * (u) + (c2)) * (u) + (c1)) * (u) + (c0)

// C6 segment structure: 16 segments covering [-13.5625, 3.0]
// Each segment: x_start, x_end, x_mid, x_scale, c0, c1, c2, c3, c4
// Polynomial is evaluated as: p((x - x_mid) / x_scale)

inline sfpi::vFloat calculate_gelu_c6(sfpi::vFloat val) {
    sfpi::vFloat result = 0.0f;
    sfpi::vFloat abs_val = sfpi::abs(val);

    // Constants
    constexpr float INV_SQRT_2PI = 0.3989422804f;

    // Region 1: Near-zero (|x| < 0.125) - Taylor series
    // GELU(x) ≈ x * (0.5 + x/√(2π)) = x * (0.5 + 0.3989*x)
    v_if(abs_val < 0.125f) { result = val * (0.5f + INV_SQRT_2PI * val); }
    // Region 2: Positive saturation (x >= 3.0)
    v_elseif(val >= 3.0f) { result = val; }
    // Region 3: Deep negative - asymptotic expansion for x < -5.5
    // GELU(x) ≈ -φ(x) where φ(x) = exp(-x²/2) / √(2π)
    //
    // Hardware limitations:
    // 1. exp() clamps input to [-88.5, 89]
    // 2. Hardware flushes denormals to zero (FTZ mode)
    // 3. exp(-87.34) = 1.18e-38 is the denormal boundary in float32
    // 4. For x < -13.2, -x²/2 < -87.12 produces denormals that get flushed
    //
    // NOTE: Research polynomial segments 4-6 don't work in float32 SFPU due to
    // precision loss with very small coefficients (1e-18 to 1e-10). The asymptotic
    // expansion provides better accuracy than polynomials for this range.
    //
    // For -13.2 < x < -5.5: Asymptotic expansion works reasonably well
    // For x < -13.2: FTZ forces result to 0 (unavoidable hardware limitation)
    v_elseif(val < -5.5f) {
        v_if(val < -13.2f) {
            // Below practical precision - exp produces denormals, FTZ flushes to 0
            result = 0.0f;
        }
        v_else {
            // Asymptotic: GELU(x) ≈ -exp(-x²/2) / √(2π)
            sfpi::vFloat x2 = val * val;
            sfpi::vFloat neg_half_x2 = x2 * sfpi::vFloat(-0.5f);
            sfpi::vFloat exp_val = _sfpu_exp_21f_<false>(neg_half_x2);
            result = exp_val * sfpi::vFloat(-0.3989422804f);
        }
        v_endif;
    }
    // Region 4: C6 adaptive polynomial segments [-5.5, 3.0]
    // Segments 7-15 from C6 research cover this range with good accuracy
    v_else {
        // Segment 7: [-5.5, -5.095]
        v_if(val < -5.095f) {
            sfpi::vFloat u = (val - sfpi::vFloat(-5.5745f)) * sfpi::vFloat(1.0f / 0.4796f);
            result = POLY4(-7.138e-08f, -1.697e-07f, -2.284e-07f, -2.645e-07f, -1.465e-07f, u);
        }
        v_else {
            // Segments 8-15: [-5.095, 3.0]
            v_if(val < -2.218f) {
                // Segments 8-10: [-5.095, -2.218]
                v_if(val < -4.136f) {
                    // Segment 8: [-5.095, -4.136]
                    sfpi::vFloat u = (val - sfpi::vFloat(-4.6154f)) * sfpi::vFloat(1.0f / 0.4796f);
                    result = POLY4(-9.126e-06f, -1.940e-05f, -2.070e-05f, -1.643e-05f, -7.190e-06f, u);
                }
                v_elseif(val < -3.177f) {
                    // Segment 9: [-4.136, -3.177]
                    sfpi::vFloat u = (val - sfpi::vFloat(-3.6563f)) * sfpi::vFloat(1.0f / 0.4796f);
                    result = POLY4(-4.680e-04f, -8.088e-04f, -6.515e-04f, -3.353e-04f, -1.001e-04f, u);
                }
                v_else {
                    // Segment 10: [-3.177, -2.218]
                    sfpi::vFloat u = (val - sfpi::vFloat(-2.6971f)) * sfpi::vFloat(1.0f / 0.4796f);
                    result = POLY4(-9.432e-03f, -1.192e-02f, -6.379e-03f, -1.642e-03f, -1.110e-04f, u);
                }
                v_endif;
            }
            v_else {
                // Segments 11-15: [-2.218, 3.0]
                v_if(val < -0.299f) {
                    // Segments 11-12: [-2.218, -0.299]
                    v_if(val < -1.258f) {
                        // Segment 11: [-2.218, -1.258]
                        sfpi::vFloat u = (val - sfpi::vFloat(-1.7380f)) * sfpi::vFloat(1.0f / 0.4796f);
                        result = POLY4(-7.144e-02f, -5.377e-02f, -1.033e-02f, 2.968e-03f, 1.521e-03f, u);
                    }
                    v_else {
                        // Segment 12: [-1.258, -0.299]
                        sfpi::vFloat u = (val - sfpi::vFloat(-0.7789f)) * sfpi::vFloat(1.0f / 0.4796f);
                        result = POLY4(-1.698e-01f, -5.330e-03f, 4.721e-02f, 1.369e-02f, -1.364e-04f, u);
                    }
                    v_endif;
                }
                v_else {
                    // Segments 13-15: [-0.299, 3.0]
                    // Note: Near-zero already handled by Taylor series
                    v_if(val < 0.660f) {
                        // Segment 13: [-0.299, 0.660]
                        sfpi::vFloat u = (val - sfpi::vFloat(0.1803f)) * sfpi::vFloat(1.0f / 0.4796f);
                        result = POLY4(1.030e-01f, 3.079e-01f, 8.875e-02f, -4.874e-03f, -3.119e-03f, u);
                    }
                    v_elseif(val < 1.644f) {
                        // Segment 14: [0.660, 1.644]
                        sfpi::vFloat u = (val - sfpi::vFloat(1.1517f)) * sfpi::vFloat(1.0f / 0.4919f);
                        result = POLY4(1.008e+00f, 5.469e-01f, 1.679e-02f, -1.223e-02f, 1.632e-03f, u);
                    }
                    v_else {
                        // Segment 15: [1.644, 3.0]
                        sfpi::vFloat u = (val - sfpi::vFloat(2.3218f)) * sfpi::vFloat(1.0f / 0.6782f);
                        result = POLY4(2.298e+00f, 7.140e-01f, -2.108e-02f, 3.512e-03f, 1.348e-03f, u);
                    }
                    v_endif;
                }
                v_endif;
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    return result;
}

// Legacy Chebyshev implementation (kept for reference/comparison)
#define POLYVAL15(c15, c14, c13, c12, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0, x)                         \
    (((((((((((((((c15) * (x) + (c14)) * (x) + (c13)) * (x) + (c12)) * (x) + (c11)) * (x) + (c10)) * (x) + (c9)) * \
                (x) +                                                                                              \
            (c8)) *                                                                                                \
               (x) +                                                                                               \
           (c7)) *                                                                                                 \
              (x) +                                                                                                \
          (c6)) *                                                                                                  \
             (x) +                                                                                                 \
         (c5)) *                                                                                                   \
            (x) +                                                                                                  \
        (c4)) *                                                                                                    \
           (x) +                                                                                                   \
       (c3)) *                                                                                                     \
          (x) +                                                                                                    \
      (c2)) *                                                                                                      \
         (x) +                                                                                                     \
     (c1)) * (x) +                                                                                                 \
        (c0)

inline sfpi::vFloat calculate_gelu_chebyshev(sfpi::vFloat val) {
    // Use C6 adaptive polynomial implementation
    return calculate_gelu_c6(val);
}

template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<APPROXIMATION_MODE>();
}

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
        v_elseif(in < 3.0f) { result = calculate_gelu_c6(in); }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<APPROXIMATION_MODE, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
