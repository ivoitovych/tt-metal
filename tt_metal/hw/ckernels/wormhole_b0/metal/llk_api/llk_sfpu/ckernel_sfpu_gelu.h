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

// C6 Adaptive Polynomial GELU Implementation (v3)
// Achieves Max ULP = 7 across entire BF16 range (DAZ+FTZ model)
// All polynomial segments have Max ULP = 1; only asymptotic region has ULP > 1
// Reference: https://github.com/ivoitovych/bf16_gelu_research

// Degree-4 polynomial evaluation: c0 + c1*u + c2*u² + c3*u³ + c4*u⁴
// Using Horner's method: ((((c4*u + c3)*u + c2)*u + c1)*u + c0)
#define POLY4(c0, c1, c2, c3, c4, u) ((((c4) * (u) + (c3)) * (u) + (c2)) * (u) + (c1)) * (u) + (c0)

// Segment architecture:
// - FTZ region (x < -13.2): Returns 0 (hardware limitation)
// - Asymptotic (-13.2 to -5.5): exp(-x²/2) approximation, Max ULP = 7
// - Range A/B/C (-5.5 to -3.177): Raw x polynomials, Max ULP = 1
// - Segments 10-15 (-3.177 to 3.0): Normalized u polynomials, Max ULP = 1
// - Positive saturation (x >= 3.0): Returns x, Max ULP = 1

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
    // For -13.2 < x < -5.5: Asymptotic expansion
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
    // Region 4: High-precision polynomial segments [-5.5, 3.0]
    // Ranges A, B, C use raw x polynomials (not normalized u)
    // Range A: [-5.5, -5.095] - 4 segments, Max ULP = 0
    // Range B: [-5.095, -4.136] - 2 segments, Max ULP = 1
    // Range C: [-4.136, -3.177] - 2 segments, Max ULP = 1
    v_else {
        v_if(val < -3.177f) {
            // Ranges A, B, C: [-5.5, -3.177] - High precision polynomials with raw x
            v_if(val < -5.095f) {
                // Range A: [-5.5, -5.095] - 4 segments, Max ULP = 0
                v_if(val < -5.2975f) {
                    v_if(val < -5.39875f) {
                        // Segment A0: [-5.5, -5.39875)
                        result = POLY4(
                            0.000313728989568f,
                            0.000253022968536f,
                            7.35517023713e-05f,
                            9.21681203181e-06f,
                            4.22158052515e-07f,
                            val);
                    }
                    v_else {
                        // Segment A1: [-5.39875, -5.2975)
                        result = POLY4(
                            2.93236917059e-06f,
                            1.44516488945e-05f,
                            4.33349441664e-06f,
                            2.32314903315e-07f,
                            -1.7470934921e-08f,
                            val);
                    }
                    v_endif;
                }
                v_else {
                    v_if(val < -5.19625f) {
                        // Segment A2: [-5.2975, -5.19625)
                        result = POLY4(
                            3.80656160814e-07f,
                            3.37766759912e-05f,
                            1.22165192806e-05f,
                            1.0748110526e-06f,
                            -6.10905770415e-09f,
                            val);
                    }
                    v_else {
                        // Segment A3: [-5.19625, -5.095]
                        result = POLY4(
                            -8.80184088601e-05f,
                            -9.4080778581e-06f,
                            2.72905685961e-06f,
                            -3.01319659002e-07f,
                            -1.06109432352e-07f,
                            val);
                    }
                    v_endif;
                }
                v_endif;
            }
            v_elseif(val < -4.136f) {
                // Range B: [-5.095, -4.136] - 2 segments, Max ULP = 1
                v_if(val < -4.6155f) {
                    // Segment B0: [-5.095, -4.6155)
                    result = POLY4(
                        -0.0321705937386f,
                        -0.0249685551971f,
                        -0.00728190457448f,
                        -0.000945661275182f,
                        -4.61338604509e-05f,
                        val);
                }
                v_else {
                    // Segment B1: [-4.6155, -4.136]
                    result = POLY4(
                        -0.129026979208f,
                        -0.109589219093f,
                        -0.0350283384323f,
                        -0.00499239563942f,
                        -0.000267637165962f,
                        val);
                }
                v_endif;
            }
            v_else {
                // Range C: [-4.136, -3.177] - 2 segments, Max ULP = 1
                v_if(val < -3.6565f) {
                    // Segment C0: [-4.136, -3.6565)
                    result = POLY4(
                        -0.396927714348f,
                        -0.371776491404f,
                        -0.131340071559f,
                        -0.0207304283977f,
                        -0.00123285665177f,
                        val);
                }
                v_else {
                    // Segment C1: [-3.6565, -3.177]
                    result = POLY4(
                        -0.665469408035f,
                        -0.666755318642f,
                        -0.252851009369f,
                        -0.0429779663682f,
                        -0.00276047317311f,
                        val);
                }
                v_endif;
            }
            v_endif;
        }
        v_elseif(val < -2.218f) {
            // Segment 10: [-3.177, -2.218]
            sfpi::vFloat u = (val - sfpi::vFloat(-2.6971f)) * sfpi::vFloat(1.0f / 0.4796f);
            result = POLY4(-9.432e-03f, -1.192e-02f, -6.379e-03f, -1.642e-03f, -1.110e-04f, u);
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
