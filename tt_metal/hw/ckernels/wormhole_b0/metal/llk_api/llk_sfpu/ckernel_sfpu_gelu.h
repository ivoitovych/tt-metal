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

// C7 Adaptive Polynomial GELU Implementation (v5)
// Target: Max ULP = 1 across entire BF16 range (DAZ+FTZ model)
// Reference: https://github.com/ivoitovych/bf16_gelu_research
// Research: GELU_BF16_Approximation_Range_-13.2_-5.5.md

// Degree-4 polynomial evaluation: c0 + c1*u + c2*u² + c3*u³ + c4*u⁴
// Using Horner's method: ((((c4*u + c3)*u + c2)*u + c1)*u + c0)
#define POLY4(c0, c1, c2, c3, c4, u) ((((c4) * (u) + (c3)) * (u) + (c2)) * (u) + (c1)) * (u) + (c0)

// Segment architecture (binary tree for efficient SFPU v_if/v_else execution):
// - FTZ region (x <= -13.1875): Returns 0 (true zero threshold verified with MPFR 256-bit)
// - Asymptotic (-13.1875 to -8.125): exp(-x²/2) approximation
// - Far-neg polynomials (-8.125 to -5.5): 11 segments (binary tree), Max ULP = 1
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
    // Region 3: Deep negative for x < -5.5
    // Architecture (binary tree for efficient SFPU execution):
    // - x <= -13.1875: FTZ to 0 (true threshold verified with MPFR 256-bit)
    // - -13.1875 < x <= -8.125: Asymptotic expansion exp(-x²/2) / √(2π)
    // - -8.125 < x <= -5.5: 11 polynomial segments with Max ULP = 1
    v_elseif(val < -5.5f) {
        v_if(val <= -8.125f) {
            // Far negative: FTZ or asymptotic
            v_if(val <= -13.1875f) {
                // True zero threshold (MPFR 256-bit verified)
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
        v_else {
            // Near negative polynomials: (-8.125, -5.5] - binary tree structure
            // Coefficients from MPFR 256-bit fitting, Max ULP = 1
            v_if(val <= -6.53125f) {
                // Left half: segments 0-5 (x <= -6.53125)
                v_if(val <= -7.28125f) {
                    // Segments 0-2 (x <= -7.28125)
                    v_if(val <= -7.78125f) {
                        // Segment 0: (-8.125, -7.78125]
                        sfpi::vFloat dx = val - sfpi::vFloat(-7.9375f);
                        result = POLY4(
                            -8.180560468869243e-15f,
                            -6.484898859467758e-14f,
                            -2.523291125856310e-13f,
                            -6.843968240039732e-13f,
                            -1.331450237814045e-12f,
                            dx);
                    }
                    v_else {
                        v_if(val <= -7.53125f) {
                            // Segment 1: (-7.78125, -7.53125]
                            sfpi::vFloat dx = val - sfpi::vFloat(-7.671875f);
                            result = POLY4(
                                -6.500450697402105e-14f,
                                -4.978690624902730e-13f,
                                -1.878765233332207e-12f,
                                -4.824423368625657e-12f,
                                -8.662829108288652e-12f,
                                dx);
                        }
                        v_else {
                            // Segment 2: (-7.53125, -7.28125]
                            sfpi::vFloat dx = val - sfpi::vFloat(-7.421875f);
                            result = POLY4(
                                -4.285191246178417e-13f,
                                -3.175470320990865e-12f,
                                -1.156774563451189e-11f,
                                -2.858713670532853e-11f,
                                -4.952127380651460e-11f,
                                dx);
                        }
                        v_endif;
                    }
                    v_endif;
                }
                v_else {
                    // Segments 3-5 (x > -7.28125 && x <= -6.53125)
                    v_if(val <= -6.78125f) {
                        v_if(val <= -7.03125f) {
                            // Segment 3: (-7.28125, -7.03125]
                            sfpi::vFloat dx = val - sfpi::vFloat(-7.171875f);
                            result = POLY4(
                                -2.652801618819745e-12f,
                                -1.899694266694665e-11f,
                                -6.677345595430689e-11f,
                                -1.585393708783509e-10f,
                                -2.643424840341101e-10f,
                                dx);
                        }
                        v_else {
                            // Segment 4: (-7.03125, -6.78125]
                            sfpi::vFloat dx = val - sfpi::vFloat(-6.921875f);
                            result = POLY4(
                                -1.542562398592692e-11f,
                                -1.066158090094188e-10f,
                                -3.610972901594848e-10f,
                                -8.224960005734471e-10f,
                                -1.317609742095899e-09f,
                                dx);
                        }
                        v_endif;
                    }
                    v_else {
                        // Segment 5: (-6.78125, -6.53125]
                        sfpi::vFloat dx = val - sfpi::vFloat(-6.671875f);
                        result = POLY4(
                            -8.425128748225761e-11f,
                            -5.612732063080472e-10f,
                            -1.828990740492901e-09f,
                            -3.990225435968794e-09f,
                            -6.129209359219904e-09f,
                            dx);
                    }
                    v_endif;
                }
                v_endif;
            }
            v_else {
                // Right half: segments 6-10 (x > -6.53125)
                v_if(val <= -5.78125f) {
                    // Segments 6-8 (x <= -5.78125)
                    v_if(val <= -6.28125f) {
                        // Segment 6: (-6.53125, -6.28125]
                        sfpi::vFloat dx = val - sfpi::vFloat(-6.421875f);
                        result = POLY4(
                            -4.322556608200194e-10f,
                            -2.771329408357892e-09f,
                            -8.674692233269561e-09f,
                            -1.809427712337921e-08f,
                            -2.659096659167986e-08f,
                            dx);
                    }
                    v_else {
                        v_if(val <= -6.03125f) {
                            // Segment 7: (-6.28125, -6.03125]
                            sfpi::vFloat dx = val - sfpi::vFloat(-6.171875f);
                            result = POLY4(
                                -2.082544510068984e-09f,
                                -1.283222811647506e-08f,
                                -3.851452122516594e-08f,
                                -7.665664854777053e-08f,
                                -1.075082558237318e-07f,
                                dx);
                        }
                        v_else {
                            // Segment 8: (-6.03125, -5.78125]
                            sfpi::vFloat dx = val - sfpi::vFloat(-5.921875f);
                            result = POLY4(
                                -9.424427685609932e-09f,
                                -5.571167607358211e-08f,
                                -1.600207559154601e-07f,
                                -3.032304568148156e-07f,
                                -4.046999703345027e-07f,
                                dx);
                        }
                        v_endif;
                    }
                    v_endif;
                }
                v_else {
                    // Segments 9-10 (x > -5.78125)
                    v_if(val <= -5.53125f) {
                        // Segment 9: (-5.78125, -5.53125]
                        sfpi::vFloat dx = val - sfpi::vFloat(-5.671875f);
                        result = POLY4(
                            -4.005568370868782e-08f,
                            -2.267462914073533e-07f,
                            -6.219276984480050e-07f,
                            -1.119222455258123e-06f,
                            -1.416892594988746e-06f,
                            dx);
                    }
                    v_else {
                        // Segment 10: (-5.53125, -5.5]
                        sfpi::vFloat dx = val - sfpi::vFloat(-5.515625f);
                        result = POLY4(-9.618875791587677e-08f, -5.282454813763634e-07f, 0.0f, 0.0f, 0.0f, dx);
                    }
                    v_endif;
                }
                v_endif;
            }
            v_endif;
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
