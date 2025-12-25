# GELU Hardware Precision Analysis Report

**Hardware:** Wormhole n150
**Date:** 2025-12-25
**Test Suite:** `tt-train/tests/ops/gelu_op_test.cpp`
**ULP Module:** `tt-train/tests/core/bf16_ulp.hpp`

## Executive Summary

Analysis of the GELU activation function implementation on Tenstorrent Wormhole hardware reveals a **precision anomaly** affecting tiny input values. The hardware produces only **two unique output values** for inputs in the range [1e-45, 1e-10]:

1. **0.0** for subnormal inputs
2. **2.980232e-05** for smallest normal inputs

This represents a loss of ~33 orders of magnitude in dynamic range, which contradicts the fundamental design goal of bfloat16 (preserving float32's exponent range while reducing mantissa precision).

## Test Methodology

### ULP (Units in Last Place) Measurement

ULP distance measures precision by counting representable floating-point values between computed and expected results. The analysis uses:

- **bf16_ulp module**: Precomputed order table for O(1) ULP distance lookup
- **Reference implementation**: Double-precision GELU with bf16 quantization
- **Formula**: GELU(x) = 0.5 × x × (1 + erf(x/√2))

### Test Coverage

| Test | Description |
|------|-------------|
| `DISABLED_GELU_ULP_DiagnosticDataCollection` | Collects ULP data for all 65,280 valid bf16 values |
| `DISABLED_GELU_ULP_SpikeAnalysis` | Analyzes ULP distribution by input region |
| `DISABLED_GELU_ULP_SpikeReport` | Generates full spike report (ULP > 10) |
| `DISABLED_GELU_ULP_FloorAnalysis` | Investigates the constant floor value |

## Findings

### 1. Output Floor Value

The hardware GELU implementation has a minimum non-zero output threshold:

```
Observed floor value: 2.980232e-05
  BF16 bits: 0x37F9
  Exponent (biased): 111, actual: -16
  Mantissa: 0x79 = 121/128
  Computed: (1 + 121/128) × 2^-16 ≈ 2.98e-05
```

Reference values for comparison:
- 2^-15 = 3.051758e-05
- 2^-16 = 1.525879e-05
- Smallest normal bf16 = 1.175494e-38

### 2. Spike Distribution by Input Magnitude

| Magnitude Range | Spike Count | Max ULP | Issue |
|-----------------|-------------|---------|-------|
| Subnormal (1e-45 to 1e-38) | 214 | 14,266 | Flushed to zero |
| Tiny (1e-38 to 1e-10) | 23,734 | 14,266 | Clamped to floor |
| Very small (1e-10 to 1e-3) | 5,948 | 2,462 | Gradual improvement |
| Small (1e-3 to 0.1) | 2 | 12 | Acceptable |
| Saturation (3 to 10) | 178 | 13,244 | GELU→0, hw→0 (correct) |

### 3. Detailed Examples

#### Tiny Normal Inputs (Worst Case)

| Input | Computed | Expected | ULP | Abs Diff |
|-------|----------|----------|-----|----------|
| 1.175494e-38 | 2.980232e-05 | 5.877472e-39 | 14,266 | 2.98e-05 |
| 1.184678e-38 | 2.980232e-05 | 5.877472e-39 | 14,266 | 2.98e-05 |
| 1.322431e-38 | 2.980232e-05 | 6.612156e-39 | 14,258 | 2.98e-05 |

**Observation:** All smallest normal bf16 inputs produce the identical output 2.980232e-05.

#### Subnormal Inputs

| Input | Computed | Expected | ULP | Subnormal |
|-------|----------|----------|-----|-----------|
| 9.184e-41 | 0.0 | 4.592e-41 | N/A | Yes |
| 1.561e-39 | 0.0 | 7.806e-40 | N/A | Yes |
| 5.969e-39 | 0.0 | 2.985e-39 | N/A | Yes |

**Observation:** All subnormal inputs are flushed to zero.

#### Negative Saturation Region

| Input | Computed | Expected | ULP | Issue |
|-------|----------|----------|-----|-------|
| -5.531250 | 0.0 | -8.754e-08 | 13,244 | Correct behavior |
| -6.000000 | 0.0 | -5.908e-09 | 12,747 | Correct behavior |
| -8.000000 | 0.0 | -4.885e-15 | 10,160 | Correct behavior |

**Observation:** This is actually correct - GELU(-6) ≈ 0, and hardware correctly outputs 0. The high ULP is an artifact of comparing tiny values.

### 4. ULP Distribution Summary

| ULP Range | Count | Percentage |
|-----------|-------|------------|
| 0 (exact) | ~35,000 | 53.6% |
| 1-10 | ~200 | 0.3% |
| 11-100 | 1,170 | 1.8% |
| 101-1000 | 2,154 | 3.3% |
| 1001-10000 | 18,011 | 27.6% |
| >10000 | 8,741 | 13.4% |

**Total values with ULP > 10:** 30,076 out of 65,280 (46.1%)

## Root Cause Analysis

### Hypothesis: erf() Approximation Underflow

The GELU function uses erf(x/√2). For tiny x:
- erf(ε) ≈ (2/√π) × ε for small ε
- GELU(x) ≈ 0.5x for tiny x

The hardware likely uses a polynomial approximation for erf() that:
1. Has a minimum representable output due to fixed-point intermediate calculations
2. Underflows to a constant when inputs are too small

### Evidence

1. **Only 2 unique outputs** for tiny inputs (0.0 and 2.98e-05)
2. **Floor value 0x37F9** suggests a specific computational artifact
3. **33 orders of magnitude error** rules out rounding - this is clamping

## Impact Assessment

### Affected Use Cases

| Use Case | Impact |
|----------|--------|
| Normal BERT/transformer training | **Low** - inputs typically in [-3, 3] |
| Gradient computation for tiny activations | **Medium** - may affect gradient flow |
| Networks with very small weight initialization | **High** - outputs will be wrong |
| Scientific computing requiring full dynamic range | **Critical** - defeats bf16 purpose |

### Unaffected Regions

The GELU active region (|x| ∈ [0.1, 3]) shows excellent precision:
- ULP typically 0-4
- Matches reference within bf16 tolerance

## Recommendations

### Short Term

1. **Document the limitation** in GELU op documentation
2. **Add input range validation** to warn users about tiny inputs
3. **Consider clamping inputs** to avoid silent precision loss

### Long Term

1. **Investigate erf() implementation** on Wormhole hardware
2. **Consider alternative GELU approximations** that handle tiny values
3. **File bug report** for hardware/firmware team

## Reproduction

```bash
# Run all diagnostic tests
cd ~/tt/tt-metal
./tt-train/build/tests/ttml_tests \
  --gtest_filter='*DISABLED_GELU_ULP*' \
  --gtest_also_run_disabled_tests

# Generate spike report
./tt-train/build/tests/ttml_tests \
  --gtest_filter='*DISABLED_GELU_ULP_SpikeReport*' \
  --gtest_also_run_disabled_tests
# Output: /tmp/gelu_ulp_spikes.md

# Generate visualization data
./tt-train/build/tests/ttml_tests \
  --gtest_filter='*DISABLED_GELU_ULP_DiagnosticDataCollection*' \
  --gtest_also_run_disabled_tests
# Output: /tmp/gelu_ulp_data.csv

# Plot visualization
python3 tt-train/tests/ops/plot_gelu_ulp.py --suffix wh_n150
```

## Files

| File | Description |
|------|-------------|
| `tt-train/tests/ops/gelu_op_test.cpp` | Test suite with diagnostic tests |
| `tt-train/tests/core/bf16_ulp.hpp` | BFloat16 ULP distance calculator |
| `tt-train/tests/core/bf16_ulp_test.cpp` | ULP module tests |
| `tt-train/tests/ops/plot_gelu_ulp.py` | Visualization script |
| `/tmp/gelu_ulp_spikes.md` | Full spike report (30,076 entries, generated) |
| `tt-train/tests/ops/gelu_ulp_plot_wh_n150_bf16_ulp_module.png` | ULP visualization |

## Appendix: Floor Value Derivation

The observed floor value 2.980232e-05 in IEEE 754 / BFloat16:

```
BFloat16 bits: 0x37F9
Binary: 0 01101111 1111001
        │ │──────│ │─────│
        │    │        │
        │    │        └── Mantissa: 121 (0x79)
        │    └── Exponent: 111 (biased), -16 (actual)
        └── Sign: positive

Value = (1 + 121/128) × 2^-16
      = (1 + 0.9453125) × 2^-16
      = 1.9453125 × 2^-16
      = 2.9802322e-05
```

This specific value (exponent -16, high mantissa) suggests the floor originates from the minimum representable non-zero result of an intermediate calculation in the GELU/erf implementation.
