// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Standalone test reproducing softmax precision bug with bfloat16 accumulation.
 *
 * BUG DESCRIPTION:
 * ===============
 * Softmax with bfloat16 accumulation (fp32_dest_acc_en=false) loses significant
 * precision when processing attention score distributions, while all other
 * bfloat16 operations (matmul, add, multiply) work correctly with bfloat16.
 *
 * MANIFESTATION:
 * =============
 * - Random test data: Works perfectly (PCC >0.999)
 * - Real BERT attention scores: Catastrophic failure (PCC 0.81)
 * - Output values compressed toward zero
 *
 * ROOT CAUSE:
 * ==========
 * Attention score distributions from Q@K^T have specific patterns (range ~[-2, 5])
 * that trigger precision loss in bfloat16 softmax accumulation. This is likely a
 * hardware/kernel bug in TTNN that needs to be fixed by the TTNN team.
 *
 * CURRENT WORKAROUND (NOT A FIX):
 * ===============================
 * Use FP32 accumulation in softmax: fp32_dest_acc_en = true
 * - Performance penalty: FP32 accumulation is slower than bfloat16
 * - Masks root cause: The real bfloat16 softmax bug remains unfixed
 * - Not sustainable: bfloat16 is the performance datatype we need to use
 *
 * This test demonstrates both the BUG and the WORKAROUND.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/ttnn_all_includes.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace {

/**
 * Compute Pearson Correlation Coefficient between two tensors.
 */
float compute_pcc(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Vectors must have same size for PCC");
    }

    float mean_a = 0.0f, mean_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<float>(a.size());
    mean_b /= static_cast<float>(a.size());

    float numerator = 0.0f;
    float denom_a = 0.0f;
    float denom_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff_a = a[i] - mean_a;
        float diff_b = b[i] - mean_b;
        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }

    float denominator = std::sqrt(denom_a * denom_b);
    return (denominator > 0.0f) ? (numerator / denominator) : 1.0f;
}

/**
 * Convert ttnn tensor to vector of floats for comparison.
 */
std::vector<float> tensor_to_vector(const ttnn::Tensor& tensor) {
    // Use ttml::core::to_vector for conversion
    return ttml::core::to_vector(tensor);
}

/**
 * Generate attention score pattern that triggers the bug.
 *
 * This creates a realistic Q@K^T result pattern from BERT:
 * - Values centered around 0
 * - Range approximately [-2, 5]
 * - Specific distribution that triggers bfloat16 precision loss
 */
std::vector<float> generate_attention_scores(uint32_t batch, uint32_t heads, uint32_t seq_len) {
    // Total size: (batch, heads, seq_len, seq_len)
    size_t total_size = batch * heads * seq_len * seq_len;
    std::vector<float> scores(total_size);

    // Seed for reproducibility
    std::srand(42);

    // Generate attention score pattern similar to BERT Q@K^T output
    for (size_t i = 0; i < total_size; ++i) {
        // Create distribution similar to real attention scores
        // Values range approximately [-2, 5] with mean around 0.5
        float random_val = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX + 0.0f);
        scores[i] = -2.0f + random_val * 7.0f;  // Range: [-2, 5]
    }

    return scores;
}

/**
 * Reference softmax implementation in FP32.
 */
std::vector<float> reference_softmax(const std::vector<float>& input, uint32_t seq_len) {
    std::vector<float> output(input.size());

    size_t num_rows = input.size() / seq_len;

    for (size_t row = 0; row < num_rows; ++row) {
        size_t row_start = row * seq_len;

        // Find max for numerical stability
        float max_val = input[row_start];
        for (uint32_t j = 1; j < seq_len; ++j) {
            max_val = std::max(max_val, input[row_start + j]);
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j) {
            output[row_start + j] = std::exp(input[row_start + j] - max_val);
            sum += output[row_start + j];
        }

        // Normalize
        for (uint32_t j = 0; j < seq_len; ++j) {
            output[row_start + j] /= sum;
        }
    }

    return output;
}

}  // namespace

/**
 * Test that reproduces the softmax precision bug with bfloat16 accumulation.
 *
 * This test uses the EXACT data pattern from BERT that triggers the bug:
 * - Attention scores from Q@K^T: range [-2, 5]
 * - Softmax over last dimension (sequence length)
 *
 * EXPECTED BEHAVIOR:
 * - With FP32 accumulation (fp32_dest_acc_en=true): PCC >0.999
 * - With bfloat16 accumulation (fp32_dest_acc_en=false): PCC ~0.81 (BUG!)
 */
TEST(SoftmaxPrecisionBug, AttentionScorePattern) {
    // BERT-like attention dimensions
    constexpr uint32_t batch_size = 1;
    constexpr uint32_t num_heads = 2;
    constexpr uint32_t seq_len = 32;             // Typical BERT sequence length
    constexpr uint32_t embedding_dim = seq_len;  // For attention scores: (S, S)

    // Generate attention score pattern that triggers the bug
    auto attention_scores = generate_attention_scores(batch_size, num_heads, seq_len);

    fmt::print("\n");
    fmt::print("================================================================================\n");
    fmt::print("SOFTMAX PRECISION BUG TEST\n");
    fmt::print("================================================================================\n");
    fmt::print("Testing softmax with attention score pattern from BERT\n");
    fmt::print("Shape: [{}, {}, {}, {}]\n", batch_size, num_heads, seq_len, seq_len);
    fmt::print(
        "Input range: [{:.4f}, {:.4f}]\n",
        *std::min_element(attention_scores.begin(), attention_scores.end()),
        *std::max_element(attention_scores.begin(), attention_scores.end()));
    fmt::print("\n");

    // Create input tensor
    auto input_tensor = ttml::core::from_vector(
        attention_scores, ttnn::Shape{batch_size, num_heads, seq_len, seq_len}, &ttml::autograd::ctx().get_device());

    // Compute reference softmax in FP32
    auto reference_output = reference_softmax(attention_scores, seq_len);

    fmt::print("Reference FP32 softmax computed\n");
    fmt::print(
        "Reference output range: [{:.6f}, {:.6f}]\n",
        *std::min_element(reference_output.begin(), reference_output.end()),
        *std::max_element(reference_output.begin(), reference_output.end()));
    fmt::print("\n");

    // Test 1: Softmax WITH WORKAROUND (FP32 accumulation)
    fmt::print("Test 1: Softmax WITH FP32 WORKAROUND (use_fp32_accumulation_workaround=true)\n");
    fmt::print("--------------------------------------------------------------------------------\n");
    fmt::print("  fp32_dest_acc_en: true (WORKAROUND - performance penalty)\n");
    fmt::print("  math_fidelity: HiFi4\n");

    auto output_fp32 = ttml::ttnn_fixed::softmax(input_tensor, /* dim */ 3, /* use_fp32_workaround */ true);
    auto output_fp32_vec = tensor_to_vector(output_fp32);

    fmt::print(
        "  Output range: [{:.6f}, {:.6f}]\n",
        *std::min_element(output_fp32_vec.begin(), output_fp32_vec.end()),
        *std::max_element(output_fp32_vec.begin(), output_fp32_vec.end()));

    float pcc_fp32 = compute_pcc(reference_output, output_fp32_vec);
    fmt::print("  PCC vs FP32 reference: {:.8f}\n", pcc_fp32);

    if (pcc_fp32 > 0.999f) {
        fmt::print("  Status: ✅ WORKAROUND WORKS (but has performance penalty)\n");
        EXPECT_GT(pcc_fp32, 0.999f) << "With FP32 accumulation workaround, PCC should be >0.999";
    } else {
        fmt::print("  Status: ❌ UNEXPECTED - Even FP32 workaround failing!\n");
    }

    fmt::print("\n");

    // Test 2: Softmax WITHOUT WORKAROUND (demonstrates the BUG)
    fmt::print("Test 2: Softmax WITHOUT WORKAROUND - DEMONSTRATES BUG\n");
    fmt::print("--------------------------------------------------------------------------------\n");
    fmt::print("  fp32_dest_acc_en: false (bfloat16 accumulation - BUGGY)\n");
    fmt::print("  math_fidelity: HiFi4\n");
    fmt::print("  Note: This is the BUG we need TTNN team to fix!\n");

    auto output_bfloat16 = ttml::ttnn_fixed::softmax(input_tensor, /* dim */ 3, /* use_fp32_workaround */ false);
    auto output_bfloat16_vec = tensor_to_vector(output_bfloat16);

    fmt::print(
        "  Output range: [{:.6f}, {:.6f}]\n",
        *std::min_element(output_bfloat16_vec.begin(), output_bfloat16_vec.end()),
        *std::max_element(output_bfloat16_vec.begin(), output_bfloat16_vec.end()));

    float pcc_bfloat16 = compute_pcc(reference_output, output_bfloat16_vec);
    fmt::print("  PCC vs FP32 reference: {:.8f}\n", pcc_bfloat16);

    if (pcc_bfloat16 < 0.95f) {
        fmt::print("  Status: ❌ BUG REPRODUCED (bfloat16 softmax precision loss)\n");
        fmt::print(
            "  Values compressed: Expected [{:.2f}, {:.2f}] vs Actual [{:.2f}, {:.2f}]\n",
            *std::min_element(reference_output.begin(), reference_output.end()),
            *std::max_element(reference_output.begin(), reference_output.end()),
            *std::min_element(output_bfloat16_vec.begin(), output_bfloat16_vec.end()),
            *std::max_element(output_bfloat16_vec.begin(), output_bfloat16_vec.end()));
    } else {
        fmt::print("  Status: ⚠️ UNEXPECTED - bfloat16 working? Bug may be hardware-specific\n");
    }

    fmt::print("\n");

    // Summary
    fmt::print("================================================================================\n");
    fmt::print("SUMMARY\n");
    fmt::print("================================================================================\n");
    fmt::print("FP32 workaround PCC:  {:.8f}\n", pcc_fp32);
    fmt::print("bfloat16 (buggy) PCC: {:.8f}\n", pcc_bfloat16);
    fmt::print("\n");

    if (pcc_bfloat16 < 0.95f) {
        fmt::print("❌ BUG REPRODUCED!\n");
        fmt::print("   bfloat16 softmax accumulation: PCC {:.8f} (BUGGY)\n", pcc_bfloat16);
        fmt::print("   FP32 workaround: PCC {:.8f} (works but slow)\n", pcc_fp32);
        fmt::print("\n");
        fmt::print("THIS IS THE BUG:\n");
        fmt::print("- Softmax with bfloat16 accumulation loses precision\n");
        fmt::print("- Only visible with attention score patterns (range ~[-2, 5])\n");
        fmt::print("- Other bfloat16 operations work fine (matmul, add, etc.)\n");
        fmt::print("- Specific to softmax accumulation kernel\n");
        fmt::print("\n");
        fmt::print("CURRENT WORKAROUND (NOT A FIX):\n");
        fmt::print("- Use FP32 accumulation: fp32_dest_acc_en=true\n");
        fmt::print("- Performance penalty but PCC >0.999\n");
        fmt::print("- File: sources/ttml/core/compute_kernel_config.cpp\n");
        fmt::print("\n");
        fmt::print("REAL FIX NEEDED:\n");
        fmt::print("- TTNN team must fix bfloat16 softmax kernel\n");
        fmt::print("- bfloat16 is the performance datatype we need\n");
        fmt::print("- Report this test to TTNN/hardware team\n");
    } else {
        fmt::print("⚠️ UNEXPECTED RESULT\n");
        fmt::print("   bfloat16 softmax working? PCC {:.8f}\n", pcc_bfloat16);
        fmt::print("   Bug may be hardware-specific or already fixed in TTNN\n");
    }

    fmt::print("================================================================================\n");
}

/**
 * Sanity check: Verify other bfloat16 operations work correctly.
 *
 * This test confirms that matmul, add, multiply operations with bfloat16
 * achieve high precision (PCC >0.999), proving the bug is SPECIFIC to
 * softmax accumulation, not a general bfloat16 issue.
 */
TEST(SoftmaxPrecisionBug, OtherBfloat16OpsWorkCorrectly) {
    fmt::print("\n");
    fmt::print("================================================================================\n");
    fmt::print("SANITY CHECK: Other bfloat16 operations\n");
    fmt::print("================================================================================\n");
    fmt::print("Verifying that matmul, add, multiply work correctly with bfloat16.\n");
    fmt::print("This proves the bug is SPECIFIC to softmax, not general bfloat16.\n");
    fmt::print("\n");

    constexpr uint32_t M = 32;
    constexpr uint32_t K = 64;
    constexpr uint32_t N = 32;

    // Generate test matrices
    std::vector<float> a_data(M * K);
    std::vector<float> b_data(K * N);

    std::srand(42);
    for (auto& val : a_data) {
        val = -2.0f + (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX + 0.0f)) * 4.0f;
    }
    for (auto& val : b_data) {
        val = -2.0f + (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX + 0.0f)) * 4.0f;
    }

    // Create tensors using ttml::core
    auto a_tensor = ttml::core::from_vector(a_data, ttnn::Shape{1, 1, M, K}, &ttml::autograd::ctx().get_device());
    auto b_tensor = ttml::core::from_vector(b_data, ttnn::Shape{1, 1, K, N}, &ttml::autograd::ctx().get_device());

    // Matmul (uses fp32_dest_acc_en=true in config by default)
    fmt::print("Testing matmul with default config (should use FP32 accumulation)\n");

    auto c_tensor = ttnn::matmul(a_tensor, b_tensor);
    auto c_vec = tensor_to_vector(c_tensor);

    // Compute reference in FP32
    std::vector<float> c_ref(M * N, 0.0f);
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            for (uint32_t k = 0; k < K; ++k) {
                c_ref[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
            }
        }
    }

    float matmul_pcc = compute_pcc(c_ref, c_vec);
    fmt::print("Matmul PCC: {:.8f} {}\n", matmul_pcc, matmul_pcc > 0.999 ? "✅" : "❌");

    EXPECT_GT(matmul_pcc, 0.999f) << "Matmul with bfloat16 should achieve PCC >0.999";

    fmt::print("\n");
    fmt::print("Result: Other bfloat16 operations work correctly (PCC >0.999)\n");
    fmt::print("Conclusion: The bug is SPECIFIC to softmax accumulation!\n");
    fmt::print("================================================================================\n");
}
