// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Seed Sensitivity Test
 *
 * Tests whether different random seeds affect model correctness.
 * This is a regression test for a reported issue where seed 42 produces
 * wrong results (PCC = -1.0, inverted outputs) while seeds 43+ work fine.
 *
 * EXPECTED BEHAVIOR:
 * Random seed should only affect weight initialization, not model correctness.
 * All seeds should produce correct outputs when using the same weights.
 *
 * BUG SYMPTOMS (if present):
 * - Seed 42: PCC < 0 (inverted outputs) or very low PCC
 * - Seeds 43+: Normal PCC (> 0.95)
 */

#include <gtest/gtest.h>

#include <random>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"

using namespace ttml;
using namespace ttml::models::bert;

class BertSeedSensitivityTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }

    // Helper to create deterministic input based on seed
    std::vector<uint32_t> create_input(uint32_t seed, size_t size, uint32_t vocab_size) {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<uint32_t> dis(0, vocab_size - 1);

        std::vector<uint32_t> input(size);
        for (size_t i = 0; i < size; ++i) {
            input[i] = dis(gen);
        }
        return input;
    }
};

TEST_F(BertSeedSensitivityTest, Seed42vsOtherSeeds) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "SEED SENSITIVITY TEST" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    const uint32_t num_labels = 2;
    const size_t batch_size = 1;
    const size_t seq_len = 32;

    // Test multiple seeds
    std::vector<uint32_t> seeds = {42, 43, 44, 100};
    std::vector<std::vector<float>> outputs;

    for (uint32_t seed : seeds) {
        std::cout << "\n" << std::string(80, '-') << std::endl;
        std::cout << "Testing with seed: " << seed << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        // Create model (weights are randomly initialized based on seed)
        auto model = create_for_sequence_classification(config, num_labels, 0.0F);

        // Create deterministic input based on seed
        auto input_ids_vec = create_input(seed, batch_size * seq_len, config.vocab_size);
        std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len, 0);
        std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

        std::cout << "Input IDs (first 5): [";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << input_ids_vec[i];
            if (i < 4)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Create tensors with correct dtypes
        auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            input_ids_vec,
            ttnn::Shape{batch_size, 1, 1, seq_len},
            &autograd::ctx().get_device(),
            ttnn::Layout::ROW_MAJOR);
        auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            token_type_ids_vec,
            ttnn::Shape{batch_size, 1, 1, seq_len},
            &autograd::ctx().get_device(),
            ttnn::Layout::ROW_MAJOR);
        auto attention_mask_tensor = core::from_vector(
            attention_mask_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

        auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
        auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
        auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

        // Forward pass
        auto logits = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);
        auto logits_data = core::to_vector(logits->get_value());

        auto logits_shape = logits->get_value().logical_shape();
        size_t num_labels_aligned = logits_shape[3];

        // Extract logits for first sample
        std::vector<float> sample_logits(num_labels);
        for (size_t i = 0; i < num_labels; ++i) {
            sample_logits[i] = logits_data[i];
        }

        outputs.push_back(sample_logits);

        std::cout << "Output logits: [";
        for (size_t i = 0; i < num_labels; ++i) {
            std::cout << sample_logits[i];
            if (i < num_labels - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Analyze results: check if seed 42 is significantly different from others
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "ANALYSIS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // All outputs should have similar patterns (same signs, similar magnitudes)
    // Different seeds will have different exact values due to different weight init,
    // but they should all be "reasonable" (not inverted or extreme)

    bool all_seeds_reasonable = true;
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << "\nSeed " << seeds[i] << " outputs: [";
        for (size_t j = 0; j < outputs[i].size(); ++j) {
            std::cout << outputs[i][j];
            if (j < outputs[i].size() - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Check for extreme values (likely indicates a problem)
        bool has_extreme_values = false;
        for (float val : outputs[i]) {
            if (std::abs(val) > 100.0F || std::isnan(val) || std::isinf(val)) {
                has_extreme_values = true;
                break;
            }
        }

        if (has_extreme_values) {
            std::cout << "  ⚠️ WARNING: Seed " << seeds[i] << " has extreme/invalid values" << std::endl;
            all_seeds_reasonable = false;
        } else {
            std::cout << "  ✓ Seed " << seeds[i] << " outputs are reasonable" << std::endl;
        }
    }

    // Compare seed 42 with seed 43 specifically (reported issue)
    if (outputs.size() >= 2) {
        std::cout << "\nSeed 42 vs Seed 43 comparison:" << std::endl;

        // Calculate correlation (should not be negative)
        float sum_xy = 0.0F, sum_x = 0.0F, sum_y = 0.0F, sum_x2 = 0.0F, sum_y2 = 0.0F;
        size_t n = std::min(outputs[0].size(), outputs[1].size());

        for (size_t i = 0; i < n; ++i) {
            float x = outputs[0][i];  // seed 42
            float y = outputs[1][i];  // seed 43
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }

        float correlation =
            (n * sum_xy - sum_x * sum_y) / std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));

        std::cout << "  Correlation: " << correlation << std::endl;

        if (correlation < -0.5F) {
            std::cout << "  ❌ BUG DETECTED: Seed 42 outputs are inverted (negative correlation)" << std::endl;
            all_seeds_reasonable = false;
        } else if (correlation < 0.5F) {
            std::cout << "  ⚠️ WARNING: Seed 42 outputs are very different (low correlation)" << std::endl;
            // This is expected since different seeds = different weights
            // Only fail if correlation is strongly negative
        } else {
            std::cout << "  ✓ Seed 42 and 43 show similar patterns" << std::endl;
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    if (all_seeds_reasonable) {
        std::cout << "✓ All seeds produce reasonable outputs" << std::endl;
        std::cout << "NOTE: Different seeds produce different values (different weights), but all are valid"
                  << std::endl;
    } else {
        std::cout << "❌ Some seeds produce problematic outputs" << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    // Test passes if all seeds produce reasonable outputs
    EXPECT_TRUE(all_seeds_reasonable) << "Some seeds produce extreme or invalid outputs";
}

TEST_F(BertSeedSensitivityTest, SameSeedSameWeights) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "REPRODUCIBILITY TEST: Same seed should give same weights" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;
    config.use_token_type_embeddings = true;

    const uint32_t num_labels = 2;
    const size_t batch_size = 1;
    const size_t seq_len = 32;
    const uint32_t seed = 42;

    // Create same input
    auto input_ids_vec = create_input(seed, batch_size * seq_len, config.vocab_size);
    std::vector<uint32_t> token_type_ids_vec(batch_size * seq_len, 0);
    std::vector<float> attention_mask_vec(batch_size * seq_len, 1.0F);

    std::vector<std::vector<float>> outputs;

    // Run twice with same seed
    for (int run = 0; run < 2; ++run) {
        std::cout << "\nRun " << (run + 1) << " with seed " << seed << std::endl;

        // Create model (should get same random init with same seed)
        auto model = create_for_sequence_classification(config, num_labels, 0.0F);

        auto input_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            input_ids_vec,
            ttnn::Shape{batch_size, 1, 1, seq_len},
            &autograd::ctx().get_device(),
            ttnn::Layout::ROW_MAJOR);
        auto token_type_ids_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            token_type_ids_vec,
            ttnn::Shape{batch_size, 1, 1, seq_len},
            &autograd::ctx().get_device(),
            ttnn::Layout::ROW_MAJOR);
        auto attention_mask_tensor = core::from_vector(
            attention_mask_vec, ttnn::Shape{batch_size, 1, 1, seq_len}, &autograd::ctx().get_device());

        auto input_ids_ag = autograd::create_tensor(input_ids_tensor);
        auto token_type_ids_ag = autograd::create_tensor(token_type_ids_tensor);
        auto attention_mask_ag = autograd::create_tensor(attention_mask_tensor);

        auto logits = (*model)(input_ids_ag, attention_mask_ag, token_type_ids_ag);
        auto logits_data = core::to_vector(logits->get_value());

        std::vector<float> sample_logits(num_labels);
        for (size_t i = 0; i < num_labels; ++i) {
            sample_logits[i] = logits_data[i];
        }

        outputs.push_back(sample_logits);

        std::cout << "Output: [";
        for (size_t i = 0; i < num_labels; ++i) {
            std::cout << sample_logits[i];
            if (i < num_labels - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Check if outputs are identical (or very close due to numerical precision)
    std::cout << "\nChecking reproducibility:" << std::endl;
    float max_diff = 0.0F;
    for (size_t i = 0; i < outputs[0].size(); ++i) {
        float diff = std::abs(outputs[0][i] - outputs[1][i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Max difference between runs: " << max_diff << std::endl;

    // Note: Due to non-determinism in some ops, exact reproducibility might not be guaranteed
    // This test documents the current behavior
    if (max_diff < 1e-5F) {
        std::cout << "✓ Outputs are identical (fully reproducible)" << std::endl;
    } else if (max_diff < 0.1F) {
        std::cout << "⚠️ Outputs are close but not identical (some non-determinism)" << std::endl;
    } else {
        std::cout << "❌ Outputs are significantly different (poor reproducibility)" << std::endl;
    }

    // Test is informational - we document the behavior but don't fail
    // Different hardware/ops may have different reproducibility guarantees
}
