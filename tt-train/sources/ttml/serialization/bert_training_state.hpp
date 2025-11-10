// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "models/base_transformer.hpp"
#include "optimizers/optimizer_base.hpp"
#include "serialization/msgpack_file.hpp"

namespace ttml::serialization {

/**
 * @brief Training state for BERT task models
 *
 * Contains all information needed to resume training:
 * - Model parameters
 * - Optimizer state
 * - Training progress (step, epoch, best loss)
 * - Configuration metadata
 */
struct BertTrainingState {
    uint32_t global_step = 0;
    uint32_t epoch = 0;
    float best_loss = std::numeric_limits<float>::max();
    float current_loss = 0.0F;
    std::string model_type;  // e.g., "BertForSequenceClassification"
    std::string timestamp;
};

/**
 * @brief Save complete training state to MsgPack file
 *
 * Saves:
 * - Model parameters (all weights and biases)
 * - Optimizer state (momentum, learning rate, etc.)
 * - Training progress metadata
 *
 * @param path Path to save checkpoint file
 * @param model The BERT task model
 * @param optimizer The optimizer
 * @param state Training state metadata
 */
void save_bert_training_state(
    const std::filesystem::path& path,
    const models::BaseTransformer& model,
    const optimizers::OptimizerBase& optimizer,
    const BertTrainingState& state);

/**
 * @brief Load complete training state from MsgPack file
 *
 * Loads:
 * - Model parameters (all weights and biases)
 * - Optimizer state (momentum, learning rate, etc.)
 * - Training progress metadata
 *
 * @param path Path to checkpoint file
 * @param model The BERT task model (must be created first)
 * @param optimizer The optimizer (must be created first)
 * @param state Training state metadata (output)
 */
void load_bert_training_state(
    const std::filesystem::path& path,
    models::BaseTransformer& model,
    optimizers::OptimizerBase& optimizer,
    BertTrainingState& state);

/**
 * @brief Convenience function: Save training state with default metadata
 *
 * Automatically fills in timestamp and basic metadata.
 *
 * @param path Path to save checkpoint file
 * @param model The BERT task model
 * @param optimizer The optimizer
 * @param global_step Current training step
 * @param best_loss Best validation loss so far
 */
void save_bert_checkpoint(
    const std::filesystem::path& path,
    const models::BaseTransformer& model,
    const optimizers::OptimizerBase& optimizer,
    uint32_t global_step,
    float best_loss);

/**
 * @brief Convenience function: Load training state
 *
 * @param path Path to checkpoint file
 * @param model The BERT task model (must be created first)
 * @param optimizer The optimizer (must be created first)
 * @return Training state metadata
 */
BertTrainingState load_bert_checkpoint(
    const std::filesystem::path& path, models::BaseTransformer& model, optimizers::OptimizerBase& optimizer);

/**
 * @brief Save only model parameters (no optimizer state)
 *
 * Useful for saving final trained model or for inference.
 *
 * @param path Path to save model file
 * @param model The BERT task model
 */
void save_bert_model_only(const std::filesystem::path& path, const models::BaseTransformer& model);

/**
 * @brief Load only model parameters (no optimizer state)
 *
 * Useful for loading a trained model for inference.
 *
 * @param path Path to model file
 * @param model The BERT task model (must be created first)
 */
void load_bert_model_only(const std::filesystem::path& path, models::BaseTransformer& model);

/**
 * @brief List available checkpoints in a directory
 *
 * Scans directory for .msgpack checkpoint files and returns metadata.
 *
 * @param directory Directory to scan
 * @return Vector of checkpoint paths with metadata
 */
std::vector<std::pair<std::filesystem::path, BertTrainingState>> list_checkpoints(
    const std::filesystem::path& directory);

/**
 * @brief Find best checkpoint in directory (by validation loss)
 *
 * @param directory Directory to scan
 * @return Path to best checkpoint, or empty if none found
 */
std::filesystem::path find_best_checkpoint(const std::filesystem::path& directory);

}  // namespace ttml::serialization
