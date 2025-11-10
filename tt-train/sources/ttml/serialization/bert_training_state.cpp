// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bert_training_state.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "serialization/serialization.hpp"

namespace ttml::serialization {

namespace {

/**
 * Get current timestamp as string
 */
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

}  // namespace

// ============================================================================
// Save Training State
// ============================================================================

void save_bert_training_state(
    const std::filesystem::path& path,
    const models::BaseTransformer& model,
    const optimizers::OptimizerBase& optimizer,
    const BertTrainingState& state) {
    fmt::print("Saving BERT training state to: {}\n", path.string());

    MsgPackFile file;

    // Save metadata
    file.put("version", "1.0");
    file.put("model_type", std::string_view{state.model_type});
    file.put("timestamp", std::string_view{state.timestamp});
    file.put("global_step", state.global_step);
    file.put("epoch", state.epoch);
    file.put("best_loss", state.best_loss);
    file.put("current_loss", state.current_loss);

    // Save model parameters
    fmt::print("  Saving model parameters...\n");
    write_module(file, "model", &model);

    // Save optimizer state
    fmt::print("  Saving optimizer state...\n");
    write_optimizer(file, "optimizer", &optimizer);

    // Serialize to disk
    file.serialize(path.string());

    fmt::print("  Checkpoint saved successfully\n");
}

// ============================================================================
// Load Training State
// ============================================================================

void load_bert_training_state(
    const std::filesystem::path& path,
    models::BaseTransformer& model,
    optimizers::OptimizerBase& optimizer,
    BertTrainingState& state) {
    fmt::print("Loading BERT training state from: {}\n", path.string());

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Checkpoint file does not exist: " + path.string());
    }

    MsgPackFile file;
    file.deserialize(path.string());

    // Load metadata
    std::string version;
    file.get("version", version);
    file.get("model_type", state.model_type);
    file.get("timestamp", state.timestamp);
    file.get("global_step", state.global_step);
    file.get("epoch", state.epoch);
    file.get("best_loss", state.best_loss);
    file.get("current_loss", state.current_loss);

    fmt::print("  Checkpoint info:\n");
    fmt::print("    Version: {}\n", version);
    fmt::print("    Model type: {}\n", state.model_type);
    fmt::print("    Step: {}, Epoch: {}\n", state.global_step, state.epoch);
    fmt::print("    Best loss: {:.4f}\n", state.best_loss);

    // Load model parameters
    fmt::print("  Loading model parameters...\n");
    read_module(file, "model", &model);

    // Load optimizer state
    fmt::print("  Loading optimizer state...\n");
    read_optimizer(file, "optimizer", &optimizer);

    fmt::print("  Checkpoint loaded successfully\n");
}

// ============================================================================
// Convenience Functions
// ============================================================================

void save_bert_checkpoint(
    const std::filesystem::path& path,
    const models::BaseTransformer& model,
    const optimizers::OptimizerBase& optimizer,
    uint32_t global_step,
    float best_loss) {
    BertTrainingState state;
    state.global_step = global_step;
    state.epoch = 0;  // Epoch tracking optional - can use full save_bert_training_state() if needed
    state.best_loss = best_loss;
    state.current_loss = 0.0F;
    state.model_type = "BertTaskModel";  // Generic
    state.timestamp = get_timestamp();

    save_bert_training_state(path, model, optimizer, state);
}

BertTrainingState load_bert_checkpoint(
    const std::filesystem::path& path, models::BaseTransformer& model, optimizers::OptimizerBase& optimizer) {
    BertTrainingState state;
    load_bert_training_state(path, model, optimizer, state);
    return state;
}

// ============================================================================
// Model-Only Save/Load (No Optimizer)
// ============================================================================

void save_bert_model_only(const std::filesystem::path& path, const models::BaseTransformer& model) {
    fmt::print("Saving BERT model (parameters only) to: {}\n", path.string());

    MsgPackFile file;

    // Save metadata
    file.put("version", "1.0");
    file.put("model_only", true);
    file.put("timestamp", std::string_view{get_timestamp()});

    // Save model parameters
    write_module(file, "model", &model);

    // Serialize to disk
    file.serialize(path.string());

    fmt::print("  Model saved successfully\n");
}

void load_bert_model_only(const std::filesystem::path& path, models::BaseTransformer& model) {
    fmt::print("Loading BERT model (parameters only) from: {}\n", path.string());

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Model file does not exist: " + path.string());
    }

    MsgPackFile file;
    file.deserialize(path.string());

    // Verify this is a model-only checkpoint
    bool model_only = false;
    file.get("model_only", model_only);

    if (!model_only) {
        fmt::print("  Warning: Loading full checkpoint but ignoring optimizer state\n");
    }

    // Load model parameters
    read_module(file, "model", &model);

    fmt::print("  Model loaded successfully\n");
}

// ============================================================================
// Checkpoint Management
// ============================================================================

std::vector<std::pair<std::filesystem::path, BertTrainingState>> list_checkpoints(
    const std::filesystem::path& directory) {
    std::vector<std::pair<std::filesystem::path, BertTrainingState>> checkpoints;

    if (!std::filesystem::exists(directory)) {
        return checkpoints;
    }

    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".msgpack") {
            try {
                MsgPackFile file;
                file.deserialize(entry.path().string());

                BertTrainingState state;
                file.get("global_step", state.global_step);
                file.get("epoch", state.epoch);
                file.get("best_loss", state.best_loss);
                file.get("model_type", state.model_type);
                file.get("timestamp", state.timestamp);

                checkpoints.emplace_back(entry.path(), state);
            } catch (const std::exception& e) {
                fmt::print("  Warning: Failed to read checkpoint {}: {}\n", entry.path().string(), e.what());
            }
        }
    }

    return checkpoints;
}

std::filesystem::path find_best_checkpoint(const std::filesystem::path& directory) {
    auto checkpoints = list_checkpoints(directory);

    if (checkpoints.empty()) {
        return {};
    }

    // Find checkpoint with lowest best_loss
    auto best_it = std::min_element(checkpoints.begin(), checkpoints.end(), [](const auto& a, const auto& b) {
        return a.second.best_loss < b.second.best_loss;
    });

    fmt::print("Found best checkpoint: {}\n", best_it->first.string());
    fmt::print("  Best loss: {:.4f}\n", best_it->second.best_loss);
    fmt::print("  Step: {}\n", best_it->second.global_step);

    return best_it->first;
}

}  // namespace ttml::serialization
