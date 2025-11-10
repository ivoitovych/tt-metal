// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT Sequence Classification Training Example (C++)
 *
 * This example demonstrates:
 * 1. Loading a BERT model from YAML configuration
 * 2. Creating a sequence classification task model
 * 3. Loading pretrained weights from SafeTensors
 * 4. Setting up optimizer (AdamW)
 * 5. Training loop with external loss computation
 * 6. Gradient clipping and checkpoint saving
 * 7. Evaluation on validation set
 *
 * Usage:
 *   ./train_bert_classifier \
 *     --config configs/bert_sequence_classification.yaml \
 *     --model_path /path/to/bert-base-uncased \
 *     --data_path /path/to/dataset \
 *     --output_dir checkpoints/
 */

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert_tasks.hpp"
#include "ops/bert_losses.hpp"
// TODO: Include optimizer when available
// #include "optimizers/adamw.hpp"

using namespace ttml;

// ============================================================================
// Command Line Arguments
// ============================================================================

struct TrainingArgs {
    std::string config_path = "configs/bert_sequence_classification.yaml";
    std::string model_path;  // Path to pretrained BERT (SafeTensors)
    std::string data_path;   // Path to training data
    std::string output_dir = "checkpoints/";

    // Training hyperparameters (override config)
    float learning_rate = 2e-5F;
    float weight_decay = 0.01F;
    uint32_t batch_size = 16;
    uint32_t num_epochs = 3;
    float max_grad_norm = 1.0F;

    // Logging
    uint32_t log_interval = 100;
    uint32_t eval_interval = 500;
    uint32_t save_interval = 1000;
};

// ============================================================================
// Dummy Dataloader (Replace with actual implementation)
// ============================================================================

struct Batch {
    autograd::TensorPtr input_ids;       // [B, 1, 1, S]
    autograd::TensorPtr attention_mask;  // [B, 1, 1, S]
    autograd::TensorPtr token_type_ids;  // [B, 1, 1, S]
    autograd::TensorPtr labels;          // [B]
};

class DummyDataLoader {
public:
    explicit DummyDataLoader(uint32_t num_batches, uint32_t batch_size, uint32_t seq_len, uint32_t num_labels) :
        m_num_batches(num_batches), m_batch_size(batch_size), m_seq_len(seq_len), m_num_labels(num_labels) {
        fmt::print(
            "Created dummy dataloader: {} batches, batch_size={}, seq_len={}\n", num_batches, batch_size, seq_len);
    }

    struct Iterator {
        const DummyDataLoader* loader;
        uint32_t current_batch;

        Batch operator*() const {
            // TODO: Load actual data
            // For now, return dummy tensors
            Batch batch;

            // Create dummy input_ids
            std::vector<uint32_t> input_ids(loader->m_batch_size * loader->m_seq_len, 100);
            batch.input_ids = autograd::create_tensor(core::from_vector(
                input_ids,
                core::create_shape({loader->m_batch_size, 1, 1, loader->m_seq_len}),
                autograd::ctx().get_device()));

            // Create dummy attention_mask
            std::vector<float> attention_mask(loader->m_batch_size * loader->m_seq_len, 1.0F);
            batch.attention_mask = autograd::create_tensor(core::from_vector(
                attention_mask,
                core::create_shape({loader->m_batch_size, 1, 1, loader->m_seq_len}),
                autograd::ctx().get_device()));

            // Create dummy token_type_ids
            std::vector<uint32_t> token_type_ids(loader->m_batch_size * loader->m_seq_len, 0);
            batch.token_type_ids = autograd::create_tensor(core::from_vector(
                token_type_ids,
                core::create_shape({loader->m_batch_size, 1, 1, loader->m_seq_len}),
                autograd::ctx().get_device()));

            // Create dummy labels
            std::vector<uint32_t> labels(loader->m_batch_size, 0);
            batch.labels = autograd::create_tensor(
                core::from_vector(labels, core::create_shape({loader->m_batch_size}), autograd::ctx().get_device()));

            return batch;
        }

        Iterator& operator++() {
            ++current_batch;
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return current_batch != other.current_batch;
        }
    };

    Iterator begin() const {
        return Iterator{this, 0};
    }

    Iterator end() const {
        return Iterator{this, m_num_batches};
    }

private:
    uint32_t m_num_batches;
    uint32_t m_batch_size;
    uint32_t m_seq_len;
    uint32_t m_num_labels;
};

// ============================================================================
// Training Functions
// ============================================================================

void train_epoch(
    models::bert::BertForSequenceClassification& model,
    DummyDataLoader& dataloader,
    // optimizers::AdamW& optimizer,  // TODO: Enable when optimizer available
    const TrainingArgs& args,
    uint32_t epoch,
    uint32_t& global_step) {
    fmt::print("\n[Epoch {}/{}] Training...\n", epoch + 1, args.num_epochs);

    // model.train();  // TODO: Enable training mode

    float total_loss = 0.0F;
    uint32_t num_batches = 0;

    for (auto batch : dataloader) {
        // Reset graph for each batch
        autograd::ctx().reset_graph();

        // TODO: Zero gradients
        // optimizer.zero_grad();

        // Forward pass
        auto logits = model(batch.input_ids, batch.attention_mask, batch.token_type_ids);

        // Compute loss (external - TTML pattern)
        auto loss = ops::bert_losses::compute_sequence_classification_loss(logits, batch.labels);

        // Get loss value for logging
        auto loss_value = core::to_scalar(loss);
        total_loss += loss_value;
        ++num_batches;

        // Backward pass
        loss->backward();

        // Gradient clipping
        // TODO: Implement gradient clipping
        // auto params = model.parameters();
        // float grad_norm = core::clip_grad_norm(params, args.max_grad_norm);

        // Optimizer step
        // TODO: Enable when optimizer available
        // optimizer.step();

        // Logging
        if (global_step % args.log_interval == 0) {
            float avg_loss = total_loss / static_cast<float>(num_batches);
            fmt::print("  Step {}: Loss = {:.4f}\n", global_step, loss_value);
        }

        // Evaluation
        if (global_step % args.eval_interval == 0) {
            fmt::print("  Evaluation at step {}...\n", global_step);
            // TODO: Implement evaluation
        }

        // Checkpoint saving
        if (global_step % args.save_interval == 0) {
            fmt::print("  Saving checkpoint at step {}...\n", global_step);
            // TODO: Implement checkpoint saving
            // std::string checkpoint_path = args.output_dir + "/checkpoint_step_" + std::to_string(global_step) +
            // ".msgpack"; serialization::save_training_state(checkpoint_path, model, optimizer, global_step, avg_loss);
        }

        ++global_step;
    }

    float epoch_avg_loss = total_loss / static_cast<float>(num_batches);
    fmt::print("[Epoch {}/{}] Average Loss: {:.4f}\n", epoch + 1, args.num_epochs, epoch_avg_loss);
}

void evaluate(models::bert::BertForSequenceClassification& model, DummyDataLoader& dataloader) {
    fmt::print("\n[Evaluation] Running...\n");

    // model.eval();  // TODO: Enable eval mode

    uint32_t correct = 0;
    uint32_t total = 0;

    for (auto batch : dataloader) {
        autograd::ctx().reset_graph();

        // Forward pass
        auto logits = model(batch.input_ids, batch.attention_mask, batch.token_type_ids);

        // TODO: Get predictions (argmax)
        // auto predictions = ops::argmax(logits, -1);

        // TODO: Compare with labels
        // correct += count_correct(predictions, batch.labels);
        // total += batch_size;

        total += 1;  // Placeholder
    }

    float accuracy = static_cast<float>(correct) / static_cast<float>(total);
    fmt::print("[Evaluation] Accuracy: {:.2f}%\n", accuracy * 100.0F);
}

// ============================================================================
// Main Training Function
// ============================================================================

int main(int argc, char** argv) {
    // TODO: Parse command line arguments
    TrainingArgs args;

    fmt::print("=" * 80 + "\n");
    fmt::print("BERT Sequence Classification Training Example\n");
    fmt::print("=" * 80 + "\n\n");

    // ========================================================================
    // 1. Load Configuration
    // ========================================================================
    fmt::print("[1/7] Loading configuration from: {}\n", args.config_path);

    YAML::Node config_yaml = YAML::LoadFile(args.config_path);
    auto task_config = models::bert::read_sequence_classification_config(config_yaml);

    fmt::print("  Model: BERT-base\n");
    fmt::print("  Num labels: {}\n", task_config.num_labels);
    fmt::print("  Dropout: {}\n", task_config.classifier_dropout);

    // ========================================================================
    // 2. Create Model
    // ========================================================================
    fmt::print("\n[2/7] Creating BertForSequenceClassification...\n");

    auto model = models::bert::create_for_sequence_classification(task_config);
    fmt::print("  Model created successfully\n");

    // ========================================================================
    // 3. Load Pretrained Weights
    // ========================================================================
    if (!args.model_path.empty()) {
        fmt::print("\n[3/7] Loading pretrained weights from: {}\n", args.model_path);
        model->load_from_safetensors(args.model_path);
        fmt::print("  Weights loaded successfully\n");
    } else {
        fmt::print("\n[3/7] Skipping weight loading (training from scratch)\n");
    }

    // ========================================================================
    // 4. Setup Optimizer
    // ========================================================================
    fmt::print("\n[4/7] Setting up optimizer...\n");
    // TODO: Enable when optimizer available
    // auto optimizer = optimizers::AdamW(
    //     model->parameters(),
    //     optimizers::AdamWConfig{
    //         .lr = args.learning_rate,
    //         .weight_decay = args.weight_decay
    //     }
    // );
    fmt::print("  Optimizer: AdamW (lr={}, weight_decay={})\n", args.learning_rate, args.weight_decay);
    fmt::print("  (Optimizer implementation pending)\n");

    // ========================================================================
    // 5. Setup Data Loaders
    // ========================================================================
    fmt::print("\n[5/7] Setting up data loaders...\n");

    // TODO: Load actual dataset
    DummyDataLoader train_loader(100, args.batch_size, 128, task_config.num_labels);
    DummyDataLoader eval_loader(20, args.batch_size, 128, task_config.num_labels);

    fmt::print("  (Using dummy data - implement actual dataloader)\n");

    // ========================================================================
    // 6. Training Loop
    // ========================================================================
    fmt::print("\n[6/7] Starting training...\n");
    fmt::print("  Epochs: {}\n", args.num_epochs);
    fmt::print("  Batch size: {}\n", args.batch_size);

    uint32_t global_step = 0;

    for (uint32_t epoch = 0; epoch < args.num_epochs; ++epoch) {
        // train_epoch(*model, train_loader, optimizer, args, epoch, global_step);
        fmt::print("\n[Epoch {}/{}] Skipping - optimizer not implemented\n", epoch + 1, args.num_epochs);

        // Evaluate after each epoch
        // evaluate(*model, eval_loader);
    }

    // ========================================================================
    // 7. Save Final Model
    // ========================================================================
    fmt::print("\n[7/7] Saving final model...\n");
    std::string final_path = args.output_dir + "/final_model.msgpack";
    // TODO: Implement model saving
    // model->save_to_msgpack(final_path);
    fmt::print("  (Model saving not implemented yet)\n");

    fmt::print("\n" + std::string(80, '=') + "\n");
    fmt::print("Training complete!\n");
    fmt::print("=" * 80 + "\n");

    fmt::print("\nNext steps:\n");
    fmt::print("  1. Implement actual dataloader for your dataset\n");
    fmt::print("  2. Enable optimizer (AdamW) when available\n");
    fmt::print("  3. Implement evaluation metrics\n");
    fmt::print("  4. Enable checkpoint saving/loading\n");

    return 0;
}
