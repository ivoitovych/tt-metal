// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "core/tt_tensor_utils.hpp"
#include "serialization/safetensors.hpp"
#include "serialization/serializable.hpp"

namespace ttml::models::bert {

/**
 * @brief Helper to load task head weights from safetensors
 *
 * This function attempts to load head-specific weights from a safetensors file.
 * If the weights don't exist (e.g., loading a pretrained base model for fine-tuning),
 * it will skip them silently.
 *
 * @param model_path Path to directory containing safetensors files
 * @param parameters Model parameters dictionary
 * @param weight_mapping Map of HF weight names to internal parameter paths
 * @param task_name Name of task for logging (e.g., "SequenceClassification")
 */
inline void load_task_head_weights(
    const std::filesystem::path& model_path,
    serialization::NamedParameters& parameters,
    const std::map<std::string, std::string>& weight_mapping,
    const std::string& task_name) {
    bool head_weights_found = false;

    for (const auto& entry : std::filesystem::directory_iterator(model_path)) {
        if (entry.path().extension() != ".safetensors") {
            continue;
        }

        auto path = entry.path();

        serialization::SafetensorSerialization::TensorCallback loading_callback =
            [&parameters, &weight_mapping, &head_weights_found](
                const serialization::SafetensorSerialization::TensorInfo& info, std::span<const std::byte> bytes) {
                // Check if this weight is in our mapping
                auto it = weight_mapping.find(info.name);
                if (it == weight_mapping.end()) {
                    return true;  // Skip this tensor, continue processing
                }

                const std::string& internal_name = it->second;

                // Find the parameter
                auto param_it = parameters.find(internal_name);
                if (param_it == parameters.end()) {
                    fmt::print("  Warning: Parameter {} not found in model\n", internal_name);
                    return true;
                }

                // Load the weight
                auto float_vec = serialization::SafetensorSerialization::bytes_to_float_vec(bytes, info.dtype);

                auto param = param_it->second;
                param->set_value(
                    core::from_vector(float_vec, param->get_value().logical_shape(), param->get_value().device()));

                fmt::print("  Loaded {} -> {}\n", info.name, internal_name);
                head_weights_found = true;

                return true;
            };

        try {
            serialization::SafetensorSerialization::visit_safetensors_file(path, loading_callback);
        } catch (const std::exception& e) {
            fmt::print("  Warning: Error loading from {}: {}\n", path.string(), e.what());
        }
    }

    if (!head_weights_found) {
        fmt::print("  Note: No {} head weights found in checkpoint.\n", task_name);
        fmt::print("        Head weights are randomly initialized.\n");
        fmt::print("        This is expected when fine-tuning from a base model.\n");
    }
}

}  // namespace ttml::models::bert
