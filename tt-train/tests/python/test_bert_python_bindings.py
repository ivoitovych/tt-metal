# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for BERT Python bindings.

Tests all aspects of the BERT Python API including:
- BertConfig creation and parameter access
- BERT model instantiation via create() and constructor
- Forward pass execution
- Weight loading from safetensors
- Error handling and validation
"""

import numpy as np
import pytest
import os
import sys
from pathlib import Path

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402


class TestBertConfig:
    """Test BertConfig Python bindings."""

    def test_config_creation(self):
        """Test that BertConfig can be created."""
        config = ttml.models.bert.BertConfig()
        assert config is not None

    def test_config_default_values(self):
        """Test BertConfig default values."""
        config = ttml.models.bert.BertConfig()
        # Should have reasonable defaults
        assert hasattr(config, "vocab_size")
        assert hasattr(config, "embedding_dim")
        assert hasattr(config, "num_heads")

    def test_config_vocab_size(self):
        """Test vocab_size parameter."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 30522
        assert config.vocab_size == 30522

    def test_config_max_sequence_length(self):
        """Test max_sequence_length parameter."""
        config = ttml.models.bert.BertConfig()
        config.max_sequence_length = 512
        assert config.max_sequence_length == 512

    def test_config_embedding_dim(self):
        """Test embedding_dim parameter."""
        config = ttml.models.bert.BertConfig()
        config.embedding_dim = 768
        assert config.embedding_dim == 768

    def test_config_intermediate_size(self):
        """Test intermediate_size parameter."""
        config = ttml.models.bert.BertConfig()
        config.intermediate_size = 3072
        assert config.intermediate_size == 3072

    def test_config_num_heads(self):
        """Test num_heads parameter."""
        config = ttml.models.bert.BertConfig()
        config.num_heads = 12
        assert config.num_heads == 12

    def test_config_num_blocks(self):
        """Test num_blocks parameter."""
        config = ttml.models.bert.BertConfig()
        config.num_blocks = 12
        assert config.num_blocks == 12

    def test_config_dropout_prob(self):
        """Test dropout_prob parameter."""
        config = ttml.models.bert.BertConfig()
        config.dropout_prob = 0.1
        assert abs(config.dropout_prob - 0.1) < 1e-6

    def test_config_layer_norm_eps(self):
        """Test layer_norm_eps parameter."""
        config = ttml.models.bert.BertConfig()
        config.layer_norm_eps = 1e-12
        assert abs(config.layer_norm_eps - 1e-12) < 1e-15

    def test_config_use_token_type_embeddings(self):
        """Test use_token_type_embeddings parameter."""
        config = ttml.models.bert.BertConfig()
        config.use_token_type_embeddings = True
        assert config.use_token_type_embeddings is True
        config.use_token_type_embeddings = False
        assert config.use_token_type_embeddings is False

    def test_config_use_pooler(self):
        """Test use_pooler parameter."""
        config = ttml.models.bert.BertConfig()
        config.use_pooler = True
        assert config.use_pooler is True
        config.use_pooler = False
        assert config.use_pooler is False

    def test_config_runner_type(self):
        """Test runner_type parameter."""
        config = ttml.models.bert.BertConfig()
        # Should be able to set runner_type
        config.runner_type = ttml.models.RunnerType.Default
        assert config.runner_type == ttml.models.RunnerType.Default

    def test_config_all_parameters_together(self):
        """Test setting all config parameters together."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 30522
        config.max_sequence_length = 512
        config.embedding_dim = 768
        config.intermediate_size = 3072
        config.num_heads = 12
        config.num_blocks = 12
        config.dropout_prob = 0.1
        config.layer_norm_eps = 1e-12
        config.use_token_type_embeddings = True
        config.use_pooler = False

        assert config.vocab_size == 30522
        assert config.max_sequence_length == 512
        assert config.embedding_dim == 768
        assert config.intermediate_size == 3072
        assert config.num_heads == 12
        assert config.num_blocks == 12
        assert abs(config.dropout_prob - 0.1) < 1e-6
        assert abs(config.layer_norm_eps - 1e-12) < 1e-15
        assert config.use_token_type_embeddings is True
        assert config.use_pooler is False


class TestBertModel:
    """Test BERT model Python bindings."""

    def test_create_function_exists(self):
        """Test that create() function exists."""
        assert hasattr(ttml.models.bert, "create")

    def test_bert_class_exists(self):
        """Test that Bert class exists."""
        assert hasattr(ttml.models.bert, "Bert")

    def test_model_creation_via_create(self):
        """Test BERT model creation via create() factory function."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 2
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        assert bert is not None

    def test_model_creation_via_constructor(self):
        """Test BERT model creation via Bert constructor."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 2
        config.dropout_prob = 0.0

        bert = ttml.models.bert.Bert(config)
        assert bert is not None

    def test_model_inherits_from_base_transformer(self):
        """Test that BERT model inherits from BaseTransformer."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        # Should have BaseTransformer methods
        assert hasattr(bert, "parameters")
        assert hasattr(bert, "load_from_safetensors")

    def test_model_has_parameters(self):
        """Test that model has parameters() method."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        params = bert.parameters()
        assert params is not None
        # Should be a NamedParameters (map-like)
        assert isinstance(params, ttml.NamedParameters)

    def test_model_has_load_model_from_safetensors(self):
        """Test that model has load_model_from_safetensors() method."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        assert hasattr(bert, "load_model_from_safetensors")

    @pytest.mark.skip(reason="Forward pass test has hardware-specific issues; covered by C++ tests")
    def test_model_forward_pass(self):
        """Test BERT model forward pass."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0
        config.use_token_type_embeddings = True

        bert = ttml.models.bert.create(config)

        # Create input tensors
        batch_size = 1
        seq_len = 128
        input_ids = np.ones((batch_size, 1, seq_len, 1), dtype=np.float32)
        token_type_ids = np.zeros((batch_size, 1, seq_len, 1), dtype=np.float32)

        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids)
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids)

        # Forward pass
        output = bert(input_ids_ttml, token_type_ids_ttml)
        assert output is not None

        # Check output shape
        output_np = output.to_numpy()
        assert output_np.shape[0] == batch_size
        assert output_np.shape[2] == seq_len
        assert output_np.shape[3] == config.embedding_dim

    @pytest.mark.skip(reason="Forward pass test has hardware-specific issues; covered by C++ tests")
    def test_model_output_no_nan_inf(self):
        """Test that model output contains no NaN or Inf."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)

        # Create input tensors
        input_ids = np.ones((1, 1, 128, 1), dtype=np.float32)
        token_type_ids = np.zeros((1, 1, 128, 1), dtype=np.float32)

        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids)
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids)

        # Forward pass
        output = bert(input_ids_ttml, token_type_ids_ttml)
        output_np = output.to_numpy()

        assert not np.isnan(output_np).any(), "Output contains NaN"
        assert not np.isinf(output_np).any(), "Output contains Inf"

    def test_model_with_different_configs(self):
        """Test model creation with various valid configurations."""
        configs = [
            # Small model
            {"vocab_size": 1000, "embedding_dim": 128, "num_heads": 4, "num_blocks": 1},
            # Medium model
            {"vocab_size": 10000, "embedding_dim": 256, "num_heads": 8, "num_blocks": 2},
            # With pooler
            {"vocab_size": 1000, "embedding_dim": 128, "num_heads": 4, "num_blocks": 1, "use_pooler": True},
            # Without token type embeddings
            {
                "vocab_size": 1000,
                "embedding_dim": 128,
                "num_heads": 4,
                "num_blocks": 1,
                "use_token_type_embeddings": False,
            },
        ]

        for cfg_dict in configs:
            config = ttml.models.bert.BertConfig()
            config.vocab_size = cfg_dict.get("vocab_size", 1000)
            config.max_sequence_length = cfg_dict.get("max_sequence_length", 128)
            config.embedding_dim = cfg_dict.get("embedding_dim", 256)
            config.intermediate_size = cfg_dict.get("intermediate_size", 512)
            config.num_heads = cfg_dict.get("num_heads", 8)
            config.num_blocks = cfg_dict.get("num_blocks", 1)
            config.dropout_prob = cfg_dict.get("dropout_prob", 0.0)
            config.use_pooler = cfg_dict.get("use_pooler", False)
            config.use_token_type_embeddings = cfg_dict.get("use_token_type_embeddings", True)

            bert = ttml.models.bert.create(config)
            assert bert is not None


class TestBertWeightLoading:
    """Test BERT weight loading functionality."""

    def test_load_model_from_safetensors_method_callable(self):
        """Test that load_model_from_safetensors is callable."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        assert callable(bert.load_model_from_safetensors)

    def test_load_model_from_safetensors_nonexistent_file(self):
        """Test that loading from non-existent file raises error."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)

        with pytest.raises(Exception):  # Should raise some error
            bert.load_model_from_safetensors("/nonexistent/path/model.safetensors")


class TestBertIntegration:
    """Integration tests for BERT Python API."""

    @pytest.mark.skip(reason="Forward pass test has hardware-specific issues; covered by C++ tests")
    def test_end_to_end_workflow(self):
        """Test complete workflow: config -> create -> forward."""
        # Create config
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0
        config.use_token_type_embeddings = True

        # Create model
        bert = ttml.models.bert.create(config)

        # Prepare inputs
        input_ids = np.random.randint(0, config.vocab_size, (1, 1, 128, 1)).astype(np.float32)
        token_type_ids = np.zeros((1, 1, 128, 1), dtype=np.float32)

        input_ids_ttml = ttml.autograd.Tensor.from_numpy(input_ids)
        token_type_ids_ttml = ttml.autograd.Tensor.from_numpy(token_type_ids)

        # Forward pass
        output = bert(input_ids_ttml, token_type_ids_ttml)
        output_np = output.to_numpy()

        # Validate
        assert output_np.shape[0] == 1
        assert output_np.shape[2] == 128
        assert output_np.shape[3] == config.embedding_dim
        assert not np.isnan(output_np).any()
        assert not np.isinf(output_np).any()

    def test_parameters_accessible(self):
        """Test that model parameters are accessible."""
        config = ttml.models.bert.BertConfig()
        config.vocab_size = 1000
        config.max_sequence_length = 128
        config.embedding_dim = 256
        config.intermediate_size = 512
        config.num_heads = 8
        config.num_blocks = 1
        config.dropout_prob = 0.0

        bert = ttml.models.bert.create(config)
        params = bert.parameters()

        # Should have embedding parameters
        assert "bert/token_embeddings/weight" in params
        assert "bert/position_embeddings/weight" in params

        # Should have layer parameters
        assert "bert/bert_block_0/attention/self_attention/qkv_linear/weight" in params


if __name__ == "__main__":
    # Run with: python test_bert_python_bindings.py
    pytest.main([__file__, "-v", "-s"])
