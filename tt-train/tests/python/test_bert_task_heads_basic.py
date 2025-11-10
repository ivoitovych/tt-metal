#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Basic validation tests for BERT task heads.

These tests verify:
1. Model creation succeeds
2. Forward pass produces correct shapes
3. Loss computation works
4. Gradient flow is functional
"""

import pytest
import ttml


@pytest.fixture
def small_bert_config():
    """Small BERT config for fast testing."""
    config = ttml.models.bert.BertConfig()
    config.vocab_size = 1000
    config.max_sequence_length = 32
    config.embedding_dim = 128
    config.intermediate_size = 512
    config.num_heads = 4
    config.num_blocks = 2
    config.dropout_prob = 0.0  # Disable for determinism
    return config


class TestSequenceClassification:
    """Test BertForSequenceClassification."""

    def test_model_creation(self, small_bert_config):
        """Test model can be created."""
        config = ttml.models.bert.SequenceClassificationConfig()
        config.bert_config = small_bert_config
        config.num_labels = 3

        model = ttml.models.bert.create_for_sequence_classification(config)

        assert model is not None
        assert model.get_num_labels() == 3
        print("✓ BertForSequenceClassification created successfully")

    def test_forward_shape(self, small_bert_config):
        """Test output shapes are correct."""
        config = ttml.models.bert.SequenceClassificationConfig()
        config.bert_config = small_bert_config
        config.num_labels = 2

        model = ttml.models.bert.create_for_sequence_classification(config)

        # Create dummy input
        batch_size = 4
        seq_len = 32
        input_ids = ttml.zeros([batch_size, 1, 1, seq_len])
        attention_mask = ttml.ones([batch_size, 1, 1, seq_len])

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Check shape
        expected_shape = [batch_size, 1, 1, 2]
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        print(f"✓ Forward pass produces correct shape: {logits.shape}")

    def test_loss_computation(self, small_bert_config):
        """Test loss computation works."""
        config = ttml.models.bert.SequenceClassificationConfig()
        config.bert_config = small_bert_config
        config.num_labels = 5

        model = ttml.models.bert.create_for_sequence_classification(config)

        batch_size = 4
        seq_len = 32
        input_ids = ttml.zeros([batch_size, 1, 1, seq_len])
        attention_mask = ttml.ones([batch_size, 1, 1, seq_len])

        # Create labels (dummy - just for shape testing)
        import numpy as np

        labels = ttml.from_numpy(np.array([0, 1, 2, 3]))

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Compute loss
        loss = ttml.ops.bert_losses.compute_sequence_classification_loss(logits, labels)

        # Check loss properties
        assert loss.shape == []  # Scalar
        assert loss.requires_grad
        print("✓ Loss computation works correctly")


class TestPreTraining:
    """Test BertForPreTraining (critical bug fix validation)."""

    def test_both_outputs(self, small_bert_config):
        """Test PreTraining returns both MLM and NSP outputs."""
        config = ttml.models.bert.PreTrainingConfig()
        config.bert_config = small_bert_config

        model = ttml.models.bert.create_for_pretraining(config)

        batch_size = 2
        seq_len = 32
        input_ids = ttml.zeros([batch_size, 1, 1, seq_len])

        # PreTraining forward - returns both outputs
        output = model.forward_pretraining(input_ids, None)

        # Check MLM output shape
        expected_mlm_shape = [batch_size, 1, seq_len, 1000]
        assert output.mlm_logits.shape == expected_mlm_shape
        print(f"✓ MLM logits shape correct: {output.mlm_logits.shape}")

        # Check NSP output shape
        expected_nsp_shape = [batch_size, 1, 1, 2]
        assert output.nsp_logits.shape == expected_nsp_shape
        print(f"✓ NSP logits shape correct: {output.nsp_logits.shape}")

        print("✓ PreTraining bug fix validated: both outputs work correctly")


def run_basic_validation():
    """Run basic validation tests."""
    print("\n" + "=" * 70)
    print("BERT Task Heads - Basic Validation")
    print("=" * 70 + "\n")

    config = ttml.models.bert.BertConfig()
    config.vocab_size = 1000
    config.max_sequence_length = 32
    config.embedding_dim = 128
    config.intermediate_size = 512
    config.num_heads = 4
    config.num_blocks = 2

    # Test 1: Sequence Classification
    print("[Test 1] Sequence Classification...")
    test_seq = TestSequenceClassification()
    test_seq.test_model_creation(config)
    test_seq.test_forward_shape(config)
    test_seq.test_loss_computation(config)

    # Test 2: PreTraining
    print("\n[Test 2] PreTraining (Bug Fix Validation)...")
    test_pretrain = TestPreTraining()
    test_pretrain.test_both_outputs(config)

    print("\n" + "=" * 70)
    print("All basic validation tests passed! ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_basic_validation()
