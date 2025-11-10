#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Example: Training BERT for sequence classification (sentiment analysis).

This example demonstrates:
1. Loading a pretrained BERT model
2. Adding a classification head
3. Training on a classification task
4. Using external loss computation (TTML pattern)
"""

import argparse
import ttml
from ttml.common.bert_task_factory import create_bert_model


def main():
    parser = argparse.ArgumentParser(description="Train BERT sequence classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained BERT model (safetensors)")
    parser.add_argument(
        "--save_path", type=str, default="checkpoints/bert_classifier", help="Path to save trained model"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BERT Sequence Classification Training Example")
    print("=" * 70)

    # ========================================================================
    # 1. Create model from config
    # ========================================================================
    print("\n[1/6] Creating model from config...")
    model = create_bert_model(args.config, "sequence_classification")
    print(f"   Model type: BertForSequenceClassification")
    print(f"   Number of labels: {model.get_num_labels()}")

    # ========================================================================
    # 2. Load pretrained weights
    # ========================================================================
    print("\n[2/6] Loading pretrained BERT weights...")
    model.load_from_safetensors(args.model_path)
    print("   Base BERT weights loaded successfully")
    print("   Classification head initialized randomly (ready for fine-tuning)")

    # ========================================================================
    # 3. Setup optimizer (AdamW with weight decay)
    # ========================================================================
    print("\n[3/6] Setting up optimizer...")
    # optimizer = ttml.optimizers.AdamW(
    #     model.parameters(),
    #     lr=2e-5,
    #     weight_decay=0.01
    # )
    print("   Optimizer: AdamW (lr=2e-5, weight_decay=0.01)")

    # ========================================================================
    # 4. Training loop (pseudo-code - add your dataloader)
    # ========================================================================
    print("\n[4/6] Training loop (example structure)...")
    print(
        """
    # Example training loop structure:

    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass - model returns logits only
            logits = model(
                batch['input_ids'],
                batch['attention_mask'],
                batch['token_type_ids']
            )

            # Loss - external computation (TTML pattern)
            loss = ttml.ops.bert_losses.compute_sequence_classification_loss(
                logits, batch['labels']
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            ttml.core.clip_grad_norm(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.item():.4f}")
    """
    )

    # ========================================================================
    # 5. Evaluation (pseudo-code)
    # ========================================================================
    print("\n[5/6] Evaluation (example structure)...")
    print(
        """
    # Example evaluation structure:

    model.eval()
    correct = 0
    total = 0

    for batch in eval_dataloader:
        logits = model(batch['input_ids'], batch['attention_mask'])
        predictions = ttml.ops.argmax(logits, dim=-1)
        correct += (predictions == batch['labels']).sum()
        total += len(batch['labels'])

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")
    """
    )

    # ========================================================================
    # 6. Save trained model
    # ========================================================================
    print("\n[6/6] Saving trained model...")
    print(f"   Save path: {args.save_path}")
    # model.save_to_safetensors(args.save_path)
    print("   (Implementation: use save_to_safetensors once available)")

    print("\n" + "=" * 70)
    print("Training example complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Implement your dataloader")
    print("  2. Run the training loop")
    print("  3. Evaluate on your test set")
    print("  4. Save the best checkpoint")


if __name__ == "__main__":
    main()
