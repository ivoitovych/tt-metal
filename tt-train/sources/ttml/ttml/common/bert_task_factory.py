# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Factory for creating BERT task models from YAML configurations.
"""

import yaml
import ttml


class BertTaskFactory:
    """Unified factory for BERT task models."""

    @staticmethod
    def create_from_yaml(config_path: str, task_type: str):
        """
        Create BERT task model from YAML configuration.

        Args:
            config_path: Path to YAML configuration file
            task_type: One of: "sequence_classification", "token_classification",
                      "question_answering", "masked_lm", "pretraining"

        Returns:
            BERT task model instance
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        bert_config = BertTaskFactory._create_bert_config(config["bert_config"])

        if task_type == "sequence_classification":
            sc_config = ttml.models.bert.SequenceClassificationConfig()
            sc_config.bert_config = bert_config
            sc_config.num_labels = config.get("num_labels", 2)
            sc_config.classifier_dropout = config.get("classifier_dropout", 0.1)
            return ttml.models.bert.create_for_sequence_classification(sc_config)

        elif task_type == "token_classification":
            tc_config = ttml.models.bert.TokenClassificationConfig()
            tc_config.bert_config = bert_config
            tc_config.num_labels = config["num_labels"]  # Required
            tc_config.classifier_dropout = config.get("classifier_dropout", 0.1)
            return ttml.models.bert.create_for_token_classification(tc_config)

        elif task_type == "question_answering":
            qa_config = ttml.models.bert.QuestionAnsweringConfig()
            qa_config.bert_config = bert_config
            return ttml.models.bert.create_for_question_answering(qa_config)

        elif task_type == "masked_lm":
            mlm_config = ttml.models.bert.MaskedLMConfig()
            mlm_config.bert_config = bert_config
            mlm_config.tie_word_embeddings = config.get("tie_word_embeddings", True)
            return ttml.models.bert.create_for_masked_lm(mlm_config)

        elif task_type == "pretraining":
            pt_config = ttml.models.bert.PreTrainingConfig()
            pt_config.bert_config = bert_config
            pt_config.tie_word_embeddings = config.get("tie_word_embeddings", True)
            pt_config.mlm_loss_weight = config.get("mlm_loss_weight", 1.0)
            pt_config.nsp_loss_weight = config.get("nsp_loss_weight", 1.0)
            return ttml.models.bert.create_for_pretraining(pt_config)

        else:
            raise ValueError(
                f"Unknown task type: {task_type}. "
                f"Must be one of: sequence_classification, token_classification, "
                f"question_answering, masked_lm, pretraining"
            )

    @staticmethod
    def _create_bert_config(cfg: dict):
        """Create BertConfig from dictionary."""
        bert_cfg = ttml.models.bert.BertConfig()
        bert_cfg.vocab_size = cfg.get("vocab_size", 30522)
        bert_cfg.max_sequence_length = cfg.get("max_sequence_length", 512)
        bert_cfg.embedding_dim = cfg.get("embedding_dim", 768)
        bert_cfg.intermediate_size = cfg.get("intermediate_size", 3072)
        bert_cfg.num_heads = cfg.get("num_heads", 12)
        bert_cfg.num_blocks = cfg.get("num_blocks", 12)
        bert_cfg.dropout_prob = cfg.get("dropout_prob", 0.1)
        bert_cfg.layer_norm_eps = cfg.get("layer_norm_eps", 1e-12)
        bert_cfg.use_token_type_embeddings = cfg.get("use_token_type_embeddings", True)
        bert_cfg.type_vocab_size = cfg.get("type_vocab_size", 2)

        # Parse runner_type
        runner_str = cfg.get("runner_type", "default").lower()
        if runner_str == "memory_efficient":
            bert_cfg.runner_type = ttml.models.RunnerType.MemoryEfficient
        else:
            bert_cfg.runner_type = ttml.models.RunnerType.Default

        return bert_cfg


def create_bert_model(config_path: str, task_type: str):
    """
    Convenience function to create BERT model from config.

    Args:
        config_path: Path to YAML configuration file
        task_type: Task type (see BertTaskFactory.create_from_yaml for options)

    Returns:
        BERT task model instance
    """
    return BertTaskFactory.create_from_yaml(config_path, task_type)
