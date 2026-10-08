"""Unit tests for PhishGuard model components."""

import pytest
from src.config import ModelConfig, DataConfig


def test_model_config_defaults():
    config = ModelConfig()
    assert config.model_name == "microsoft/deberta-v3-base"
    assert config.num_labels == 2
    assert config.max_length == 512


def test_data_config_splits():
    config = DataConfig()
    total = config.train_split + config.val_split + config.test_split
    assert abs(total - 1.0) < 1e-6
