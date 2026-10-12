"""Unit tests for feature extraction and ensemble logic."""

import pytest
import numpy as np
from src.features import EmailFeatureExtractor


def test_shannon_entropy():
    # Empty string should yield 0
    assert EmailFeatureExtractor.shannon_entropy("") == 0.0
    # Low diversity string should have lower entropy than high diversity
    low_ent = EmailFeatureExtractor.shannon_entropy("aaaaaaa")
    high_ent = EmailFeatureExtractor.shannon_entropy("abcdefg")
    assert low_ent == 0.0
    assert high_ent > 2.0


def test_feature_extractor_outputs():
    extractor = EmailFeatureExtractor()
    subject = "URGENT: Password Reset"
    body = "Please visit http://192.168.1.1/login to verify your banking credentials."

    features = extractor.extract_features(subject, body)

    assert "text_entropy" in features
    assert "url_count" in features
    assert features["url_count"] == 1.0
    assert features["ip_url_count"] == 1.0
    assert features["suspicious_keyword_count"] >= 1.0


def test_batch_feature_transform():
    extractor = EmailFeatureExtractor()
    emails = [
        {"subject": "Hello", "body": "How are you doing?"},
        {"subject": "Alert", "body": "Account suspended: http://evil.com"},
    ]
    matrix = extractor.transform(emails)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[0] == 2
    assert matrix.shape[1] > 5
