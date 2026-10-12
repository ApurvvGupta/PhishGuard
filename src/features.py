"""Feature extraction module for phishing detection.

Extracts lexical, structural, and statistical features from email bodies
and embedded URLs to feed into classical ML models and ensemble classifiers.
"""

import math
import re
from typing import Dict, List, Any
import numpy as np


class EmailFeatureExtractor:
    """Extract tabular features from email raw text and metadata."""

    SUSPICIOUS_KEYWORDS = [
        "urgent", "verify", "suspended", "account", "bank", "password",
        "security", "login", "update", "billing", "confirm", "action required",
        "immediately", "unauthorized", "compromised", "lottery", "winner"
    ]

    def __init__(self):
        self.url_regex = re.compile(
            r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*",
            re.IGNORECASE
        )
        self.ip_regex = re.compile(
            r"https?://(?:\d{1,3}\.){3}\d{1,3}(?:[^\s]*)?",
            re.IGNORECASE
        )

    @staticmethod
    def shannon_entropy(text: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        return -sum(p * math.log(p) / math.log(2.0) for p in prob)

    def extract_features(self, subject: str, body: str) -> Dict[str, float]:
        """Extract a structured feature dictionary from an email."""
        text = f"{subject} {body}".strip()
        text_lower = text.lower()
        urls = self.url_regex.findall(text)
        ip_urls = self.ip_regex.findall(text)

        length = len(text)
        words = text.split()
        num_words = len(words)

        keyword_hits = sum(1 for kw in self.SUSPICIOUS_KEYWORDS if kw in text_lower)

        features = {
            "char_count": float(length),
            "word_count": float(num_words),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(length, 1),
            "digit_ratio": sum(1 for c in text if c.isdigit()) / max(length, 1),
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(length, 1),
            "url_count": float(len(urls)),
            "ip_url_count": float(len(ip_urls)),
            "suspicious_keyword_count": float(keyword_hits),
            "text_entropy": self.shannon_entropy(text),
            "has_attachment_mention": float(any(w in text_lower for w in ["attached", "attachment", ".pdf", ".zip", ".exe"])),
            "has_urgency_cue": float(any(w in text_lower for w in ["24 hours", "immediate", "within 48 hours", "suspend"])),
        }

        # Average URL length and entropy if URLs exist
        if urls:
            features["avg_url_length"] = float(np.mean([len(u) for u in urls]))
            features["max_url_entropy"] = float(max([self.shannon_entropy(u) for u in urls]))
        else:
            features["avg_url_length"] = 0.0
            features["max_url_entropy"] = 0.0

        return features

    def transform(self, emails: List[Dict[str, str]]) -> np.ndarray:
        """Batch feature extraction returning a 2D numpy array."""
        feature_dicts = [
            self.extract_features(e.get("subject", ""), e.get("body", ""))
            for e in emails
        ]
        keys = sorted(feature_dicts[0].keys())
        matrix = [[f[k] for k in keys] for f in feature_dicts]
        return np.array(matrix, dtype=np.float32)
