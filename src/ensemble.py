"""Ensemble classification module for PhishGuard.

Combines fine-tuned DeBERTa contextual representations with gradient-boosted decision
trees (GBDT / XGBoost) trained on tabular statistical and lexical signals.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

from .model import PhishingDetector
from .features import EmailFeatureExtractor


class PhishGuardEnsemble:
    """Soft-voting and stacking ensemble model combining Deep Learning and GBDT."""

    def __init__(
        self,
        transformer_model: PhishingDetector,
        gbdt_model: any,
        feature_extractor: EmailFeatureExtractor,
        transformer_weight: float = 0.65,
        gbdt_weight: float = 0.35,
    ):
        self.transformer = transformer_model
        self.gbdt = gbdt_model
        self.feature_extractor = feature_extractor
        self.transformer_weight = transformer_weight
        self.gbdt_weight = gbdt_weight

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emails_metadata: List[Dict[str, str]],
    ) -> np.ndarray:
        """Calculate ensembled prediction probabilities using soft voting."""
        # 1. Transformer predictions
        with torch.no_grad():
            _, transformer_probs = self.transformer.predict(input_ids, attention_mask)
            p_trans = transformer_probs.cpu().numpy()

        # 2. Extract engineered features and run GBDT inference
        feat_matrix = self.feature_extractor.transform(emails_metadata)
        p_gbdt = self.gbdt.predict_proba(feat_matrix)

        # 3. Weighted convex combination of output probability distributions
        ensemble_probs = (
            self.transformer_weight * p_trans + self.gbdt_weight * p_gbdt
        )
        return ensemble_probs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emails_metadata: List[Dict[str, str]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Classify emails as phishing (1) or benign (0) with configurable decision boundary."""
        probs = self.predict_proba(input_ids, attention_mask, emails_metadata)
        phishing_probs = probs[:, 1]
        return (phishing_probs >= threshold).astype(int)

    def evaluate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        emails_metadata: List[Dict[str, str]],
        y_true: List[int],
    ) -> Dict[str, float]:
        """Compute end-to-end evaluation metrics for the ensemble."""
        probs = self.predict_proba(input_ids, attention_mask, emails_metadata)
        preds = np.argmax(probs, axis=-1)
        phish_scores = probs[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary"
        )
        auc = roc_auc_score(y_true, phish_scores)
        acc = accuracy_score(y_true, preds)

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }
