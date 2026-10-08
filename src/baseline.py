"""Classical ML baselines for phishing detection."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_tfidf_logreg() -> Pipeline:
    """TF-IDF + Logistic Regression baseline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])


def build_tfidf_gbt() -> Pipeline:
    """TF-IDF + Gradient Boosted Trees baseline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        )),
    ])


def evaluate_baseline(
    pipeline: Pipeline,
    X_train: list[str],
    y_train: list[int],
    X_test: list[str],
    y_test: list[int],
) -> dict:
    """Train and evaluate a baseline pipeline."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary"
    )
    auc = roc_auc_score(y_test, probs)
    accuracy = accuracy_score(y_test, preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    logger.info(f"Baseline Results: {metrics}")
    logger.info(f"\n{classification_report(y_test, preds, target_names=['Legitimate', 'Phishing'])}")

    return metrics
