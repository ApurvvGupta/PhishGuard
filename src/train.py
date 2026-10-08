"""Training pipeline for phishing detection models."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)
import numpy as np
from tqdm import tqdm
import logging

from .config import ModelConfig, DataConfig, TrainingConfig
from .model import PhishingDetector
from .dataset import PhishingEmailDataset

logger = logging.getLogger(__name__)


def train_epoch(
    model: PhishingDetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: PhishingDetector,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds, probs = model.predict(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }

    logger.info(f"Evaluation Results: {metrics}")
    logger.info(f"\n{classification_report(all_labels, all_preds, target_names=['Legitimate', 'Phishing'])}")

    return metrics
