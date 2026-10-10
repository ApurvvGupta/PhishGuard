"""Dataset loading and preprocessing for phishing email detection."""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional
from .config import ModelConfig, DataConfig


class PhishingEmailDataset(Dataset):
    """PyTorch Dataset for phishing email classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_and_preprocess(
    data_path: str,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data and split into train/val/test."""
    import logging
    logger = logging.getLogger(__name__)

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} raw samples from {data_path}")
        
        # Simple random split based on config
        train_df = df.sample(frac=config.train_split, random_state=config.random_seed)
        val_test_df = df.drop(train_df.index)
        
        val_frac = config.val_split / (config.val_split + config.test_split)
        val_df = val_test_df.sample(frac=val_frac, random_state=config.random_seed)
        test_df = val_test_df.drop(val_df.index)
        
        logger.info(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"Error loading dataset from {data_path}: {str(e)}")
        raise
