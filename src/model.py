"""Model definitions for phishing detection."""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from .config import ModelConfig


class PhishingDetector(nn.Module):
    """DeBERTa-based phishing email detector with optional dropout."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            config=model_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Run inference and return predicted class and probabilities."""
        self.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs
