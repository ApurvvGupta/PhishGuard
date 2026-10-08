from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1

@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

@dataclass
class TrainingConfig:
    output_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    save_best_only: bool = True
    early_stopping_patience: int = 3
    eval_steps: int = 100
    fp16: bool = True
