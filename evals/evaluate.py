"""Evaluation framework for comparing phishing detection models."""

import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    num_test_samples: int


def compare_models(results: list[EvalResult]) -> None:
    """Print comparison table of model results."""
    print(f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.model_name:<30} {r.accuracy:>10.4f} {r.precision:>10.4f} "
            f"{r.recall:>10.4f} {r.f1:>10.4f} {r.auc:>10.4f}"
        )


def save_results(results: list[EvalResult], output_path: Path) -> None:
    """Save evaluation results to JSON."""
    data = [asdict(r) for r in results]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
