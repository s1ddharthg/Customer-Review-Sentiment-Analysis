from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    dataset_path: Path = (
        Path(__file__).resolve().parent
        / "dataset"
        / "datafiniti"
        / "consumer-reviews-of-amazon-products"
    )
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 1000
    criterion: str = "gini"
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1