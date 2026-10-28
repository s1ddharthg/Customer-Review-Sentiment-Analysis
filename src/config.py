from dataclasses import dataclass
import os


@dataclass
class TrainConfig:
    dataset_path: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dataset",
    )
    dataset_handle: str = "parve05/customer-review-dataset"
    dataset_file: str = "redmi6.csv"

    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000

    criterion: str = "gini"
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
