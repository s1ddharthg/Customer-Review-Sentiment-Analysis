from pathlib import Path

import kagglehub

from config import TrainConfig


def download_dataset() -> Path:
    dataset_path = TrainConfig.dataset_path
    if dataset_path.is_dir() and any(dataset_path.iterdir()):
        print("Dataset already exists at:", dataset_path)
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    print("Dataset not found. Downloading...")
    downloaded_path = kagglehub.dataset_download(
        "datafiniti/consumer-reviews-of-amazon-products",
        output_dir=str(dataset_path),
    )
    print("Dataset downloaded to:", downloaded_path)
    return Path(downloaded_path)


if __name__ == "__main__":
    download_dataset()