import os

import kagglehub

from src.config import TrainConfig


def download_dataset() -> str:
    config = TrainConfig()
    dataset_path = config.dataset_path

    dataset_file = os.path.join(dataset_path, config.dataset_file)
    if os.path.isfile(dataset_file):
        print("Dataset already exists at:", dataset_file)
        return dataset_path

    print("Dataset not found. Downloading...")
    os.makedirs(dataset_path, exist_ok=True)

    downloaded_path = kagglehub.dataset_download(
        config.dataset_handle,
        output_dir=dataset_path,
    )
    print("Dataset downloaded to:", downloaded_path)
    return dataset_path


if __name__ == "__main__":
    download_dataset()