import os
import random
from pathlib import Path

import torch
import typer
from PIL import Image
from torchvision import transforms


class MyDataset:
    def __init__(self, size, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.size = size
        self.image_paths_cats = list(raw_data_path.glob("cats/*.jpg"))
        self.image_paths_dogs = list(raw_data_path.glob("dogs/*.jpg"))

        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjusted for single channel
            ]
        )
        self.getImagesAndTargets()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images) if hasattr(self, "images") else 0

    def getcat(self, index: int):
        """Return a given sample from the cat dataset."""
        image_path = self.image_paths_cats[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image)

    def getdog(self, index: int):
        """Return a given sample from the dog dataset."""
        image_path = self.image_paths_dogs[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image)

    def getImagesAndTargets(self):
        """Load images and targets into memory."""
        dataset = []
        max_per_class = self.size // 2

        # Add cat images to the dataset
        for idx, image_path in enumerate(self.image_paths_cats):
            if idx >= max_per_class:
                break
            image = self.getcat(idx)
            dataset.append((image, 0))  # Label 0 for cats

        # Add dog images to the dataset
        for idx, image_path in enumerate(self.image_paths_dogs):
            if idx >= max_per_class:
                break
            image = self.getdog(idx)
            dataset.append((image, 1))  # Label 1 for dogs

        if not dataset:
            raise ValueError("No images found in the dataset directories.")

        # Shuffle and unpack the dataset
        random.shuffle(dataset)
        self.images, self.targets = zip(*dataset)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save all images with labels to a single .pt file."""
        train_images = self.images[: len(self.images) // 2]
        train_targets = self.targets[: len(self.targets) // 2]
        test_images = self.images[len(self.images) // 2 :]
        test_targets = self.targets[len(self.targets) // 2 :]

        torch.save(torch.stack(train_images), output_folder / "train_images.pt")
        torch.save(torch.tensor(train_targets), output_folder / "train_target.pt")
        torch.save(torch.stack(test_images), output_folder / "test_images.pt")
        torch.save(torch.tensor(test_targets), output_folder / "test_target.pt")


def preprocess(size, raw_data_path: Path, output_folder: Path) -> None:
    if os.path.exists("/.dockerenv") or os.getenv("container", "") == "docker":
        raw_data_path = Path("mnt") / raw_data_path
        print(raw_data_path)
    print("Preprocessing data...")
    dataset = MyDataset(int(size), raw_data_path)
    dataset.preprocess(output_folder)
    print("Data preprocessed")


def load_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load the preprocessed data from a .pt file."""
    train_images = torch.load("data/processed/train_images.pt").float()
    train_targets = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt").float()
    test_targets = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_targets)
    test_set = torch.utils.data.TensorDataset(test_images, test_targets)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(lambda size: preprocess(size, Path("data/raw/"), Path("data/processed/")))
