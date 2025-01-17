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
        return self.size

    def getcat(self, index: int):
        """Return a given sample from the dataset."""
        image_path = self.image_paths_cats[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image)

    def getdog(self, index: int):
        """Return a given sample from the dataset."""
        image_path = self.image_paths_dogs[index]
        image = Image.open(image_path).convert("L")
        return self.transform(image)

    def getImagesAndTargets(self):
        dataset = []
        for idx, _ in enumerate(self.image_paths_cats):
            image = self.getcat(idx)
            image = (image * 255).byte()  # Convert to uint8
            dataset.append((image, 0))
            if idx >= self.size / 2:
                break
        for idx, _ in enumerate(self.image_paths_dogs):
            image = self.getdog(idx)
            image = (image * 255).byte()  # Convert to uint8
            dataset.append((image, 1))
            if idx >= self.size / 2:
                break

        random.shuffle(dataset)
        self.images, self.targets = zip(*dataset)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save all images with labels to a single .pt file."""

        train_images = self.images[: self.size // 2]
        train_targets = self.targets[: self.size // 2]
        test_images = self.images[self.size // 2 :]
        test_targets = self.targets[self.size // 2 :]

        all_images_tensor = torch.stack(train_images)
        all_labels_tensor = torch.tensor(train_targets)

        test_images_tensor = torch.stack(test_images)
        test_labels_tensor = torch.tensor(test_targets)

        torch.save(all_images_tensor, output_folder / "train_images.pt")
        torch.save(all_labels_tensor, output_folder / "train_target.pt")

        torch.save(test_images_tensor, output_folder / "test_images.pt")
        torch.save(test_labels_tensor, output_folder / "test_target.pt")


def preprocess(size, raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(int(size), raw_data_path)
    dataset.preprocess(output_folder)
    print("Data preprocessed")


def load_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load the preprocessed data from a .pt file."""
    train_images = torch.load("data/processed/train_images.pt", weights_only=True).float()
    train_target = torch.load("data/processed/train_target.pt", weights_only=True)
    test_images = torch.load("data/processed/test_images.pt", weights_only=True).float()
    test_target = torch.load("data/processed/test_target.pt", weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(lambda size: preprocess(size, Path("data/raw/"), Path("data/processed/")))
