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
        # print(f"Found {self.num_images} images in {raw_data_path}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjusted for single channel
            ]
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

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

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save all images with labels to a single .pt file."""
        # output_folder.mkdir(parents=True, exist_ok=True)
        all_images = []
        all_labels = []
        test_images = []
        test_labels = []
        for idx, image_path in enumerate(self.image_paths_cats):
            image = self.getcat(idx)
            image = (image * 255).byte()  # Convert to uint8
            if idx >= self.size / 2:
                test_images.append(image)
                test_labels.append(0)
            else:
                all_images.append(image)
                all_labels.append(0)  # Label for cats

            if idx >= self.size:
                break
        for idx, image_path in enumerate(self.image_paths_dogs):
            image = self.getdog(idx)
            image = (image * 255).byte()  # Convert to uint8
            if idx >= self.size / 2:
                test_images.append(image)
                test_labels.append(1)
            else:
                all_images.append(image)
                all_labels.append(1)  # Label for dogs

            if idx >= self.size:
                break
        all_images_tensor = torch.stack(all_images)
        all_labels_tensor = torch.tensor(all_labels)

        test_images_tensor = torch.stack(test_images)
        test_labels_tensor = torch.tensor(test_labels)

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
