import os
from pathlib import Path

import pytest
import torch

from src.catdogdetection.data import MyDataset


@pytest.fixture
def dataset():
    # Point to data/test instead of data/raw
    test_data_path = Path(os.path.dirname(__file__), "../data/test").resolve()
    # Since you only have 2 total images (1 cat + 1 dog), set size=2
    return MyDataset(2, test_data_path)

def test_dataset_length(dataset):
    # Half of 'size=2' means 1 cat and 1 dog
    max_per_class = dataset.size // 2
    expected_cats = min(len(dataset.image_paths_cats), max_per_class)
    expected_dogs = min(len(dataset.image_paths_dogs), max_per_class)
    expected_length = expected_cats + expected_dogs

    actual_length = len(dataset)
    assert actual_length == expected_length, (
        f"Expected dataset length {expected_length}, got {actual_length}"
    )

def test_transform_exists(dataset):
    """Test that the dataset has a transform defined."""
    assert dataset.transform is not None, "Transform should not be None"


def test_getcat_shape(dataset):
    """Test shape of the first cat image after transformation."""
    if not dataset.image_paths_cats:
        pytest.skip("No cat images available for testing.")
    cat_image = dataset.getcat(0)
    assert cat_image.shape == (1, 150, 150), \
        f"Expected cat image shape (1, 150, 150), got {cat_image.shape}"


def test_getdog_shape(dataset):
    """Test shape of the first dog image after transformation."""
    if not dataset.image_paths_dogs:
        pytest.skip("No dog images available for testing.")
    dog_image = dataset.getdog(0)
    assert dog_image.shape == (1, 150, 150), \
        f"Expected dog image shape (1, 150, 150), got {dog_image.shape}"


def test_data_path(dataset):
    """Test that the dataset is reading from the correct folder."""
    raw_data_path = Path(os.path.dirname(__file__), "../data/test").resolve()
    assert dataset.data_path.resolve() == raw_data_path, \
        f"Expected data path {raw_data_path}, got {dataset.data_path.resolve()}"
