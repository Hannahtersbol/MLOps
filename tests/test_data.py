import os
from pathlib import Path

import pytest
import torch

from src.catdogdetection.data import MyDataset


@pytest.fixture
def dataset():
    try:
        return MyDataset(30060, Path(os.path.join(os.path.dirname(__file__), "../data/raw")))
    except ValueError as e:
        pytest.skip(f"Skipping test due to dataset error: {e}")


def test_dataset_length(dataset):
    """Test the length of the dataset."""
    expected_length = min(30060, len(dataset.image_paths_cats) + len(dataset.image_paths_dogs))
    actual_length = len(dataset)
    assert actual_length == expected_length, f"Expected dataset length {expected_length}, got {actual_length}"


def test_transform_exists(dataset):
    """Test if the transform attribute is set."""
    assert dataset.transform is not None, "Transform should not be None"


def test_getcat_shape(dataset):
    """Test the shape of the transformed cat image."""
    if not dataset.image_paths_cats:
        pytest.skip("No cat images available for testing.")
    cat_image = dataset.getcat(0)
    assert cat_image.shape == (1, 150, 150), f"Expected cat image shape (1, 150, 150), got {cat_image.shape}"


def test_getdog_shape(dataset):
    """Test the shape of the transformed dog image."""
    if not dataset.image_paths_dogs:
        pytest.skip("No dog images available for testing.")
    dog_image = dataset.getdog(0)
    assert dog_image.shape == (1, 150, 150), f"Expected dog image shape (1, 150, 150), got {dog_image.shape}"


def test_data_path(dataset):
    """Test if the data path is set correctly."""
    raw_data_path = Path(os.path.join(os.path.dirname(__file__), "../data/raw")).resolve()
    assert (
        dataset.data_path.resolve() == raw_data_path
    ), f"Expected data path {raw_data_path}, got {dataset.data_path.resolve()}"
