import pytest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

import cats
import dogs

@pytest.fixture
def dataset():
    raw_data_path = Path("/path/to/your/data")
    return MyDataset(size=100, raw_data_path=raw_data_path)

def test_my_dataset_instance(dataset):
    """Test if MyDataset is an instance of torch.utils.data.Dataset."""
    assert isinstance(dataset, Dataset), "MyDataset should be an instance of torch.utils.data.Dataset"

def test_dataset_length(dataset):
    """Test the length of the dataset."""
    expected_length = 100
    assert len(dataset) == expected_length, f"Expected dataset length {expected_length}, got {len(dataset)}"

def test_transform_exists(dataset):
    """Test if the transform attribute is set."""
    assert dataset.transform is not None, "Transform should not be None"

def test_getcat_shape(dataset):
    """Test the shape of the transformed cat image."""
    cat_image = dataset.getcat(0)
    assert cat_image.shape == (1, 150, 150), f"Expected cat image shape (1, 150, 150), got {cat_image.shape}"

def test_getdog_shape(dataset):
    """Test the shape of the transformed dog image."""
    dog_image = dataset.getdog(0)
    assert dog_image.shape == (1, 150, 150), f"Expected dog image shape (1, 150, 150), got {dog_image.shape}"

def test_data_path(dataset):
    """Test if the data path is set correctly."""
    raw_data_path = Path("/path/to/your/data")
    assert dataset.data_path == raw_data_path, f"Expected data path {raw_data_path}, got {dataset.data_path}"