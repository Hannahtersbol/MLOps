import os
import sys

import pytest
import torch

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.catdogdetection.model import Model

# To run test run pytest tests/test_model.py


@pytest.fixture
def model_instance():
    return Model()


def test_forward_pass(model_instance):
    """Test if the forward pass produces the correct output shape."""
    dummy_input = torch.randn(1, 1, 150, 150)  # Batch size 1, 1 channel, 150x150 image
    output = model_instance(dummy_input)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}."


def test_model_parameters(model_instance):
    """Test the number of parameters in the model."""
    num_params = sum(p.numel() for p in model_instance.parameters())
    assert num_params > 0, "Model should have parameters."


def test_model_structure(model_instance):
    """Test if the model structure is as expected."""
    # Check if the output layer has 2 output features
    assert model_instance.model.fc.out_features == 2, "Output layer should have 2 output features."
    # Check if the input layer accepts 1 input channel
    assert model_instance.model.conv1.in_channels == 1, "Input layer should accept 1 input channel."
