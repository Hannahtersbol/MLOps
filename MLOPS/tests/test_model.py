import pytest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from catdogdetection.model import Model, count_parameters


#To run test run pytest src/catdogdetection/tests/test_model.py
 
@pytest.fixture
def model_instance():
    return Model()

def test_forward_pass(model_instance):
    """Test if the forward pass produces the correct output shape."""
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output = model_instance(dummy_input)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}."

def test_count_parameters(model_instance):
    """Test the count_parameters function."""
    num_params = count_parameters(model_instance)
    expected_params = sum(p.numel() for p in model_instance.parameters())
    assert num_params == expected_params, (
        f"Expected parameter count {expected_params}, got {num_params}."
    )

def test_model_structure(model_instance):
    """Test if the model structure is as expected."""
    # Check if the output layer has 2 output features
    assert model_instance.model.fc.out_features == 2, "Output layer should have 2 output features."
    # Check if the input layer accepts 1 input channel
    assert model_instance.model.conv1.in_channels == 1, "Input layer should accept 1 input channel."