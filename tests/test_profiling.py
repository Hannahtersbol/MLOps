import pytest
import torch

from src.catdogdetection.profiling import TorchProfiler


def test_profiler_initialization_with_cuda():
    """Test initialization of TorchProfiler with CUDA enabled."""
    # Simulate CUDA being available
    cuda_available = torch.cuda.is_available()
    profiler = TorchProfiler(log_dir="./log", use_cuda=True)

    # Check that CPU profiling is always enabled
    assert torch.profiler.ProfilerActivity.CPU in profiler.profiler.activities

    # Check if CUDA profiling is included when available
    if cuda_available:
        assert torch.profiler.ProfilerActivity.CUDA in profiler.profiler.activities
    else:
        assert torch.profiler.ProfilerActivity.CUDA not in profiler.profiler.activities


def test_profiler_initialization_without_cuda():
    """Test initialization of TorchProfiler with CUDA disabled."""
    profiler = TorchProfiler(log_dir="./log", use_cuda=False)

    # Check that CPU profiling is enabled
    assert torch.profiler.ProfilerActivity.CPU in profiler.profiler.activities

    # Check that CUDA profiling is not included
    assert torch.profiler.ProfilerActivity.CUDA not in profiler.profiler.activities
