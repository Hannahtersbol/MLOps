import torch

class TorchProfiler:
    def __init__(self, log_dir="./log", use_cuda=True):
        """
        Initialize the profiler.

        Args:
            log_dir (str): Directory to save profiling traces for TensorBoard.
            use_cuda (bool): Whether to include CUDA profiling.
        """
        activities = [torch.profiler.ProfilerActivity.CPU]
        if use_cuda and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            with_stack=True
        )

    def __enter__(self):
        """Start the profiler."""
        self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the profiler and print results."""
        self.profiler.__exit__(exc_type, exc_value, traceback)
        print(self.profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    def export_trace(self):
        """Export the profiling trace to TensorBoard."""
        self.profiler.export_chrome_trace("trace.json")
