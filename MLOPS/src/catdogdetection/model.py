import timm
import torch
from torch import nn
import typing

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model("resnet18", pretrained=True, in_chans=1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model = Model()
    model.eval()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {count_parameters(model)}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
