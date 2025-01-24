import torch
import typer

from data import load_data
from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(DEVICE)
    print(model_checkpoint)

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint + ".pth", weights_only=True, map_location=DEVICE))

    _, test_set = load_data()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)

        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(model)
    print(f"Test accuracy: {correct / total}")
    return correct / total


if __name__ == "__main__":
    typer.run(evaluate)
