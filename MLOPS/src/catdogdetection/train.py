import os
import torch
from model import Model
from data import load_data
from profiling import TorchProfiler
from omegaconf import OmegaConf

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def train(configName) -> None:
    """Train a model on MNIST."""
    print("Training day and night")

    # Dynamically calculate the path to the configs folder
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../../configs")
    config_path = os.path.join(CONFIG_DIR, configName + ".yaml")
    config = OmegaConf.load(config_path)

    lr = config.hyperparameters.learning_rate
    batch_size = config.hyperparameters.batch_size
    epochs = config.hyperparameters.epochs
    torch.manual_seed(config.hyperparameters.seed)

    model = Model().to(DEVICE)
    train_set, _ = load_data()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    # Start the profiler
    with TorchProfiler(log_dir="./log", use_cuda=torch.cuda.is_available()):
        for epoch in range(epochs):
            model.train()
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE).float(), target.to(DEVICE)  # Convert to float
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()
                statistics["train_loss"].append(loss.item())

                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)

                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    # Save the model to the models directory
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../models")
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pth"))
    print("Model saved")

if __name__ == "__main__":
    train("Exp1")
