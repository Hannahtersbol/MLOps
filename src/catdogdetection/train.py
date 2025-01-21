import hydra
import torch
from model import Model
from profiling import TorchProfiler
from datetime import datetime
from google.cloud import storage
import requests
import time

from data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", version_base=None)
def train(config) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(config)
    now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")

    # config = OmegaConf.load(f"configs/{config_name}.yaml")
    lr = config.hyperparameters.learning_rate
    batch_size = config.hyperparameters.batch_size
    epochs = config.hyperparameters.epochs
    torch.manual_seed(config.hyperparameters.seed)
    print("after torch.manual")
    model = Model().to(DEVICE)
    print("after model.todevice")
    train_set, _ = load_data()
    print("after load data")
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=4, pin_memory=True)
    print("after datalost")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}

    # Start the profiler
    with TorchProfiler(log_dir="./log", use_cuda=torch.cuda.is_available()):
        for epoch in range(epochs):
            model.train()

            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE).float(), target.to(DEVICE)

                optimizer.zero_grad()
                y_pred = model(img)

                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                # Accumulate loss and accuracy in tensors
                epoch_loss += loss.item() * img.size(0)
                with torch.no_grad():
                    correct_predictions += (y_pred.argmax(dim=1) == target).sum().item()
                total_samples += img.size(0)

                # Log less frequently
                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, batch loss: {loss.item():.4f}")

            # Compute average metrics for the epoch
            average_loss = epoch_loss / total_samples
            accuracy = correct_predictions / total_samples

            statistics["train_loss"].append(average_loss)
            statistics["train_accuracy"].append(accuracy)

            print(f"Epoch {epoch} complete. Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    print("Training complete")
    model_filename = f"models/Exp-{now.replace('/', '-')}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    upload_to_gcs(bucket_name="catdog-models", source_file_name=model_filename, destination_blob_name=model_filename)

    config_filename = f"outputs/{now}/.hydra/config.yaml"
    config_destination = f"{now}/config{now.split('/')[0]}.yaml"
    upload_to_gcs(bucket_name="catdog-models", source_file_name=config_filename, destination_blob_name=config_destination)
    print(f"Config file uploaded as {config_destination}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            blob.upload_from_filename(source_file_name)
            print(f"File {source_file_name} uploaded to {destination_blob_name}.")
            break
        except (requests.exceptions.ConnectionError) as e:
            retries += 1
            print(f"Upload failed: {e}. Retrying {retries}/{max_retries}...")
            time.sleep(2 ** retries)  # Exponential backoff
    else:
        print(f"Failed to upload {source_file_name} after {max_retries} attempts.")



if __name__ == "__main__":
    train()
