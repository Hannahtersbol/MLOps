import timm
import torch
import typer
from PIL import Image
from torch import nn
from torchvision import transforms

from src.catdogdetection.model import Model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess the input image to match the model's requirements."""
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize the image to the size expected by the model
            transforms.Grayscale(num_output_channels=1),  # Ensure the image has a single channel
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for a single-channel image
        ]
    )

    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0)  # Add batch dimension


def evaluate_single_image(model_checkpoint: str, image_path: str) -> None:
    """Evaluate a single image using the trained model."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluating single image")
    print(DEVICE)
    print(model_checkpoint)

    # Load the model
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint + ".pth", map_location=DEVICE, weights_only=True))

    # Preprocess the input image
    image = preprocess_image(image_path).to(DEVICE)

    # Perform inference
    model.eval()
    with torch.no_grad():
        y_pred = model(image)
        prediction = y_pred.argmax(dim=1).item()  # Get the predicted class index

    # Map the prediction to the class label
    class_labels = {0: "cat", 1: "dog"}  # Adjust as per your dataset's class mapping
    result = class_labels.get(prediction, "Unknown")

    print(f"The image is classified as: {result}")
    return result


if __name__ == "__main__":
    typer.run(evaluate_single_image)


def preprocess_image_from_bytes(image_bytes: bytes) -> torch.Tensor:
    """Preprocess the input image from raw bytes to match the model's requirements."""
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize the image to the size expected by the model
            transforms.Grayscale(num_output_channels=1),  # Ensure the image has a single channel
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for a single-channel image
        ]
    )

    # Open the image from bytes
    image = Image.open(BytesIO(image_bytes)).convert("L")
    return transform(image).unsqueeze(0)  # Add batch dimension


def evaluate_single_image_from_bytes(model_checkpoint: str, image_bytes: bytes) -> str:
    """Evaluate a single image (provided as bytes) using the trained model."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluating single image")
    print(DEVICE)
    print(model_checkpoint)

    # Load the model
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint + ".pth", map_location=DEVICE, weights_only=True))

    # Preprocess the input image
    image = preprocess_image_from_bytes(image_bytes).to(DEVICE)

    # Perform inference
    model.eval()
    with torch.no_grad():
        y_pred = model(image)
        prediction = y_pred.argmax(dim=1).item()  # Get the predicted class index

    # Map the prediction to the class label
    class_labels = {0: "cat", 1: "dog"}  # Adjust as per your dataset's class mapping
    result = class_labels.get(prediction, "Unknown")

    print(f"The image is classified as: {result}")
    return result
