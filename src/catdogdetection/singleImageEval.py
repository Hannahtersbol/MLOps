import timm
import torch
import typer
from src.catdogdetection.model import Model
from PIL import Image
from torch import nn
from torchvision import transforms


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
    #evaluate a single image given a local image path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluating single image")
    print(DEVICE)
    print(model_checkpoint)

    # Load the model the same way as done in evaluate
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint + ".pth", map_location=DEVICE, weights_only=True))

    # Preprocess the input image
    image = preprocess_image(image_path).to(DEVICE)

    # evaluate the picture using the model
    model.eval()
    with torch.no_grad():
        y_pred = model(image)
        prediction = y_pred.argmax(dim=1).item()  

    #see if it is a cat or a dog 
    class_labels = {0: "cat", 1: "dog"}  
    result = class_labels.get(prediction, "Unknown")

    print(f"The image is classified as: {result}")
    return result


if __name__ == "__main__":
    typer.run(evaluate_single_image)


def preprocess_image_from_bytes(image_bytes: bytes) -> torch.Tensor:
    #presprocess image even if it is a color(3-dimensional) image
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5], std=[0.5]),  
        ]
    )

    # Open the image from bytes
    image = Image.open(BytesIO(image_bytes)).convert("L")
    return transform(image).unsqueeze(0)  # Add batch dimension


def evaluate_single_image_from_bytes(model_checkpoint: str, image_bytes: bytes) -> str:
    #evaulates a single image given in bytes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Evaluating single image")
    

    # Load the model the same way as done in evaluate.py
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("models/" + model_checkpoint + ".pth", map_location=DEVICE, weights_only=True))

    # Preprocess the input image
    image = preprocess_image_from_bytes(image_bytes).to(DEVICE)

    # Perform inference by putting it into the model
    model.eval()
    with torch.no_grad():
        y_pred = model(image)
        prediction = y_pred.argmax(dim=1).item()  

    # Map the prediction to the class label to see if it is a cat or a dog 
    class_labels = {0: "cat", 1: "dog"}  
    result = class_labels.get(prediction, "Unknown")

    print(f"The image is classified as: {result}")
    return result
