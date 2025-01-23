from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.catdogdetection.api import app

client = TestClient(app)


def test_preprocess_with_s_20():
    """Test the preprocess endpoint with query parameter s=20."""
    with patch("src.catdogdetection.api.preprocess") as mock_preprocess:
        # Mock the preprocess function to simulate successful processing
        mock_preprocess.return_value = None

        # Make a GET request to the /preprocess endpoint
        response = client.get("/preprocess?s=20")

        # Assertions
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "message": "Data preprocessing started with parameter s=20",
        }

        # Ensure the mock was called with the correct arguments
        mock_preprocess.assert_called_once_with(
            20, raw_data_path=Path("data/raw/"), output_folder=Path("data/processed/")
        )


def test_evaluate_image():
    """Test the evaluate_image endpoint."""
    # Mock the evaluate_single_image_from_bytes function
    with patch("src.catdogdetection.api.evaluate_single_image_from_bytes") as mock_evaluate:
        # Set up the mock to return a classification result
        mock_evaluate.return_value = "cat"

        # Simulate uploading a file
        file_content = b"dummy image content"
        files = {"file": ("test_image.jpg", file_content, "image/jpeg")}
        data = {"model_checkpoint": "M_Exp1"}

        # Make the POST request to the /evaluate-image/ endpoint
        response = client.post("/evaluate-image/", data=data, files=files)

        # Assertions
        assert response.status_code == 200
        assert response.json() == {"classification": "cat"}

        # Verify the mock was called with the correct arguments
        mock_evaluate.assert_called_once_with("M_Exp1", file_content)
