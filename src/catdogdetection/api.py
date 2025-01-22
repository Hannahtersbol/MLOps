import asyncio
import os
import subprocess
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from invoke import Context

from src.catdogdetection.evaluate import evaluate
from src.catdogdetection.singleImageEval import evaluate_single_image_from_bytes
from tasks import preprocess_data
from src.catdogdetection.download_bucket import download_files_with_prefix

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """
    Function to run before the application starts.
    """
    try:
        print("Downloading files from GCP bucket...")
        await asyncio.to_thread(download_files_with_prefix, 'catdog-models', 'models', 'models')
        # await asyncio.to_thread(download_files_with_prefix, 'catdog-data', 'data/raw/cats', 'data/raw/cats')
        # await asyncio.to_thread(download_files_with_prefix, 'catdog-data', 'data/raw/dogs', 'data/raw/dogs')
    except Exception as e:
        print(f"Error during startup: {e}")


@app.get("/")
def example():
    try:
        files = os.listdir("models")
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/getaccuracy/{model_checkpoint}")
def get_accuracy(model_checkpoint: str):
    result = evaluate(model_checkpoint=model_checkpoint)  # Make sure 'evaluate' is a callable or value
    return {"message": f"Accuracy on model is {result}"}


@app.get("/preprocess")
async def preprocess_data_endpoint(s: int = 1000):
    """
    API endpoint to preprocess data.
    Accepts an optional query parameter `s`.
    """
    try:
        # Run the task asynchronously
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: preprocess_data(Context(), s=s))
        return {"status": "success", "message": f"Data preprocessing started with parameter s={s}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/train")
def train_model(config_name: str = "Exp1"):
    """
    API endpoint to train the model.
    Accepts an optional query parameter `config_name` to specify the configuration.
    """
    try:
        # Run the Invoke train task
        result = subprocess.run(
            ["invoke", "train", f"--config-name={config_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the command succeeded
        if result.returncode == 0:
            return {"status": "success", "output": result.stdout}
        else:
            return {"status": "error", "output": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Define the endpoint
@app.post("/evaluate-image/")
async def evaluate_image(model_checkpoint: str, file: UploadFile):
    """
    API endpoint to evaluate an image.
    - model_checkpoint: The model checkpoint file to load.
    - file: The uploaded image file.
    """
    try:
        # Read the uploaded image file bytes
        image_bytes = await file.read()

        # Call the evaluation function
        result = evaluate_single_image_from_bytes(model_checkpoint, image_bytes)

        # Return the classification result
        return JSONResponse(content={"classification": result})

    except Exception as e:
        # Handle any errors and return a meaningful error message
        raise HTTPException(status_code=500, detail=str(e))
