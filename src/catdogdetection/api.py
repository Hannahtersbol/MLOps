import asyncio
import subprocess

from fastapi import FastAPI
from invoke import Context

from tasks import preprocess_data

app = FastAPI()


@app.get("/")
def example():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/getaccuracy")
def get_accuracy():
    return {}


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
