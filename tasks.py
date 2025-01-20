import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "catdogdetection"
PYTHON_VERSION = "3.12"


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context, s: int = 1000) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py {s}", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, config_name: str = "Exp1") -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py --config-name={config_name}", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, m: str = "model") -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/evaluate.py {m}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    # ctx.run(
    #     f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
    #     echo=True,
    #     pty=not WINDOWS
    # )


@task
def docker_build_in_cloud(ctx: Context, b: str) -> None:
    """Build docker image in the vloud."""
    ctx.run(f"gcloud builds submit --config={b}" + ".yaml" + " .", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
    

@task
def count_files(ctx: Context, dir_path: str) -> None:
    """Count the number of files in a directory."""
    try:
        num_files = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
        print(f"The directory '{dir_path}' contains {num_files} files.")
    except Exception as e:
        print(f"An error occurred: {e}")   
