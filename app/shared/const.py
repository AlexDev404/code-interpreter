from pathlib import Path
import os

# Get the current working directory or use environment variable if set
HOST_PATH = Path(os.environ.get("HOST_PATH", os.getcwd()))

# Absolute path inside the container where files are read/written.
# Reads from CONTAINER_UPLOAD_PATH env var; default matches the Dockerfile layout.
UPLOAD_PATH = Path(os.environ.get("CONTAINER_UPLOAD_PATH", "/app/uploads"))
CONFIG_PATH = Path("config")
