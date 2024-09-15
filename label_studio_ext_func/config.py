import os
from pathlib import Path

from dotenv import load_dotenv

env_path = f"{Path(__file__).parent.parent.absolute()}/.env.server"
load_dotenv(dotenv_path=env_path)

TITLE = os.environ.get("TITLE", "lbs_extended_function")
SWAGGER = int(os.environ.get("SWAGGER", 1))

# possible labels in LB projects
labels = (
    "car",
    "ferry",
    "airplane",
)


class LB:
    LABEL_STUDIO_API = os.environ.get("LABEL_STUDIO_API")
    LB_SERVICE_NAME = os.environ.get("LABEL_STUDIO_HOST")
    LABEL_STUDIO_PORT = os.environ.get("LABEL_STUDIO_PORT")
