# app/core/settings.py
import os
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config", ".env")
load_dotenv(ENV_PATH)

JETSON = os.getenv("JETSON", "0") == "1"
CAM_INDEX = int(os.getenv("CAM_INDEX", "0"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))
WS_RATE_HZ = float(os.getenv("WS_RATE_HZ", "10"))