import time
from fastapi import APIRouter, Response
from typing import Optional
from ..core.bus import Bus, last_or
from ..sensors.camera import camera, get_telemetry_snapshot  # ajusta import si no moviste

router = APIRouter()

BUS: Optional[Bus] = None
T0 = time.time()

@router.get("/health")
async def health():
    tel = await last_or(BUS, "telemetry", get_telemetry_snapshot()) if BUS else get_telemetry_snapshot()
    up = time.time() - T0
    return {
        "ok": True,
        "uptime_sec": round(up, 1),
        "resolution": tel.get("resolution", [0, 0]),
        "fps_current": tel.get("fps", 0.0),
    }

@router.get("/snapshot.jpg")
def snapshot_jpg(quality: int = 85):
    jpg = camera.snapshot_jpeg(quality=quality)
    return Response(content=jpg, media_type="image/jpeg")
