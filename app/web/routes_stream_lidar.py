# app/api/lidar_stream.py
from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from ..sensors.lidar import lidar, BOUNDARY_NAME

router = APIRouter(prefix="/lidar", tags=["lidar"])

# --- Canonical endpoints (con extensión) ---
@router.get("/snapshot.jpg")
def lidar_snapshot_jpg(mode: Optional[str] = "map",
                       size: Optional[str] = None,
                       quality: int = 80):
    jpg = lidar().snapshot_jpeg(mode=mode, size_str=size, quality=quality)
    if not jpg:
        raise HTTPException(500, "No se pudo generar snapshot del LiDAR")
    return Response(content=jpg, media_type="image/jpeg")

@router.get("/stream.mjpg")
def lidar_stream_mjpg(mode: Optional[str] = "map",
                      fps: float = 8.0,
                      size: Optional[str] = None,
                      quality: int = 80):
    gen = lidar().mjpeg_generator(mode=mode, fps=fps, size_str=size, quality=quality)
    return StreamingResponse(gen, media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY_NAME}")

# --- Alias sin extensión (compatibilidad) ---
@router.get("/snapshot")
def lidar_snapshot_alias(mode: Optional[str] = "map",
                         size: Optional[str] = None,
                         quality: int = 80):
    return lidar_snapshot_jpg(mode=mode, size=size, quality=quality)

@router.get("/stream")
def lidar_stream_alias(mode: Optional[str] = "map",
                       fps: float = 8.0,
                       size: Optional[str] = None,
                       quality: int = 80):
    return lidar_stream_mjpg(mode=mode, fps=fps, size=size, quality=quality)

@router.get("/status")
def lidar_status():
    a, d, q, ts = lidar().get_latest_scan()
    import time as _t
    return {
        "backend": "rplidar",
        "points": int(a.size),
        "age_s": None if ts <= 0 else (_t.time() - float(ts)),
        "port": lidar().port,
        "baud": lidar().baudrate,
        "map_size": getattr(lidar(), "_map", None).shape if getattr(lidar(), "_map", None) is not None else None,
        "scale_px_per_m": float(getattr(__import__("os"), "environ", {}).get("LIDAR_MAP_SCALE", 100.0)),
    }

