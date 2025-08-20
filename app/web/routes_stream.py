# app/web/routes_stream.py
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio

from ..streaming import mjpeg_generator

router = APIRouter()

@router.get("/stream.mjpg")
def stream_mjpg(
    request: Request,
    mode: Optional[str] = Query("none"),              # "none" | "color" | "face" || "person"
    color: Optional[str] = Query(None),               # "red"|"green"|"blue"|"yellow" (si mode=color)
    overlay: bool = Query(True),
    quality: int = Query(70, ge=40, le=95),
    fps: float = Query(15.0, ge=1.0, le=30.0),
    size: Optional[str] = Query(None, description="Ej: 640x360")
):
    mode = (mode or "none").lower()
    if mode not in ("none", "color", "face", "person"):
        raise HTTPException(status_code=400, detail="mode debe ser 'none'|'color'|'face'")

    if color is not None:
        color = color.lower()
        valid = {"red", "green", "blue", "yellow"}
        if color not in valid:
            raise HTTPException(status_code=400, detail=f"color inválido. Usa {sorted(valid)}")
        if mode != "color":
            raise HTTPException(status_code=400, detail="param 'color' solo aplica con mode=color")

    # ---- watcher de desconexión (async) -> flag que puede leer el generador sync
    closed_flag = {"closed": False}
    async def _watch_disconnect():
        await request.is_disconnected()
        closed_flag["closed"] = True
    try:
        asyncio.get_event_loop().create_task(_watch_disconnect())
    except RuntimeError:
        # Si no hay loop (caso edge), lo ignoramos: el generador igual seguirá.
        pass

    def on_disconnect() -> bool:
        return closed_flag["closed"]

    gen = mjpeg_generator(
        mode=mode, color=color, overlay=overlay,
        quality=quality, fps=fps, size=size,
        is_disconnected=on_disconnect
    )

    # headers para minimizar buffering del cliente/proxy/CDN
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        # Algunos navegadores agradecen este header en multipart
        "Connection": "keep-alive",
    }

    return StreamingResponse(
        gen,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=headers,
    )

