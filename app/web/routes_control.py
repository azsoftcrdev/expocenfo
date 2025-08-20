from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from ..streaming import mjpeg_generator

router = APIRouter()

@router.get("/stream.mjpg")
def stream_mjpg(
    request: Request,
    mode: Optional[str] = None,
    color: Optional[str] = None,
    overlay: bool = True,
    quality: int = 80
):
    if mode is not None:
        mode = mode.lower()
        if mode not in ("color", "face"):
            raise HTTPException(status_code=400, detail="mode debe ser 'color' o 'face'")
    if color is not None:
        color = color.lower()
        valid = {"red", "green", "blue", "yellow"}
        if color not in valid:
            raise HTTPException(status_code=400, detail=f"color inválido. Usa {sorted(valid)}")
        if mode != "color":
            raise HTTPException(status_code=400, detail="param 'color' solo aplica con mode=color")
    if not (10 <= quality <= 95):
        raise HTTPException(status_code=400, detail="quality debe estar entre 10 y 95")

    gen = mjpeg_generator(mode=mode, color=color, overlay=overlay, quality=quality)

    async def _stream():
        try:
            async for chunk in gen:
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception:
            pass

    return StreamingResponse(_stream(), media_type="multipart/x-mixed-replace; boundary=frame")
