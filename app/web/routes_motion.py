from fastapi import APIRouter
from pydantic import BaseModel, conint
from typing import Optional
import asyncio

from ..motion.controller_vel import MotionControllerVel

router = APIRouter(prefix="/motion", tags=["motion"])
_controller: MotionControllerVel | None = None

def ctl() -> MotionControllerVel:
    global _controller
    if _controller is None:
        _controller = MotionControllerVel(hz=15.0, deadman_s=0.8)
        _controller.start(asyncio.get_event_loop())
    return _controller

# -------- API nueva: setpoint de velocidad --------
class VelBody(BaseModel):
    x: conint(ge=-30, le=30)
    y: conint(ge=-30, le=30)
    z: conint(ge=-30, le=30)
    speed: Optional[conint(ge=1, le=5)] = None

@router.post("/vel")
def set_vel(b: VelBody):
    return {"ok": True, "state": ctl().set_vel(b.x, b.y, b.z, b.speed)}

@router.post("/stop")
def stop():
    return {"ok": True, "state": ctl().stop()}

@router.get("/status")
def status():
    return {"ok": True, "state": ctl().snapshot()}

# -------- Compatibilidad (alias “por pasos”) --------
# Mapeamos a vectores unitarios, el bucle re-emite continuo sin colas
def _unit(v:int) -> int:
    # para sentirlo suave usa ±12..20 como “velocidad” base
    return max(-30, min(30, int(v)))

@router.post("/forward")
def forward():
    return {"ok": True, "state": ctl().set_vel(_unit(+15), 0, 0)}

@router.post("/back")
def back():
    return {"ok": True, "state": ctl().set_vel(_unit(-15), 0, 0)}

@router.post("/left")
def left():
    return {"ok": True, "state": ctl().set_vel(0, _unit(-15), 0)}

@router.post("/right")
def right():
    return {"ok": True, "state": ctl().set_vel(0, _unit(+15), 0)}

@router.post("/turnleft")
def turnleft():
    return {"ok": True, "state": ctl().set_vel(0, 0, _unit(-15))}

@router.post("/turnright")
def turnright():
    return {"ok": True, "state": ctl().set_vel(0, 0, _unit(+15))}
