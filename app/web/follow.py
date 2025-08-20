from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional

from ..motion.controller_vel import MotionControllerVel
from ..auto.follow import FollowController

router = APIRouter(prefix="/follow", tags=["follow"])

# Evita el operador | de Python 3.10 para compatibilidad 3.9
_ctrl: Optional[MotionControllerVel] = None
_fol: Optional[FollowController] = None

def ctl() -> MotionControllerVel:
    global _ctrl
    if _ctrl is None:
        _ctrl = MotionControllerVel(hz=15.0, deadman_s=0.8)
        _ctrl.start()
    return _ctrl

def fol() -> FollowController:
    global _fol
    if _fol is None:
        _fol = FollowController()
    return _fol

# ------- Schemas -------
class StartBody(BaseModel):
    speed: Optional[conint(ge=1, le=5)] = Field(default=None, description="Velocidad UI 1..5")
    standoff: Optional[confloat(gt=0)] = None
    band: Optional[confloat(gt=0)] = None

class TargetBody(BaseModel):
    # bearing en grados: 0 = frente, +derecha, -izquierda
    bearing_deg: confloat(ge=-180, le=180)
    conf: Optional[confloat(ge=0.0, le=1.0)] = None  # opcional

class ConfigBody(BaseModel):
    standoff: Optional[confloat(gt=0)] = None
    band: Optional[confloat(gt=0)] = None
    kp_turn: Optional[confloat(gt=0)] = None
    kp_fwd: Optional[confloat(gt=0)] = None
    max_z_deg: Optional[confloat(gt=0)] = None
    tgt_win_deg: Optional[confloat(gt=0)] = None
    min_quality: Optional[conint(ge=0, le=255)] = None
    avoid_stop: Optional[confloat(gt=0)] = None
    avoid_slow: Optional[confloat(gt=0)] = None
    k_avoid: Optional[confloat()] = None
    speed: Optional[conint(ge=1, le=5)] = None

# ------- Endpoints -------
@router.post("/start")
def follow_start(body: StartBody):
    f = fol()
    if body.speed is not None:
        f.set_speed_ui(int(body.speed))
    f.set_params(standoff=body.standoff, band=body.band)
    f.start(ctl())
    return {"ok": True, "status": f.snapshot()}

@router.post("/stop")
def follow_stop():
    global _fol, _ctrl
    if _fol:
        _fol.stop()
    if _ctrl:
        _ctrl.stop()
    return {"ok": True}

@router.post("/target")
def follow_target(body: TargetBody):
    f = fol()
    if not f.is_running():
        raise HTTPException(400, "Follow no está corriendo. Llama /follow/start primero.")
    f.update_target_bearing(float(body.bearing_deg))
    return {"ok": True}

@router.post("/config")
def follow_config(body: ConfigBody):
    f = fol()
    if body.speed is not None:
        f.set_speed_ui(int(body.speed))
    # pasa el resto de parámetros excepto speed
    f.set_params(**{k: v for k, v in body.dict().items() if v is not None and k != "speed"})
    return {"ok": True, "status": f.snapshot()}

@router.get("/status")
def follow_status():
    if _fol:
        return {"ok": True, "status": _fol.snapshot()}
    return {"ok": False, "status": {"running": False}}

