# app/web/routes_motion.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import Optional, Literal
from MutoLib import Muto
import threading, time

router = APIRouter(prefix="/motion", tags=["motion"])
g_bot = Muto()

class MotionState:
    def __init__(self):
        self.active: Optional[Literal["left","right"]] = None
        self.step: int = 20
        self.speed: float = 2.0
        self._lock = threading.Lock()
motion = MotionState()

class Deadman:
    def __init__(self, timeout_s=1.0):
        self.timeout_s = timeout_s
        self._last = time.time()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()
    def tick(self): self._last = time.time()
    def _loop(self):
        while not self._stop:
            if time.time() - self._last > self.timeout_s:
                try: g_bot.stop()
                except: pass
            time.sleep(0.05)
deadman = Deadman(timeout_s=1.0)

class StepBody(BaseModel):
    value: conint(ge=10, le=25)

class SpeedBody(BaseModel):
    value: confloat(gt=0)

def _apply_active():
    g_bot.speed(motion.speed)
    if motion.active == "left":
        g_bot.left(motion.step)
    elif motion.active == "right":
        g_bot.right(motion.step)
    deadman.tick()

@router.post("/stop")
def stop():
    with motion._lock:
        motion.active = None
    try:
        g_bot.stop()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, f"Stop failed: {e}")

@router.post("/left")
def left(body: StepBody):
    with motion._lock:
        motion.step = int(body.value)
        motion.active = "left"
    _apply_active()
    return {"ok": True, "active": "left", "step": motion.step, "speed": motion.speed}

@router.post("/right")
def right(body: StepBody):
    with motion._lock:
        motion.step = int(body.value)
        motion.active = "right"
    _apply_active()
    return {"ok": True, "active": "right", "step": motion.step, "speed": motion.speed}

@router.post("/step")
def set_step(body: StepBody):
    with motion._lock:
        motion.step = int(body.value)
    if motion.active in ("left","right"):
        _apply_active()
    else:
        deadman.tick()
    return {"ok": True, "step": motion.step, "active": motion.active}

@router.post("/speed")
def set_speed(body: SpeedBody):
    with motion._lock:
        motion.speed = float(body.value)
    if motion.active in ("left","right"):
        _apply_active()
    else:
        g_bot.speed(motion.speed)
        deadman.tick()
    return {"ok": True, "speed": motion.speed, "active": motion.active}

