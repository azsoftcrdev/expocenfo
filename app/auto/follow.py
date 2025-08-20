# app/auto/follow.py
import time, math, threading, os
from typing import Optional, Dict, Any

import numpy as np

from ..sensors.lidar import lidar
from ..motion.controller_vel import MotionControllerVel

def _env_float(k: str, d: float) -> float:
    v = os.getenv(k, "")
    try: return float(v) if v != "" else d
    except: return d

def _env_int(k: str, d: int) -> int:
    v = os.getenv(k, "")
    try: return int(v) if v != "" else d
    except: return d

# ====== Parámetros por .env ======
FOLLOW_HZ                = _env_float("FOLLOW_HZ", 12.0)
FOLLOW_STANDOFF_M        = _env_float("FOLLOW_STANDOFF_M", 1.0)   # distancia objetivo a la persona
FOLLOW_BAND_M            = _env_float("FOLLOW_BAND_M", 0.2)       # banda muerta en distancia
FOLLOW_KP_TURN           = _env_float("FOLLOW_KP_TURN", 0.9)      # deg → z_cmd
FOLLOW_KP_FWD            = _env_float("FOLLOW_KP_FWD", 0.8)       # (m) → x_cmd scaling (0..base)
FOLLOW_MAX_Z_DEG         = _env_float("FOLLOW_MAX_Z_DEG", 25.0)   # límite de giro (deg mapeado a 30)
FOLLOW_TARGET_WIN_DEG    = _env_float("FOLLOW_TARGET_WIN_DEG", 10.0)  # ventana para medir rango a objetivo
FOLLOW_MIN_QUALITY       = _env_int("FOLLOW_MIN_QUALITY", 0)      # calidad LiDAR
FOLLOW_LOST_TIMEOUT_S    = _env_float("FOLLOW_LOST_TIMEOUT_S", 0.6)
FOLLOW_SCAN_Z_CMD        = _env_int("FOLLOW_SCAN_Z_CMD", 8)       # giro lento al buscar
FOLLOW_SPEED_UI_DEFAULT  = _env_int("FOLLOW_SPEED_UI_DEFAULT", 3) # 1..5 base
# Evasión
AVOID_STOP_DIST          = _env_float("AVOID_STOP_DIST", 0.35)
AVOID_SLOW_DIST          = _env_float("AVOID_SLOW_DIST", 0.60)
AVOID_K_YAW              = _env_float("AVOID_K_YAW", 0.6)         # sesgo de giro por despeje lateral
AVOID_FRONT_WIN_DEG      = _env_float("AVOID_FRONT_WIN_DEG", 30.0)
AVOID_SIDE_WIN_DEG       = _env_float("AVOID_SIDE_WIN_DEG", 30.0)

def _ui_speed_to_cmd(speed_ui: int) -> int:
    s = max(1, min(5, int(speed_ui)))
    return min(30, 6 * s)  # 1..5 → 6,12,18,24,30

def _clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))

class FollowController:
    """
    Sigue a un objetivo dado por 'bearing_deg' (actualizado por la cámara)
    y evita obstáculos con el LiDAR. Controla MotionControllerVel (x,z).
    """
    def __init__(self):
        self.hz = FOLLOW_HZ
        self.standoff = FOLLOW_STANDOFF_M
        self.band = FOLLOW_BAND_M
        self.kp_turn = FOLLOW_KP_TURN
        self.kp_fwd  = FOLLOW_KP_FWD
        self.max_z_deg = FOLLOW_MAX_Z_DEG
        self.tgt_win_deg = FOLLOW_TARGET_WIN_DEG
        self.min_qual = FOLLOW_MIN_QUALITY
        self.lost_timeout = FOLLOW_LOST_TIMEOUT_S
        self.scan_z_cmd = int(FOLLOW_SCAN_Z_CMD)

        self.avoid_stop = AVOID_STOP_DIST
        self.avoid_slow = AVOID_SLOW_DIST
        self.k_avoid    = AVOID_K_YAW
        self.front_win  = AVOID_FRONT_WIN_DEG
        self.side_win   = AVOID_SIDE_WIN_DEG

        self.speed_ui = FOLLOW_SPEED_UI_DEFAULT

        self._bearing_deg: Optional[float] = None
        self._bearing_ts: float = 0.0

        self._ctl: Optional[MotionControllerVel] = None
        self._thr: Optional[threading.Thread] = None
        self._running: bool = False

        self._last_decision: Dict[str, Any] = {}

    # ------------- ciclo -------------
    def start(self, ctl: Optional[MotionControllerVel] = None):
        if self._running: return
        self._ctl = ctl or MotionControllerVel(hz=15.0, deadman_s=0.8)
        self._ctl.start()
        self._running = True
        self._thr = threading.Thread(target=self._loop, name="FollowController", daemon=True)
        self._thr.start()

    def stop(self):
        self._running = False
        t = self._thr
        if t and t.is_alive():
            try: t.join(timeout=0.5)
            except: pass
        self._thr = None
        try:
            if self._ctl: self._ctl.stop()
        except: pass

    def is_running(self) -> bool:
        return self._running

    # ------------- API -------------
    def set_speed_ui(self, s: int):
        self.speed_ui = max(1, min(5, int(s)))

    def set_params(self, **kw):
        for k,v in kw.items():
            if v is None: continue
            if k == "standoff": self.standoff = float(v)
            elif k == "band": self.band = float(v)
            elif k == "kp_turn": self.kp_turn = float(v)
            elif k == "kp_fwd": self.kp_fwd = float(v)
            elif k == "max_z_deg": self.max_z_deg = float(v)
            elif k == "tgt_win_deg": self.tgt_win_deg = float(v)
            elif k == "min_quality": self.min_qual = int(v)
            elif k == "avoid_stop": self.avoid_stop = float(v)
            elif k == "avoid_slow": self.avoid_slow = float(v)
            elif k == "k_avoid": self.k_avoid = float(v)

    def update_target_bearing(self, bearing_deg: float):
        # Normaliza -180..180
        b = float(((bearing_deg + 180.0) % 360.0) - 180.0)
        self._bearing_deg = b
        self._bearing_ts  = time.time()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "speed_ui": self.speed_ui,
            "params": {
                "standoff": self.standoff, "band": self.band,
                "kp_turn": self.kp_turn, "kp_fwd": self.kp_fwd,
                "max_z_deg": self.max_z_deg, "tgt_win_deg": self.tgt_win_deg,
                "min_quality": self.min_qual,
                "avoid": {"stop": self.avoid_stop, "slow": self.avoid_slow, "k_yaw": self.k_avoid},
            },
            "target": {"bearing_deg": self._bearing_deg, "age_s": None if self._bearing_ts==0 else time.time()-self._bearing_ts},
            "decision": self._last_decision,
        }

    # ------------- loop -------------
    def _loop(self):
        period = 1.0 / max(1.0, self.hz)
        while self._running:
            t0 = time.perf_counter()
            try:
                self._step()
            except Exception as e:
                self._last_decision = {"error": str(e)}
                try:
                    if self._ctl: self._ctl.stop()
                except: pass
            dt = period - (time.perf_counter() - t0)
            if dt > 0: time.sleep(min(dt, 0.02))

    # ------------- lógica -------------
    def _step(self):
        now = time.time()
        bearing = self._bearing_deg
        age = (now - self._bearing_ts) if self._bearing_ts>0 else 1e9

        # Rangos LiDAR por sectores (para evitar)
        ranges = lidar().get_sector_ranges(sectors=16, min_quality=self.min_qual)  # 16→22.5°/sector
        def _sector_clear(lo_deg: float, hi_deg: float) -> float:
            # usa get_min_distance del LiDAR
            d = lidar().get_min_distance(lo_deg%360, hi_deg%360, min_quality=self.min_qual)
            return float(d)

        # Distancia estimada al objetivo según LiDAR en ventana alrededor del bearing
        def _dist_to_target(bd: float) -> float:
            win = self.tgt_win_deg * 0.5
            return _sector_clear(bd - win, bd + win)

        base = _ui_speed_to_cmd(self.speed_ui)

        # Si no tenemos bearing reciente → “buscar”
        if (bearing is None) or (age > self.lost_timeout):
            x_cmd = 0
            z_cmd = self.scan_z_cmd if (bearing is None or bearing >= 0) else -self.scan_z_cmd
            if ranges.size and np.isfinite(ranges[0]):
                front_cl = _sector_clear(-self.front_win/2, +self.front_win/2)
                if front_cl <= self.avoid_stop: x_cmd = 0
            self._apply_cmd(x_cmd, z_cmd)
            self._last_decision = {
                "state": "searching",
                "bearing": bearing, "age_s": age,
                "x_cmd": x_cmd, "z_cmd": z_cmd
            }
            return

        # Control de giro hacia el objetivo
        z_from_target = self.kp_turn * float(bearing)
        z_from_target = _clamp(z_from_target, -self.max_z_deg, +self.max_z_deg)

        # Distancia a objetivo
        d_tgt = _dist_to_target(bearing)
        # Política de avance
        if not math.isfinite(d_tgt):
            # no se ve con LiDAR → avanza despacio manteniendo giro
            x_cmd = int(0.4 * base)
        else:
            if d_tgt <= self.standoff - self.band:
                x_cmd = 0  # no retrocedemos de momento (puedes poner -0.3*base)
            elif d_tgt >= self.standoff + self.band:
                # avanza proporcional a cuán lejos está, cap a base
                gain = _clamp((d_tgt - (self.standoff + self.band)) / max(0.01, self.standoff), 0.2, 1.0)
                x_cmd = int(self.kp_fwd * gain * base)
            else:
                x_cmd = int(0.2 * base)  # dentro de banda → avance suave

        # Evasión de obstáculos al frente y sesgo lateral
        front_cl = _sector_clear(-self.front_win/2, +self.front_win/2)
        left_cl  = _sector_clear(+60 - self.side_win/2, +60 + self.side_win/2)
        right_cl = _sector_clear(-60 - self.side_win/2, -60 + self.side_win/2)

        if math.isfinite(front_cl):
            if front_cl <= self.avoid_stop:
                x_cmd = 0
            elif front_cl <= self.avoid_slow:
                x_cmd = int(x_cmd * _clamp((front_cl - self.avoid_stop)/max(1e-3, self.avoid_slow-self.avoid_stop), 0.2, 0.9))

        # Sesgo de giro hacia el lado más despejado
        if math.isfinite(left_cl) and math.isfinite(right_cl):
            bias = self.k_avoid * (right_cl - left_cl)  # positivo → gira a la derecha
            z_cmd = int(_clamp(z_from_target + bias, -30, 30))
        else:
            z_cmd = int(_clamp(z_from_target, -30, 30))

        self._apply_cmd(x_cmd, z_cmd)
        self._last_decision = {
            "state": "tracking",
            "bearing_deg": bearing, "age_s": round(age, 3),
            "d_target_m": None if not math.isfinite(d_tgt) else round(d_tgt, 3),
            "front_clear_m": None if not math.isfinite(front_cl) else round(front_cl, 3),
            "left_clear_m": None if not math.isfinite(left_cl) else round(left_cl, 3),
            "right_clear_m": None if not math.isfinite(right_cl) else round(right_cl, 3),
            "cmd": {"x": x_cmd, "z": z_cmd},
        }

    def _apply_cmd(self, x: int, z: int):
        try:
            if self._ctl:
                self._ctl.set_vel(int(x), 0, int(z), speed=None)
        except: pass

