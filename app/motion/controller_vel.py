# app/motion/controller_vel.py
import time, threading
from dataclasses import dataclass
from typing import Optional
from MutoLib import Muto

@dataclass
class VelSP:
    x:int=0; y:int=0; z:int=0
    speed:int=3                 # UI: 1..5 (5 = más rápido en nuestra UI)
    ts:float=time.time()
    gen:int=0

class MotionControllerVel:
    """
    - Bucle de control en HILO dedicado (no bloquea asyncio).
    - Coalescing latest-wins (set_vel solo actualiza sp).
    - Llama a Muto.move() continuamente (requerido por Yahboom).
    - Llama a Muto.speed() solo cuando cambia (rate-limited).
    - Dead-man: si no hay updates, se detiene con stay_put().
    - Mapea velocidad UI (1..5, 5=rápido) → driver (1..5, 1=rápido).
    """
    def __init__(self, hz:float=15.0, deadman_s:float=0.8):
        self.bot = Muto()                 # abrir puerto serie aquí (puede tardar, pero solo una vez)
        self.sp = VelSP()
        self._lock = threading.Lock()
        self._hz = max(5.0, float(hz))    # 15 Hz por defecto
        self._deadman = float(deadman_s)
        self._thr: Optional[threading.Thread] = None
        self._running = False

        # cache para rate-limit y estado
        self._last_speed_ui: Optional[int] = None
        self._last_speed_ts: float = 0.0
        self._was_active: bool = False    # para no spamear stay_put()

    # ------- API pública (latest-wins) -------
    def start(self, _loop=None):
        if self._running: return
        self._running = True
        self._thr = threading.Thread(target=self._loop_thread, daemon=True)
        self._thr.start()

    def stop_loop(self):
        self._running = False
        if self._thr and self._thr.is_alive():
            try:
                self._thr.join(timeout=0.5)
            except Exception:
                pass
            self._thr = None

    def set_vel(self, x:int, y:int, z:int, speed:Optional[int]=None):
        x = max(-30, min(30, int(x)))
        y = max(-30, min(30, int(y)))
        z = max(-30, min(30, int(z)))
        with self._lock:
            self.sp.x, self.sp.y, self.sp.z = x, y, z
            if speed is not None:
                self.sp.speed = max(1, min(5, int(speed)))  # UI 1..5 (5=rápido)
            self.sp.ts = time.time()
            self.sp.gen += 1
        return self.snapshot()

    def stop(self):
        return self.set_vel(0, 0, 0)

    def snapshot(self):
        with self._lock:
            return dict(x=self.sp.x, y=self.sp.y, z=self.sp.z,
                        speed=self.sp.speed, ts=self.sp.ts, gen=self.sp.gen)

    # ------- Bucle en HILO dedicado -------
    def _loop_thread(self):
        period = 1.0 / self._hz
        while self._running:
            t0 = time.perf_counter()
            # toma snapshot sin bloquear mucho
            with self._lock:
                sp = VelSP(**self.sp.__dict__)
            now = time.time()

            try:
                # Dead-man: si no hay vida, bajar piernas una vez y no spamear
                if (now - sp.ts) > self._deadman:
                    if self._was_active:
                        try:
                            self.bot.stay_put()
                        except Exception:
                            pass
                        self._was_active = False
                    self._rate_sleep(t0, period)
                    continue

                # Mapear velocidad UI (5=rápido) → driver (1=rápido)
                # La librería Yahboom invierte internamente (5-level),
                # pero su API espera 1..5 donde 1 es más rápido.
                ui_speed = sp.speed
                drv_speed = max(1, min(5, 6 - ui_speed))  # 5→1 (rápido), 1→5 (lento)

                # Rate-limit de speed(): solo si cambió y no más de ~5Hz
                if (self._last_speed_ui != ui_speed) and ((now - self._last_speed_ts) > 0.2):
                    try:
                        self.bot.speed(drv_speed)
                    except Exception:
                        pass
                    self._last_speed_ui = ui_speed
                    self._last_speed_ts = now

                # Activity: ¿hay movimiento?
                active = bool(sp.x or sp.y or sp.z)

                if active:
                    # Llamada CONTINUA requerida por MutoLib
                    try:
                        self.bot.move(sp.x, sp.y, sp.z)
                    except Exception:
                        pass
                    self._was_active = True
                else:
                    # Si veníamos activos y ahora no, una sola stay_put()
                    if self._was_active:
                        try:
                            self.bot.stay_put()
                        except Exception:
                            pass
                        self._was_active = False

            except Exception:
                # evita que cualquier excepción mate el hilo
                pass

            self._rate_sleep(t0, period)

    @staticmethod
    def _rate_sleep(t0: float, period: float):
        dt = period - (time.perf_counter() - t0)
        if dt > 0:
            # cap a 20ms para no “derrape” de fase
            time.sleep(min(dt, 0.02))

