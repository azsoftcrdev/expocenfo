import time, threading
from dataclasses import dataclass
from typing import Optional, Callable, Any
from MutoLib import Muto

@dataclass
class HeightSP:
    h:int = 0
    ts:float = time.time()
    gen:int = 0

class HeightController:
    def __init__(self, hz: float = 15.0, step: int = 2):
        self.bot = Muto()
        self._hz = max(5.0, float(hz))
        self._step = max(1, int(step))
        self._lock = threading.Lock()
        self._thr: Optional[threading.Thread] = None
        self._running = False
        self.sp = HeightSP()
        self._cur_h: int = 0
        self._resolved: Optional[Callable[[int], Any]] = None

    def start(self):
        if self._running: return
        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop_loop(self):
        self._running = False
        if self._thr and self._thr.is_alive():
            try: self._thr.join(timeout=0.5)
            except Exception: pass
        self._thr = None

    def set_height(self, h:int):
        h = max(-30, min(30, int(h)))
        with self._lock:
            self.sp.h = h; self.sp.ts = time.time(); self.sp.gen += 1
        return self.snapshot()

    def delta_height(self, dh:int):
        with self._lock:
            self.sp.h = max(-30, min(30, int(self.sp.h + dh)))
            self.sp.ts = time.time(); self.sp.gen += 1
        return self.snapshot()

    def set_step(self, step:int):
        self._step = max(1, int(step))
        return {"step": self._step}

    def snapshot(self):
        with self._lock:
            return {"h": self.sp.h, "cur_h": self._cur_h, "ts": self.sp.ts, "gen": self.sp.gen}

    # --- internals ---
    def _loop(self):
        period = 1.0 / self._hz
        self._resolve_method_once()
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                target = int(self.sp.h)
            self._step_towards(target)
            dt = period - (time.perf_counter() - t0)
            if dt > 0: time.sleep(min(dt, 0.02))

    def _resolve_method_once(self):
        if self._resolved: return
        for name in ("height","set_height","body_height","setBodyHeight","updown","set_updown"):
            fn = getattr(self.bot, name, None)
            if callable(fn):
                self._resolved = fn; break

    def _apply_height(self, h:int):
        if not self._resolved: return
        try: self._resolved(int(h))
        except TypeError:
            try: self._resolved(float(h))
            except Exception: pass
        except Exception: pass

    def _step_towards(self, target:int):
        if target == self._cur_h: return
        sign = 1 if target > self._cur_h else -1
        step = min(abs(target - self._cur_h), self._step)
        self._cur_h += sign * step
        self._apply_height(self._cur_h)

