# app/sensors/camera.py
import os, time, threading
from typing import Optional, Tuple, Union
from pathlib import Path
import cv2
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / "config" / ".env"
load_dotenv(ENV_PATH)

def _parse_device(dev: str) -> Union[int, str]:
    try:
        return int(dev)
    except (TypeError, ValueError):
        return dev or 0

class Camera:
    def __init__(self):
        # AHORA sí lee .env (CAMERA_DEVICE puede ser "0" o "/dev/video0")
        dev_str = os.getenv("CAMERA_DEVICE", "0")
        self.device = _parse_device(dev_str)

        self.width  = int(os.getenv("CAMERA_WIDTH", "640"))
        self.height = int(os.getenv("CAMERA_HEIGHT", "480"))
        self.fps    = int(os.getenv("CAMERA_FPS", "30"))
        self.codec  = os.getenv("CAMERA_CODEC", "MJPG").strip().upper()

        self._cap: Optional[cv2.VideoCapture] = None

        # doble buffer para minimizar tiempo bajo lock
        self._last_frame = None
        self._last_frame_copy = None

        self._t_last = time.time()
        self._fps_actual = 0.0

        # hilo de captura
        self._grab_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # watchdog
        self._no_frame_count = 0
        self._no_frame_limit = 50  # ~ 1–2s según FPS

    def open(self):
        if self._cap is not None:
            return

        # IMPORTANTE: en V4L2, setear FOURCC antes del tamaño/FPS
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if self.codec == "MJPG":
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        # reducir buffer (algunos drivers lo ignoran, pero ayuda cuando aplica)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara: {self.device}")

        # warm-up: descarta algunos frames iniciales
        for _ in range(5):
            self._cap.read()

        # inicia hilo de captura
        if not self._running:
            self._running = True
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._grab_thread.start()

    def _reopen(self):
        # Cierra y reabre en caso de falla prolongada
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None
        time.sleep(0.1)
        try:
            self.open()
        except Exception:
            # Si no puede reabrir, reintenta luego
            pass

    def _grab_loop(self):
        # Captura continua, guarda SOLO el último frame (doble buffer)
        while self._running:
            if self._cap is None:
                time.sleep(0.02)
                continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._no_frame_count += 1
                if self._no_frame_count >= self._no_frame_limit:
                    self._reopen()
                    self._no_frame_count = 0
                time.sleep(0.005)
                continue

            self._no_frame_count = 0
            t = time.time()
            with self._lock:
                self._last_frame = frame
                dt = max(t - self._t_last, 1e-6)
                self._fps_actual = 1.0 / dt
                self._t_last = t

    def read_latest(self):
        # devuelve la última imagen sin bloquear la captura
        if self._cap is None:
            self.open()

        # copia fuera del lock: baja contención
        with self._lock:
            if self._last_frame is None:
                raise RuntimeError("No hay frame todavía")
            # crea buffer persistente y copia
            if self._last_frame_copy is None or self._last_frame_copy.shape != self._last_frame.shape:
                self._last_frame_copy = self._last_frame.copy()
            else:
                self._last_frame_copy[:] = self._last_frame
            out = self._last_frame_copy

        # devolvemos una copia para seguridad aguas arriba
        return out.copy()

    # compat anterior
    def read(self):
        return self.read_latest()

    def get_fps_actual(self) -> float:
        return float(self._fps_actual)

    def get_resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def snapshot_jpeg(self, quality: int = 80) -> bytes:
        frame = self.read_latest()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("No se pudo codificar JPG")
        return buf.tobytes()

    def release(self):
        self._running = False
        # espera a que termine el hilo
        if self._grab_thread is not None:
            try:
                self._grab_thread.join(timeout=0.5)
            except Exception:
                pass
            self._grab_thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def get_telemetry_snapshot(self) -> dict:
        try:
            w, h = self.get_resolution()
        except Exception:
            w, h = 0, 0
        return {
            "fps": round(float(self.get_fps_actual()), 2),
            "resolution": [w, h],
            "ts": time.time(),
        }

    # (Opcional) Ajuste de controles V4L2 comunes
    def set_control(self, prop: int, value: float) -> bool:
        try:
            return bool(self._cap and self._cap.set(prop, value))
        except Exception:
            return False

camera = Camera()

def get_telemetry_snapshot() -> dict:
    return camera.get_telemetry_snapshot()

