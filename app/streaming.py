# app/streaming.py
import cv2, numpy as np, time, threading
from typing import Optional, Callable, Tuple
from .sensors.camera import camera

# === TurboJPEG opcional (si está instalado acelera el JPG) ===
try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
except Exception:
    _jpeg = None

BOUNDARY_NAME = "frame"
BOUNDARY_BYTES = b"--" + BOUNDARY_NAME.encode()

def _parse_size(size: Optional[str]) -> Optional[Tuple[int, int]]:
    if not size:
        return None
    try:
        w, h = size.lower().split("x")
        return (int(w), int(h))
    except Exception:
        return None

# Placeholders (si la IA no está disponible no rompe nada)
class _NoOp:
    def set_current_color(self, *_a, **_k): pass
    def process_frame(self, frame, *a, **k): return type("R", (), {"frame": frame})

# Carga módulos IA (opcionales)
try:
    from .IA import ColorRecognizer
    _recog = ColorRecognizer()
except Exception:
    _recog = _NoOp()

try:
    from .IA.face_recognition import FaceDetector
    _face = FaceDetector()
except Exception:
    _face = _NoOp()

# --- PERSON: usa worker en background (rápido, no bloquea streaming) ---
try:
    from .IA.person_detection import (
        start_person_detection_worker,
        get_last_person_boxes,
        overlay_person_boxes,
    )
    _person_ok = True
except Exception as e:
    print("[WARN] person_detection no disponible:", e)
    _person_ok = False

# ─────────────────────────────────────────────────────────────
# Encoder global (para mode="none") — 1 solo hilo produce JPG,
# todos los clientes leen el último frame (casi 0 CPU extra).
# ─────────────────────────────────────────────────────────────
class _Encoder:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpg: Optional[bytes] = None
        self._seq: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.fps = 15.0
        self.quality = 70
        self.size: Optional[Tuple[int,int]] = (640, 360)

    def configure(self, fps: float, quality: int, size: Optional[str]):
        self.fps = max(1.0, min(60.0, float(fps)))
        self.quality = int(max(40, min(95, int(quality))))
        self.size = _parse_size(size)

    def start(self):
        if self._running:
            return
        camera.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _encode_jpg(self, frame) -> Optional[bytes]:
        if _jpeg is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return _jpeg.encode(rgb, quality=self.quality)
            except Exception:
                pass
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if ok else None

    def _loop(self):
        period = 1.0 / self.fps
        next_t = time.time()
        while self._running:
            try:
                frame = camera.read_latest()
            except Exception:
                time.sleep(0.01)
                continue

            if self.size:
                try:
                    frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
                except Exception:
                    pass

            jpg = self._encode_jpg(frame)
            if jpg:
                with self._lock:
                    self._jpg = jpg
                    self._seq = (self._seq + 1) & 0x7FFFFFFF

            # pace
            next_t += period
            dt = next_t - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.02))
            else:
                next_t = time.time()

    def latest(self) -> Tuple[Optional[bytes], int]:
        with self._lock:
            return self._jpg, self._seq

# singleton del encoder compartido
_encoder = _Encoder()

# ─────────────────────────────────────────────────────────────
# Generador MJPEG
#  - mode="none"   → encoder global (más eficiente)
#  - mode="color"  → resaltado de color
#  - mode="face"   → detección de rostros
#  - mode="person" → detección de personas (NUEVO, con worker)
# ─────────────────────────────────────────────────────────────
def mjpeg_generator(
    mode: Optional[str] = "none",
    color: Optional[str] = None,
    overlay: bool = True,
    quality: int = 70,
    fps: float = 15.0,
    size: Optional[str] = None,
    is_disconnected: Optional[Callable[[], bool]] = None
):
    camera.open()

    _mode = (mode or "none").lower()
    if _mode not in ("none", "color", "face", "person"):
        _mode = "none"

    # === Camino rápido y compartido ===
    if _mode == "none":
        _encoder.configure(fps=fps, quality=quality, size=size)
        _encoder.start()

        last_seq = -1
        while True:
            if is_disconnected and is_disconnected():
                break

            jpg, seq = _encoder.latest()
            if jpg is None:
                time.sleep(0.01)
                continue

            if seq != last_seq:
                last_seq = seq
                header = (
                    BOUNDARY_BYTES +
                    b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                    str(len(jpg)).encode() + b"\r\n\r\n"
                )
                yield header + jpg + b"\r\n"
            else:
                time.sleep(0.005)
        return

    # === Camino con procesamiento por cliente ===
    if _mode == "color":
        try:
            _recog.set_current_color(color)
        except Exception:
            pass

    # PERSON: arranca worker una vez (si existe)
    if _mode == "person" and _person_ok:
        # 6–10 FPS es razonable para ARM; ajusta si quieres
        start_person_detection_worker(fps=8)

    target = max(1.0, min(60.0, float(fps)))
    period = 1.0 / target
    next_t = time.time()
    out_size = _parse_size(size)

    while True:
        if is_disconnected and is_disconnected():
            break

        try:
            frame = camera.read_latest()
        except Exception:
            time.sleep(0.01)
            continue

        if out_size:
            try:
                frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            except Exception:
                pass

        if _mode == "color":
            try:
                res = _recog.process_frame(frame)
                frame_out = res.frame if overlay else frame
            except Exception:
                frame_out = frame
        elif _mode == "face":
            try:
                fres = _face.process_frame(frame, draw=overlay)
                frame_out = fres.frame
            except Exception:
                frame_out = frame
        else:  # --- person (rápido: solo dibuja resultados del worker) ---
            if _person_ok:
                try:
                    boxes = get_last_person_boxes()
                    frame_out = overlay_person_boxes(frame, boxes) if overlay else frame
                except Exception:
                    frame_out = frame
            else:
                frame_out = frame

        # Encode por cliente (TurboJPEG si existe)
        if _jpeg is not None:
            try:
                rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                jpg = _jpeg.encode(rgb, quality=int(quality))
            except Exception:
                jpg = None
        else:
            ok, buf = cv2.imencode(".jpg", frame_out, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            jpg = buf.tobytes() if ok else None

        if jpg is None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if not ok:
                time.sleep(0.01); continue
            jpg = buf.tobytes()

        header = (
            BOUNDARY_BYTES +
            b"\r\nContent-Type: image/jpeg\r\nContent-Length: " +
            str(len(jpg)).encode() + b"\r\n\r\n"
        )
        yield header + jpg + b"\r\n"

        next_t += period
        dt = next_t - time.time()
        if dt > 0:
            time.sleep(min(dt, 0.02))
        else:
            next_t = time.time()

