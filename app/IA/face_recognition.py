# app/IA/face_recognition.py
import os
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ====== MediaPipe (CPU) ======
import mediapipe as mp


@dataclass
class FaceResult:
    frame: np.ndarray
    faces: List[Tuple[int, int, int, int]]  # (x,y,w,h) en píxeles del frame devuelto


class FaceDetector:
    """
    Uso:
        fd = FaceDetector()
        res = fd.process_frame(frame_bgr, draw=True)
        frame_out = res.frame
        boxes = res.faces

    Env vars:
        FACE_CONF (float, default 0.6)
        FACE_MAX_SIDE (int, default 640)
        FACE_FPS (float, default 12.0)
        FACE_COLOR (B,G,R, default "0,255,0")
        FACE_BLUR (0/1, default "0")
    """

    def __init__(self,
                 detection_confidence: Optional[float] = None,
                 max_side: Optional[int] = None,
                 worker_fps: Optional[float] = None) -> None:

        # Config desde .env o parámetros
        self.conf = float(os.getenv("FACE_CONF", "0.6")) if detection_confidence is None else float(detection_confidence)
        self.max_side = int(os.getenv("FACE_MAX_SIDE", "640")) if not max_side else int(max_side)
        self.worker_fps = float(os.getenv("FACE_FPS", "12")) if worker_fps is None else float(worker_fps)

        col = tuple(int(x) for x in os.getenv("FACE_COLOR", "0,255,0").split(","))
        self.color = (col + (0, 255, 0))[:3]
        self.blur = str(os.getenv("FACE_BLUR", "0")) == "1"

        # MediaPipe FaceDetection
        self._mp_face = mp.solutions.face_detection
        self._fd = self._mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.conf
        )

        # Estado del worker
        self._lock = threading.Lock()
        self._new_frame: Optional[np.ndarray] = None
        self._last_boxes: List[Tuple[int, int, int, int]] = []
        self._last_size: Tuple[int, int] = (0, 0)  # (H, W) del frame usado para la última detección
        self._last_fps: float = 0.0
        self._stop = False
        self._thread: Optional[threading.Thread] = None

        # Fuente para overlay
        self._font = cv2.FONT_HERSHEY_SIMPLEX

        # Arranca el worker lazy la primera vez que se llame process_frame()
        # (también puedes forzar con self.start())
        # Nada más aquí para no penalizar el arranque del server.

    # -------------- API pública --------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run_worker, name="FaceWorker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        t = self._thread
        if t:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None

    def process_frame(self, frame_bgr: np.ndarray, draw: bool = True) -> FaceResult:
        """
        - Envia el frame al worker (coalescing: solo el último).
        - Dibuja las últimas cajas disponibles (no espera a la inferencia).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceResult(frame_bgr, [])

        # arranque lazy del worker
        if not (self._thread and self._thread.is_alive()):
            self.start()

        # Enviar frame al worker (copia ligera para evitar data races)
        with self._lock:
            self._new_frame = frame_bgr.copy()

            # Tomamos las últimas cajas y el tamaño del frame con el que fueron calculadas
            boxes = list(self._last_boxes)
            H_det, W_det = self._last_size

        # Si el tamaño del frame actual difiere del usado en detección, escalamos cajas
        H_cur, W_cur = frame_bgr.shape[:2]
        if H_det > 0 and W_det > 0 and (H_det != H_cur or W_det != W_cur):
            sx = W_cur / float(W_det)
            sy = H_cur / float(H_det)
            boxes = [
                (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                for (x, y, w, h) in boxes
            ]

        out = frame_bgr
        if draw and boxes:
            for (x, y, w, h) in boxes:
                if self.blur:
                    roi = out[y:y + h, x:x + w]
                    if roi.size > 0:
                        roi = cv2.GaussianBlur(roi, (0, 0), 15)
                        out[y:y + h, x:x + w] = roi
                else:
                    cv2.rectangle(out, (x, y), (x + w, y + h), self.color, 2)

            # Debug overlay: #rostros y FPS IA
            with self._lock:
                ia_fps = self._last_fps
            dbg = f"faces={len(boxes)} | IA_FPS={ia_fps:.1f}"
            cv2.putText(out, dbg, (10, 22), self._font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, dbg, (10, 22), self._font, 0.55, self.color, 1, cv2.LINE_AA)

        return FaceResult(out, boxes)

    # -------------- Worker interno --------------

    def _resize_for_detection(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Devuelve (small_bgr, scale) donde 'scale' es factor small/original."""
        H, W = frame_bgr.shape[:2]
        if self.max_side and max(H, W) > self.max_side:
            scale = self.max_side / float(max(H, W))
            new_w = int(W * scale)
            new_h = int(H * scale)
            small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return small, scale
        return frame_bgr, 1.0

    def _run_worker(self) -> None:
        period = 1.0 / max(1e-6, float(self.worker_fps))
        next_t = time.time()
        ema = None

        while not self._stop:
            # Pace del worker
            now = time.time()
            if now < next_t:
                time.sleep(min(0.005, next_t - now))
                continue
            next_t = now + period

            # Toma el último frame disponible
            with self._lock:
                frame = self._new_frame
                self._new_frame = None

            if frame is None:
                continue

            try:
                H, W = frame.shape[:2]
                small_bgr, scale = self._resize_for_detection(frame)
                small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

                t0 = time.time()
                results = self._fd.process(small_rgb)
                dt = max(1e-6, time.time() - t0)
                fps = 1.0 / dt
                ema = fps if ema is None else (0.9 * ema + 0.1 * fps)

                boxes: List[Tuple[int, int, int, int]] = []
                if results and results.detections:
                    inv = 1.0 / scale
                    for det in results.detections:
                        rb = det.location_data.relative_bounding_box
                        # coords en la imagen "small"
                        x_s = rb.xmin * small_bgr.shape[1]
                        y_s = rb.ymin * small_bgr.shape[0]
                        w_s = rb.width * small_bgr.shape[1]
                        h_s = rb.height * small_bgr.shape[0]
                        # mapear a la original de este frame
                        x = int(x_s * inv); y = int(y_s * inv)
                        w = int(w_s * inv); h = int(h_s * inv)

                        # clip seguro
                        x = max(0, min(x, W - 1))
                        y = max(0, min(y, H - 1))
                        w = max(1, min(w, W - x))
                        h = max(1, min(h, H - y))
                        boxes.append((x, y, w, h))

                with self._lock:
                    self._last_boxes = boxes
                    self._last_size = (H, W)
                    self._last_fps = float(ema or fps)

            except Exception as e:
                # No queremos matar el worker por un frame malo
                with self._lock:
                    self._last_boxes = []
                print("[face_recognition] worker error:", e)
                continue

