from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# ----------------------------
# Rangos HSV por color (OpenCV: H[0-180], S[0-255], V[0-255])
# Nota: rojo suele requerir DOS rangos (wrap-around del H).
# ----------------------------
DEFAULT_HSV_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    # (minH,minS,minV), (maxH,maxS,maxV)
    "red":    [((0, 43, 46), (10, 255, 255)), ((156, 43, 46), (180, 255, 255))],
    "yellow": [((26, 43, 46), (34, 255, 255))],
    "green":  [((35, 43, 46), (77, 255, 255))],
    "blue":   [((100, 43, 46), (124, 255, 255))],
}

# ----------------------------
# Utilidades para persistir configuración
# ----------------------------
def load_hsv_config(path: str) -> Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normaliza a tuplas
    norm: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = {}
    for k, ranges in data.items():
        norm[k] = [ (tuple(r[0]), tuple(r[1])) for r in ranges ]
    return norm

def save_hsv_config(path: str, ranges: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)

# ----------------------------
# Core
# ----------------------------
@dataclass
class DetectResult:
    color: Optional[str]
    mask: Optional[np.ndarray]
    frame: np.ndarray
    scores: Dict[str, int]  # pixeles encendidos por color (ROI)

class ColorRecognizer:
    """
    Detector de color por HSV con:
    - ROI opcional (x, y, w, h) para estabilizar detección
    - Modo 'auto' (elige el color con mayor respuesta en la ROI)
    - Soporte para múltiples rangos por color (p. ej. rojo)
    """
    def __init__(
        self,
        hsv_ranges: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = None,
        roi: Optional[Tuple[int,int,int,int]] = None,   # (x, y, w, h)
        blur_ksize: int = 5,
        morph_kernel: int = 3,
    ) -> None:
        self.hsv_ranges = hsv_ranges.copy() if hsv_ranges else DEFAULT_HSV_RANGES.copy()
        self.roi = roi  # si es None, se usa un cuadro central por defecto
        self.blur_ksize = blur_ksize
        self.morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.current_color: Optional[str] = None  # si se fija, solo evalúa ese color; si None → auto

    # ----------------------------
    # Configuración
    # ----------------------------
    def set_roi(self, roi: Optional[Tuple[int,int,int,int]]) -> None:
        self.roi = roi

    def set_current_color(self, color: Optional[str]) -> None:
        """Si `color` es None, se activa el modo auto."""
        if color is not None and color not in self.hsv_ranges:
            raise ValueError(f"Color desconocido: {color}")
        self.current_color = color

    def update_range(self, color: str, ranges: List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]) -> None:
        self.hsv_ranges[color] = ranges

    # ----------------------------
    # Procesamiento
    # ----------------------------
    def _ensure_roi(self, frame: np.ndarray) -> Tuple[int,int,int,int]:
        if self.roi is not None:
            return self.roi
        # ROI central por defecto (un cuadrado ~25% del ancho/alto)
        h, w = frame.shape[:2]
        rw, rh = int(w * 0.25), int(h * 0.25)
        x = (w - rw) // 2
        y = (h - rh) // 2
        return (x, y, rw, rh)

    def _mask_for_color(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """Combina múltiples rangos para un mismo color (OR)."""
        ranges = self.hsv_ranges[color]
        masks = []
        for (minv, maxv) in ranges:
            lower = np.array(minv, dtype=np.uint8)
            upper = np.array(maxv, dtype=np.uint8)
            masks.append(cv2.inRange(hsv, lower, upper))
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        # morfología para limpiar ruido
        mask = cv2.medianBlur(mask, self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize+1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph, iterations=1)
        return mask

    def process_frame(self, frame_bgr: np.ndarray) -> DetectResult:
        frame = frame_bgr.copy()
        x, y, w, h = self._ensure_roi(frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        colors_to_check = [self.current_color] if self.current_color else list(self.hsv_ranges.keys())
        scores: Dict[str, int] = {}
        best_color: Optional[str] = None
        best_score = -1
        best_mask_roi: Optional[np.ndarray] = None

        for color in colors_to_check:
            if color is None:
                continue
            mask_roi = self._mask_for_color(hsv, color)
            score = int(cv2.countNonZero(mask_roi))
            scores[color] = score
            if score > best_score:
                best_score = score
                best_color = color
                best_mask_roi = mask_roi

        # Construye la máscara completa (frame-size) para el color ganador
        full_mask = None
        if best_color is not None and best_mask_roi is not None:
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = best_mask_roi
            # Visual: sobreponer resultado
            colored = cv2.bitwise_and(frame, frame, mask=full_mask)
            overlay = frame.copy()
            cv2.putText(overlay, f"Color: {best_color}", (x, max(30, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            # mezcla suave
            frame = cv2.addWeighted(colored, 0.8, frame, 0.5, 0)

        return DetectResult(color=best_color, mask=full_mask, frame=frame, scores=scores)

# ----------------------------
# Helper funcional sencillo
# ----------------------------
def detect_color(frame_bgr: np.ndarray, recognizer: Optional[ColorRecognizer] = None) -> DetectResult:
    if recognizer is None:
        recognizer = ColorRecognizer()
    return recognizer.process_frame(frame_bgr)

# ----------------------------
# Demo local opcional: abre webcam y muestra detección
# (puedes ejecutar: python -m app.detections.color_recognition)
# ----------------------------
def _try_set_mjpg(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

def _open_camera(index: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    _try_set_mjpg(cap)
    return cap

def main_demo():
    recog = ColorRecognizer()
    cap = _open_camera()
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    print("[ColorRecognition] Demo iniciada. Tecla 'q' para salir, 'r/g/b/y' para fijar color o 'a' para modo auto.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = recog.process_frame(frame)

        # cabecera con puntuaciones
        y0 = 30
        for k, v in sorted(res.scores.items(), key=lambda x: -x[1]):
            cv2.putText(res.frame, f"{k}: {v}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y0 += 22

        cv2.imshow("Color Recognition", res.frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('a'), ):  # auto
            recog.set_current_color(None)
        elif key == ord('r'):
            recog.set_current_color("red")
        elif key == ord('g'):
            recog.set_current_color("green")
        elif key == ord('b'):
            recog.set_current_color("blue")
        elif key == ord('y'):
            recog.set_current_color("yellow")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_demo()
