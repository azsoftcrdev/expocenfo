# app/IA/person_detection.py
# YOLO (Ultralytics) asíncrono para "person" con filtros anti-FP y gating de movimiento
# Python 3.9 compatible · Auto-fallback a CPU · Interfaz estable para streaming.py

import os, time, threading
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:
    torch = None

# Ruta por defecto si no hay variables de entorno
DEFAULT_MODEL = "/home/jetson/Desktop/hexamind-main/robot-server/app/IA/yolov5/yolov5s.pt"

Box = Tuple[int, int, int, int]   # x,y,w,h
Det = Tuple[Box, int, float]      # ((x,y,w,h), cls, score)

_worker = None  # type: Optional["_YOLOWorker"]
_last_names: Optional[List[str]] = None

# ---------------- utils/env ----------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except Exception:
        return default

def _resolve_device(req) -> Union[str, int]:
    has_cuda = False
    if torch is not None:
        try:
            has_cuda = bool(torch.cuda.is_available())
        except Exception:
            has_cuda = False
    if not has_cuda:
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            print('[person_detection] CUDA no disponible; usando "cpu".')
        return "cpu"
    if req in (None, "", "auto"):
        return 0
    return req

def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    ix1, iy1 = max(ax, bx), max(ay, by); ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter / max(1.0, union)

def _laplacian_mean(gray_roi: np.ndarray) -> float:
    if gray_roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(gray_roi, cv2.CV_32F)
    return float(np.mean(np.abs(lap)))

def _motion_fraction(mask: Optional[np.ndarray], box: Box) -> float:
    if mask is None: return 0.0
    x, y, w, h = box
    H, W = mask.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    roi = mask[y:y+h, x:x+w]
    if roi.size == 0: return 0.0
    return float(cv2.countNonZero(roi)) / float(w*h)

# ---------------- worker ----------------
class _YOLOWorker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = _env_float("PD_CONF", 0.25),  # umbral que pasa a YOLO
        imgsz: int = 640,
        device: Optional[object] = "auto",
        max_fps: float = 12.0
    ) -> None:
        # Acepta HEXAMIND_YOLO_MODEL o YOLO_MODEL_PATH
        self.model_path = (
            model_path
            or os.getenv("HEXAMIND_YOLO_MODEL")
            or os.getenv("YOLO_MODEL_PATH")
            or DEFAULT_MODEL
        )
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Modelo YOLO no encontrado: {self.model_path}")

        # Inferencia y loop
        self.conf_infer = float(conf)
        self.imgsz = _env_int("PD_IMGSZ", int(imgsz))  # permite bajar a 416/480 para CPU
        self.device_resolved = _resolve_device(device)
        self.period = 1.0 / max(1e-6, float(max_fps))

        # Filtros anti-FP
        self.show_conf = _env_float("PD_SHOW_CONF", 0.45)     # conf mínima para MOSTRAR
        self.min_area_ratio = _env_float("PD_MIN_AREA", 0.03) # % del área del frame
        self.ar_min = _env_float("PD_AR_MIN", 0.25)           # min W/H
        self.ar_max = _env_float("PD_AR_MAX", 2.50)           # max W/H (acostadas)
        self.min_short = _env_float("PD_MIN_SHORT", 0.06)     # % del lado corto del frame (anti-cajas “filo”)
        self.nms_iou = _env_float("PD_NMS_IOU", 0.55)         # NMS extra
        self.persist_hits = _env_int("PD_PERSIST", 3)         # frames para estabilizar
        self.max_miss = 6                                     # frames sin ver para borrar pista
        self.texture_min = _env_float("PD_TEXTURE_MIN", 6.0)  # textura mínima (Laplaciano) para nuevas pistas

        # Movimiento (gating) + aceptación estática
        self.motion_thr = _env_int("PD_MOTION_THR", 18)             # 8..25 típico (más alto = menos sensible)
        self.motion_frac_min = _env_float("PD_MOTION_FRAC", 0.02)   # 2% píxeles moviéndose
        self.static_conf = _env_float("PD_STATIC_CONF", 0.60)       # conf para aceptar estático
        self.static_min_area = _env_float("PD_STATIC_MIN_AREA", 0.06)

        # Estado
        self._lock = threading.Lock()
        self._new_frame: Optional[np.ndarray] = None
        self._last_dets: List[Det] = []
        self._last_names: Optional[List[str]] = None
        self._last_fps: float = 0.0
        self._stop = False

        # Tracking simple
        self._tracks = {}  # id -> dict(box, score, hits, miss, need)
        self._next_id = 1

        # Memoria para movimiento
        self._prev_gray: Optional[np.ndarray] = None

        print(f"[person_detection] cargando modelo: {self.model_path} | device={self.device_resolved} | "
              f"conf_infer={self.conf_infer} | imgsz={self.imgsz}")
        self.model = YOLO(self.model_path)

        self._t = threading.Thread(target=self._run, name="PersonYOLOWorker", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop = True
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass

    def submit(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return
        with self._lock:
            self._new_frame = frame_bgr

    def get_last(self) -> Tuple[List[Det], Optional[List[str]], float, str]:
        with self._lock:
            dets = list(self._last_dets)
            names = list(self._last_names) if self._last_names else None
            fps = float(self._last_fps)
        return dets, names, fps, str(self.device_resolved)

    # --------- core loop ---------
    def _run(self) -> None:
        next_ts, ema = 0.0, None
        while not self._stop:
            now = time.time()
            if now < next_ts:
                time.sleep(min(0.005, next_ts - now)); continue
            next_ts = now + self.period

            frame = None
            with self._lock:
                if self._new_frame is not None:
                    frame = self._new_frame; self._new_frame = None
            if frame is None: continue

            H, W = frame.shape[:2]

            # ---- máscara de movimiento (barata en CPU) ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mask = None
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                blur = cv2.GaussianBlur(diff, (5,5), 0)
                _, motion_mask = cv2.threshold(blur, self.motion_thr, 255, cv2.THRESH_BINARY)
                motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
                motion_mask = cv2.dilate(motion_mask, np.ones((3,3), np.uint8), iterations=1)
            self._prev_gray = gray

            # ---- inferencia YOLO (todas las clases; filtramos luego) ----
            try:
                t0 = time.time()
                results = self.model.predict(
                    source=frame, imgsz=self.imgsz, conf=self.conf_infer,
                    device=self.device_resolved, verbose=False
                )
                dt = max(1e-6, time.time() - t0)
                fps = 1.0 / dt; ema = fps if ema is None else (0.9*ema + 0.1*fps)

                result = results[0]
                raw = _parse_dets(result)  # [(box, cls, score)]
                refined = self._refine_and_stabilize(raw, (H, W), motion_mask, gray)

                names = None
                if hasattr(result, "names"):
                    if isinstance(result.names, dict): names = [result.names[k] for k in sorted(result.names.keys())]
                    elif isinstance(result.names, list): names = result.names

                with self._lock:
                    self._last_dets = refined
                    self._last_names = names
                    self._last_fps = float(ema or fps)
            except Exception as e:
                with self._lock: self._last_dets = []
                print(f"[person_detection] error de inferencia: {e}")

    # --------- postproc + filtros ---------
    def _refine_and_stabilize(self, dets: List[Det], hw: Tuple[int,int],
                               motion_mask: Optional[np.ndarray], gray: np.ndarray) -> List[Det]:
        H, W = hw
        frame_area = float(H*W)

        # 1) Filtros duros + CLASE person
        tmp: List[Tuple[Det, float, float, float]] = []  # (det, motion_frac, area_frac, texture)
        for (x, y, w, h), cls_id, score in dets:
            if cls_id != 0:      # solo 'person'
                continue
            if score < self.show_conf:
                continue
            area = w*h
            area_frac = area / max(1.0, frame_area)
            if area_frac < self.min_area_ratio:
                continue
            ar = w / max(1.0, float(h))
            if not (self.ar_min <= ar <= self.ar_max):
                continue
            # lado mínimo relativo del cuadro (anti-cajas muy delgadas)
            short_frac = min(w / float(W), h / float(H))
            if short_frac < self.min_short:
                continue

            # clip
            x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
            w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
            box = (x, y, w, h)

            # movimiento + textura
            mfrac = _motion_fraction(motion_mask, box)
            g_roi = gray[y:y+h, x:x+w]
            tex = _laplacian_mean(g_roi)

            tmp.append(((box, cls_id, score), mfrac, area_frac, tex))

        if not tmp:
            self._decay_tracks(); 
            return []

        # 2) NMS extra
        tmp.sort(key=lambda t: t[0][2], reverse=True)
        kept: List[Tuple[Det,float,float,float]] = []
        sup = [False]*len(tmp)
        for i in range(len(tmp)):
            if sup[i]: continue
            kept.append(tmp[i])
            for j in range(i+1, len(tmp)):
                if sup[j]: continue
                if _iou(tmp[i][0][0], tmp[j][0][0]) > self.nms_iou:
                    sup[j] = True

        # 3) Persistencia con gating de movimiento/estática + textura
        out = self._update_tracks(kept)
        return out

    def _decay_tracks(self) -> None:
        to_del = []
        for tid, t in self._tracks.items():
            t["miss"] += 1
            if t["miss"] > self.max_miss:
                to_del.append(tid)
        for tid in to_del:
            self._tracks.pop(tid, None)

    def _update_tracks(self, dets_with_meta: List[Tuple[Det, float, float, float]]) -> List[Det]:
        # dets_with_meta: [((box, cls, score), mfrac, area_frac, tex), ...]
        used = set()
        for t in self._tracks.values():
            t["miss"] += 1

        for ((box, cls_id, score), mfrac, area_frac, tex) in dets_with_meta:
            # Criterios de creación de pista
            allow_motion = (mfrac >= self.motion_frac_min)
            allow_static = (score >= self.static_conf and area_frac >= self.static_min_area and tex >= self.texture_min)

            # Intento de emparejar con pistas existentes (IoU)
            best_tid, best_iou = None, 0.0
            for tid, t in self._tracks.items():
                if tid in used: continue
                iouv = _iou(box, t["box"])
                if iouv > best_iou:
                    best_iou, best_tid = iouv, tid

            if best_tid is not None and best_iou >= 0.3:
                t = self._tracks[best_tid]
                t["box"], t["score"], t["hits"], t["miss"] = box, score, t["hits"]+1, 0
                used.add(best_tid)
            else:
                if allow_motion or allow_static:
                    tid = self._next_id; self._next_id += 1
                    need = self.persist_hits + (1 if (allow_static and not allow_motion) else 0)
                    self._tracks[tid] = {"box": box, "score": score, "hits": 1, "miss": 0, "need": need}

        # Purga por miss
        to_del = [tid for tid, t in self._tracks.items() if t["miss"] > self.max_miss]
        for tid in to_del:
            self._tracks.pop(tid, None)

        # Emitir solo pistas estables
        out: List[Det] = []
        for t in self._tracks.values():
            need = t.get("need", self.persist_hits)
            if t["hits"] >= need:
                out.append((t["box"], 0, float(t["score"])))
        return out

# ========= Helpers (usados por streaming.py) =========
def start_person_detection_worker(
    fps: float = 8.0,
    model_path: Optional[str] = None,
    conf: float = None,
    imgsz: int = 640,
    device: Optional[object] = "auto"
) -> bool:
    global _worker, _last_names
    if _worker is not None:
        return True
    try:
        _worker = _YOLOWorker(
            model_path=model_path,
            conf=(float(conf) if conf is not None else _env_float("PD_CONF", 0.25)),
            imgsz=imgsz,
            device=device,
            max_fps=fps
        )
        dets, names, _, _ = _worker.get_last()
        _last_names = names
        print(f"[person_detection] worker iniciado (fps_infer={fps}, imgsz={_worker.imgsz})")
        return True
    except Exception as e:
        print("[person_detection] no se pudo iniciar el worker:", e)
        _worker = None; _last_names = None
        return False

def get_last_person_boxes() -> List[Det]:
    global _worker, _last_names
    if _worker is None:
        return []
    dets, names, _, _ = _worker.get_last()
    if names is not None:
        _last_names = names
    return dets

def overlay_person_boxes(frame_bgr: np.ndarray, boxes: List[Det]) -> np.ndarray:
    global _worker, _last_names
    if _worker is not None:
        try:
            _worker.submit(frame_bgr)
            dets, _, ia_fps, dev = _worker.get_last()
            dbg = "YOLO person | dets={} | IA_FPS={:.1f} | dev={}".format(len(dets), ia_fps, dev)
            cv2.putText(frame_bgr, dbg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, dbg, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1, cv2.LINE_AA)
        except Exception:
            pass
    # Dibuja las cajas ya estabilizadas
    for (x, y, w, h), c, s in boxes:
        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
        label = "person:{:.2f}".format(s)
        y_txt = y - 6 if y - 6 > 6 else y + 14
        cv2.putText(frame_bgr, label, (x, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return frame_bgr

def _parse_dets(result) -> List[Det]:
    dets: List[Det] = []
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy; conf = result.boxes.conf; cls_ = result.boxes.cls
    if hasattr(xyxy, "cpu"): xyxy = xyxy.cpu().numpy()
    if hasattr(conf, "cpu"):  conf = conf.cpu().numpy()
    if hasattr(cls_, "cpu"):  cls_ = cls_.cpu().numpy()
    for i in range(xyxy.shape[0]):
        x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
        dets.append(((x1, y1, int(x2-x1), int(y2-y1)), int(cls_[i]), float(conf[i])))
    return dets

