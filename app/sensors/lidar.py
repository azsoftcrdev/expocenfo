# app/sensors/lidar.py

import os
import math
import time
import threading
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2

# TurboJPEG opcional (como en camera.py/streaming.py)
try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
except Exception:
    _jpeg = None

# ---- Intenta cargar backend RPLIDAR ----
_RPLIDAR_OK = True
try:
    from rplidar import RPLidar, RPLidarException
except Exception:
    _RPLIDAR_OK = False
    RPLidar = None
    RPLidarException = Exception

# ─────────────────────────────────────────────────────────────
# Config vía .env (con valores razonables por defecto)
# ─────────────────────────────────────────────────────────────
def _env_float(key: str, default: float) -> float:
    try:
        v = os.getenv(key, "")
        return float(v) if v != "" else float(default)
    except Exception:
        return float(default)

def _env_int(key: str, default: int) -> int:
    try:
        v = os.getenv(key, "")
        return int(v) if v != "" else int(default)
    except Exception:
        return int(default)

def _env_str(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

# Backend y puerto
LIDAR_BACKEND = _env_str("LIDAR_BACKEND", "rplidar").lower()
LIDAR_PORT    = _env_str("LIDAR_PORT", "/dev/ttyUSB0")
LIDAR_BAUD    = _env_int("LIDAR_BAUD", 115200)   # A1/A2=115200; S1=256000; S2=1000000 (ver doc)
LIDAR_TIMEOUT = _env_float("LIDAR_TIMEOUT", 1.0)

# Filtro de distancias (m) y calidad
LIDAR_MIN_DIST = _env_float("LIDAR_MIN_DIST", 0.10)
LIDAR_MAX_DIST = _env_float("LIDAR_MAX_DIST", 8.00)
LIDAR_MIN_QUAL = _env_int("LIDAR_MIN_QUAL", 0)  # umbral de calidad por punto (0..255)

# Transform del sensor (m y grados) → para orientación/offset en el chasis
LIDAR_OFFSET_X = _env_float("LIDAR_OFFSET_X", 0.0)
LIDAR_OFFSET_Y = _env_float("LIDAR_OFFSET_Y", 0.0)
LIDAR_YAW_DEG  = _env_float("LIDAR_YAW_DEG",  0.0)

# Worker
LIDAR_FPS     = _env_float("LIDAR_FPS", 12.0)  # lazo de lectura (no fuerza HW)
LIDAR_RETRY_S = _env_float("LIDAR_RETRY_S", 2.0)

# Mapa de ocupación (canvas)
MAP_SIZE     = _env_int("LIDAR_MAP_SIZE", 640)        # px (cuadrado)
MAP_SCALE    = _env_float("LIDAR_MAP_SCALE", 100.0)   # px por metro (100 px/m → 1 cm/px)
MAP_DECAY    = _env_float("LIDAR_MAP_DECAY", 0.94)    # 0.90..0.99 (memoria)
MAP_POINT_PX = _env_int("LIDAR_MAP_POINT", 2)         # radio del punto en px

# Parámetros de clustering/detección (ajustables por .env)
OBS_MIN_AREA_PX     = _env_int("LIDAR_OBS_MIN_AREA_PX", 12)      # área mínima de cluster (px^2)
OBS_MAX_AREA_PX     = _env_int("LIDAR_OBS_MAX_AREA_PX", 2000)    # área máxima (descarta parches enormes)
PEOPLE_MIN_DIAM_M   = _env_float("LIDAR_PEOPLE_MIN_DIAM_M", 0.10)  # ~10 cm (pierna)
PEOPLE_MAX_DIAM_M   = _env_float("LIDAR_PEOPLE_MAX_DIAM_M", 0.45)  # ~45 cm (torso)
CCL_KERNEL          = max(1, _env_int("LIDAR_CCL_KERNEL", 3))       # tamaño kernel cierre
CCL_CLOSE_ITERS     = max(0, _env_int("LIDAR_CCL_CLOSE_ITERS", 1))  # iteraciones de MORPH_CLOSE
CCL_DILATE_ITERS    = max(0, _env_int("LIDAR_CCL_DILATE_ITERS", 0)) # dilatación opcional

# MJPEG
BOUNDARY_NAME  = "lidar"
BOUNDARY_BYTES = b"--" + BOUNDARY_NAME.encode()

# ─────────────────────────────────────────────────────────────
# Lidar clase principal
# ─────────────────────────────────────────────────────────────
class Lidar:
    """
    Lidar con backend RPLIDAR (por ahora).
    Mantiene el último escaneo (angles[d], dists[m], quals[0..255]) y un mapa 2D.
    Provee renderizados "por modo" (map/raw/obstacles/people) y generadores MJPEG.
    """

    def __init__(self,
                 port: Optional[str] = None,
                 baudrate: Optional[int] = None,
                 backend: Optional[str] = None):
        self.port = port or LIDAR_PORT
        self.baudrate = int(baudrate or LIDAR_BAUD)
        self.backend = (backend or LIDAR_BACKEND).lower()

        # Estado de lectura
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._ts = 0.0
        self._angles = np.zeros(0, dtype=np.float32)
        self._dists  = np.zeros(0, dtype=np.float32)  # metros
        self._quals  = np.zeros(0, dtype=np.uint8)

        # Backend driver
        self._dev = None

        # Mapa (acumulado)
        self._map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        self._map_center = (MAP_SIZE // 2, MAP_SIZE // 2)
        self._cos_yaw = math.cos(math.radians(LIDAR_YAW_DEG))
        self._sin_yaw = math.sin(math.radians(LIDAR_YAW_DEG))

    # -------------------- API pública --------------------

    def open(self) -> None:
        if self.backend != "rplidar":
            raise RuntimeError(f"LIDAR_BACKEND '{self.backend}' no soportado (por ahora)")

        if not _RPLIDAR_OK:
            raise RuntimeError("Paquete 'rplidar' no instalado. Instala con: pip install rplidar")

        # Conectar
        self._dev = RPLidar(self.port, baudrate=self.baudrate, timeout=LIDAR_TIMEOUT)
        # Prueba info/health (opcional)
        try:
            _ = self._dev.get_info()
            _ = self._dev.get_health()
        except Exception:
            pass

    def start(self) -> None:
        if self._running:
            return
        if self._dev is None:
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LidarReader", daemon=True)
        self._thread.start()
        print(f"[lidar] running backend={self.backend} port={self.port} baud={self.baudrate}")

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None

    def close(self) -> None:
        self.stop()
        try:
            if self._dev:
                try:
                    self._dev.stop()
                except Exception:
                    pass
                try:
                    self._dev.stop_motor()
                except Exception:
                    pass
                try:
                    self._dev.disconnect()
                except Exception:
                    pass
        finally:
            self._dev = None

    def get_latest_scan(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Devuelve (angles_deg, dists_m, quals, ts). Copias seguras."""
        with self._lock:
            return self._angles.copy(), self._dists.copy(), self._quals.copy(), float(self._ts)

    # Helpers clásicos para navegación/evitación
    def get_min_distance(self, deg_start: float, deg_end: float, min_quality: int = 0) -> float:
        """Distancia mínima en sector [deg_start, deg_end] (grados absolutos Lidar). Devuelve metros (np.inf si vacío)."""
        a, d, q, _ = self.get_latest_scan()
        if a.size == 0:
            return float("inf")
        qthr = max(min_quality, LIDAR_MIN_QUAL)
        mask = (a >= deg_start) & (a <= deg_end) & (q >= qthr) & (d >= LIDAR_MIN_DIST) & (d <= LIDAR_MAX_DIST)
        if not np.any(mask):
            return float("inf")
        return float(np.min(d[mask]))

    def get_sector_ranges(self, sectors: int = 8, min_quality: int = 0) -> np.ndarray:
        """Distancia mínima por sector (0..360 dividido en N). Array de metros (len=N)."""
        a, d, q, _ = self.get_latest_scan()
        out = np.full(sectors, np.inf, dtype=np.float32)
        if a.size == 0:
            return out
        step = 360.0 / float(sectors)
        qthr = max(min_quality, LIDAR_MIN_QUAL)
        for i in range(sectors):
            lo = i * step
            hi = (i + 1) * step
            m = (a >= lo) & (a < hi) & (q >= qthr) & (d >= LIDAR_MIN_DIST) & (d <= LIDAR_MAX_DIST)
            if np.any(m):
                out[i] = np.min(d[m])
        return out

    def get_cartesian(self) -> np.ndarray:
        """Devuelve puntos Nx2 en metros (x,y) ya rotados/trasladados según OFFSET/YAW."""
        with self._lock:
            a = self._angles.copy()
            d = self._dists.copy()
            q = self._quals.copy()
        if a.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        qthr = int(LIDAR_MIN_QUAL)
        m = (d >= LIDAR_MIN_DIST) & (d <= LIDAR_MAX_DIST) & (q >= qthr)
        if not np.any(m):
            return np.zeros((0, 2), dtype=np.float32)
        a = np.deg2rad(a[m]); r = d[m]
        xs = r * np.cos(a)
        ys = r * np.sin(a)
        xr = xs * self._cos_yaw - ys * self._sin_yaw + LIDAR_OFFSET_X
        yr = xs * self._sin_yaw + ys * self._cos_yaw + LIDAR_OFFSET_Y
        pts = np.stack([xr, yr], axis=1).astype(np.float32)
        return pts

    # -------------------- Generadores MJPEG --------------------

    def mjpeg_map_generator(self, fps: float = 8.0, size: Optional[str] = None):
        """
        Generador MJPEG del mapa (compatibilidad con versiones previas).
        """
        self.start()

        out_size = None
        if size:
            try:
                w, h = size.lower().split("x")
                out_size = (int(w), int(h))
            except Exception:
                out_size = None

        period = 1.0 / max(1.0, float(fps))
        next_t = time.time()

        while True:
            try:
                frame = self._render_map()
                if out_size:
                    try:
                        frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass

                if _jpeg is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        jpg = _jpeg.encode(rgb, quality=80)
                    except Exception:
                        jpg = None
                else:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpg = buf.tobytes() if ok else None

                if jpg is None:
                    continue

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

            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.05)

    def mjpeg_generator(self, mode: str = "map", fps: float = 8.0, size_str: Optional[str] = None, quality: int = 80):
        """
        Generador MJPEG genérico con 'mode': 'map' | 'raw' | 'obstacles' | 'people'.
        """
        self.start()

        out_size = None
        if size_str:
            try:
                w, h = size_str.lower().split("x")
                out_size = (int(w), int(h))
            except Exception:
                out_size = None

        period = 1.0 / max(1.0, float(fps))
        next_t = time.time()

        while True:
            try:
                frame = self.render_frame(mode, out_size)

                if _jpeg is not None:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        jpg = _jpeg.encode(rgb, quality=int(quality))
                    except Exception:
                        jpg = None
                else:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                    jpg = buf.tobytes() if ok else None

                if jpg is None:
                    continue

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

            except GeneratorExit:
                break
            except Exception:
                time.sleep(0.05)

    def snapshot_jpeg(self, mode: str = "map", size_str: Optional[str] = None, quality: int = 80) -> Optional[bytes]:
        """
        Devuelve un solo JPEG del frame bajo 'mode'. size_str '640x480' (opcional).
        """
        out_size = None
        if size_str:
            try:
                w, h = size_str.lower().split("x")
                out_size = (int(w), int(h))
            except Exception:
                out_size = None

        frame = self.render_frame(mode, out_size)
        if _jpeg is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return _jpeg.encode(rgb, quality=int(quality))
            except Exception:
                pass
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        return buf.tobytes() if ok else None

    # -------------------- Renderizados por modo --------------------

    def render_frame(self, mode: str = "map", size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Devuelve un frame BGR uint8 según 'mode': 'map' | 'raw' | 'obstacles' | 'people'.
        Si 'size' se da, reescala la salida.
        """
        mode = (mode or "map").lower()
        if mode == "raw":
            frame = self._render_raw()
        elif mode == "obstacles":
            frame = self._render_obstacles()
        elif mode == "people":
            frame = self._render_people()
        else:
            frame = self._render_map()

        if size:
            try:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            except Exception:
                pass
        return frame

    def _render_raw(self) -> np.ndarray:
        """Puntos del último scan sobre fondo negro con rejilla."""
        H = W = MAP_SIZE
        scale = MAP_SCALE
        cx, cy = self._map_center

        img = np.zeros((H, W, 3), dtype=np.uint8)

        # rejilla (cada 1 m)
        step = int(scale)
        if step >= 20:
            for x in range(0, W, step):
                cv2.line(img, (x, 0), (x, H-1), (40, 40, 40), 1)
            for y in range(0, H, step):
                cv2.line(img, (0, y), (W-1, y), (40, 40, 40), 1)

        # ejes
        cv2.line(img, (cx, 0), (cx, H-1), (80, 80, 80), 1)
        cv2.line(img, (0, cy), (W-1, cy), (80, 80, 80), 1)

        pts = self.get_cartesian()  # Nx2 (m)
        if pts.shape[0]:
            x_px = (cx + pts[:, 0] * scale).astype(np.int32)
            y_px = (cy - pts[:, 1] * scale).astype(np.int32)
            valid = (x_px >= 0) & (x_px < W) & (y_px >= 0) & (y_px < H)
            xs, ys = x_px[valid], y_px[valid]
            for (xx, yy) in zip(xs, ys):
                cv2.circle(img, (int(xx), int(yy)), max(1, MAP_POINT_PX), (0, 255, 0), -1)

        # HUD
        with self._lock:
            npts = int(self._angles.size)
            ts = self._ts
        age = time.time() - ts if ts > 0 else 0.0
        hud = f"RAW | pts={npts} | age={age:.1f}s"
        cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        return img

    def _render_obstacles(self) -> np.ndarray:
        """
        Clustering simple por conectividad en grilla (OpenCV CCL) sobre el scan actual.
        Dibuja puntos y círculos en cada cluster válido.
        """
        H = W = MAP_SIZE
        scale = MAP_SCALE
        cx, cy = self._map_center

        # lienzo binario del scan
        bin_img = np.zeros((H, W), dtype=np.uint8)
        pts = self.get_cartesian()
        if pts.shape[0]:
            x_px = (cx + pts[:, 0] * scale).astype(np.int32)
            y_px = (cy - pts[:, 1] * scale).astype(np.int32)
            valid = (x_px >= 0) & (x_px < W) & (y_px >= 0) & (y_px < H)
            xs, ys = x_px[valid], y_px[valid]
            for (xx, yy) in zip(xs, ys):
                cv2.circle(bin_img, (int(xx), int(yy)), max(1, MAP_POINT_PX), 255, -1)

        # --- cierre morfológico para unir fragmentos cercanos ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CCL_KERNEL, CCL_KERNEL))
        bin_proc = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=int(CCL_CLOSE_ITERS))
        if CCL_DILATE_ITERS > 0:
            bin_proc = cv2.dilate(bin_proc, kernel, iterations=int(CCL_DILATE_ITERS))

        # componentes conectados
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_proc, connectivity=8)
        out = cv2.cvtColor(bin_proc, cv2.COLOR_GRAY2BGR)

        count = 0
        for i in range(1, num):  # 0 = fondo
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < OBS_MIN_AREA_PX or area > OBS_MAX_AREA_PX:
                continue
            cx_i, cy_i = centroids[i]
            # diámetro equivalente (px → m)
            eq_diam_px = 2.0 * (area / np.pi) ** 0.5
            eq_diam_m = float(eq_diam_px) / float(scale)
            # círculo que representa el cluster
            cv2.circle(out, (int(cx_i), int(cy_i)), int(eq_diam_px / 2), (0, 215, 255), 2)
            cv2.circle(out, (int(cx_i), int(cy_i)), 2, (0, 0, 255), -1)
            count += 1

        hud = f"OBSTACLES | clusters={count}"
        cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def _render_people(self) -> np.ndarray:
        """
        Heurística simple de 'personas' basada en el diámetro equivalente del cluster
        (piernas/torso). Filtro por rango de diámetro en metros.
        """
        H = W = MAP_SIZE
        scale = MAP_SCALE
        cx, cy = self._map_center

        bin_img = np.zeros((H, W), dtype=np.uint8)
        pts = self.get_cartesian()
        if pts.shape[0]:
            x_px = (cx + pts[:, 0] * scale).astype(np.int32)
            y_px = (cy - pts[:, 1] * scale).astype(np.int32)
            valid = (x_px >= 0) & (x_px < W) & (y_px >= 0) & (y_px < H)
            xs, ys = x_px[valid], y_px[valid]
            for (xx, yy) in zip(xs, ys):
                cv2.circle(bin_img, (int(xx), int(yy)), max(1, MAP_POINT_PX), 255, -1)

        # cierre morfológico (mismo pipeline que obstacles)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CCL_KERNEL, CCL_KERNEL))
        bin_proc = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=int(CCL_CLOSE_ITERS))
        if CCL_DILATE_ITERS > 0:
            bin_proc = cv2.dilate(bin_proc, kernel, iterations=int(CCL_DILATE_ITERS))

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_proc, connectivity=8)
        out = cv2.cvtColor(bin_proc, cv2.COLOR_GRAY2BGR)

        count = 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < OBS_MIN_AREA_PX or area > OBS_MAX_AREA_PX:
                continue
            eq_diam_px = 2.0 * (area / np.pi) ** 0.5
            eq_diam_m = float(eq_diam_px) / float(scale)
            if PEOPLE_MIN_DIAM_M <= eq_diam_m <= PEOPLE_MAX_DIAM_M:
                cx_i, cy_i = centroids[i]
                cv2.circle(out, (int(cx_i), int(cy_i)), int(eq_diam_px / 2), (0, 140, 255), 2)
                cv2.putText(out, "person", (int(cx_i) + 6, int(cy_i) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1, cv2.LINE_AA)
                count += 1

        hud = f"PEOPLE | candidates={count} | d in [{PEOPLE_MIN_DIAM_M:.2f},{PEOPLE_MAX_DIAM_M:.2f}] m"
        cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    # -------------------- Internos --------------------

    def _loop(self) -> None:
        """Hilo lector (reconecta si hay error)."""
        period = 1.0 / max(1e-6, float(LIDAR_FPS))
        while self._running:
            try:
                if self._dev is None:
                    self.open()

                # iter_scans devuelve lista de (quality, angle_deg, distance_mm)
                for scan in self._dev.iter_scans(min_len=5):
                    if not self._running:
                        break

                    angs, dists, quals = [], [], []
                    for q, ang, dist in scan:
                        d_m = float(dist) / 1000.0  # metros
                        angs.append(float(ang))
                        dists.append(d_m)
                        quals.append(int(q))

                    with self._lock:
                        self._angles = np.asarray(angs, dtype=np.float32)
                        self._dists  = np.asarray(dists, dtype=np.float32)
                        self._quals  = np.asarray(quals, dtype=np.uint8)
                        self._ts = time.time()

                    # actualizar mapa con este escaneo
                    try:
                        self._integrate_into_map()
                    except Exception:
                        pass

                    time.sleep(period)

            except Exception as e:
                print("[lidar] loop error:", e)
                try:
                    if self._dev:
                        try:
                            self._dev.stop()
                        except Exception:
                            pass
                        try:
                            self._dev.stop_motor()
                        except Exception:
                            pass
                        try:
                            self._dev.disconnect()
                        except Exception:
                            pass
                finally:
                    self._dev = None
                time.sleep(LIDAR_RETRY_S)

    def _integrate_into_map(self) -> None:
        """Acumula puntos actuales en el mapa con 'decay'."""
        H = W = MAP_SIZE
        cx, cy = self._map_center
        scale = MAP_SCALE  # px/m

        pts = self.get_cartesian()  # Nx2 en metros, ya con yaw+offset
        # Decay siempre
        self._map *= MAP_DECAY

        if pts.shape[0] == 0:
            # nada más que hacer si no hay puntos
            cv2.circle(self._map, (cx, cy), 3, (1.0,), -1)  # sensor
            return

        x_px = (cx + (pts[:, 0] * scale)).astype(np.int32)
        y_px = (cy - (pts[:, 1] * scale)).astype(np.int32)
        valid = (x_px >= 0) & (x_px < W) & (y_px >= 0) & (y_px < H)
        xs = x_px[valid]
        ys = y_px[valid]

        if xs.size:
            for (xx, yy) in zip(xs, ys):
                cv2.circle(self._map, (int(xx), int(yy)), MAP_POINT_PX, (1.0,), -1)

        # posición del sensor
        cv2.circle(self._map, (cx, cy), 3, (1.0,), -1)

    def _render_map(self) -> np.ndarray:
        """Convierte el mapa float32 [0..1] a BGR y agrega overlays."""
        with self._lock:
            m = self._map.copy()
            npts = int(self._angles.size)
            ts = self._ts

        m = np.clip(m, 0.0, 1.0)
        img = (m * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

        # Rejilla y ejes (cada 1 m si MAP_SCALE=100 → 100 px)
        step = int(MAP_SCALE)
        if step >= 20:
            for x in range(0, MAP_SIZE, step):
                cv2.line(color, (x, 0), (x, MAP_SIZE - 1), (40, 40, 40), 1)
            for y in range(0, MAP_SIZE, step):
                cv2.line(color, (0, y), (MAP_SIZE - 1, y), (40, 40, 40), 1)

        cx, cy = self._map_center
        cv2.line(color, (cx, 0), (cx, MAP_SIZE - 1), (80, 80, 80), 1)
        cv2.line(color, (0, cy), (MAP_SIZE - 1, cy), (80, 80, 80), 1)

        # HUD
        age = time.time() - ts if ts > 0 else 0.0
        hud = f"LIDAR {self.backend} | pts={npts} | port={self.port} | baud={self.baudrate} | age={age:.1f}s"
        cv2.putText(color, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(color, hud, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        scale_txt = f"{int(MAP_SCALE)} px/m"
        cv2.putText(color, scale_txt, (10, MAP_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 230, 10), 1, cv2.LINE_AA)
        return color


# ─────────────────────────────────────────────────────────────
# Singleton práctico
# ─────────────────────────────────────────────────────────────
_lidar_singleton: Optional[Lidar] = None

def lidar() -> Lidar:
    global _lidar_singleton
    if _lidar_singleton is None:
        _lidar_singleton = Lidar()
        _lidar_singleton.start()
    return _lidar_singleton

