# 🤖 Robot Server (Backend)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-green)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Servidor **backend** en **FastAPI** para controlar y monitorear un robot hexápodo (**HexaMind**).
Optimizado para ejecutarse tanto en **PCs de desarrollo** como en **Jetson** (edge computing).

Incluye:

* Control de movimiento del robot.
* Streaming de cámara en tiempo real (MJPEG).
* IA modular:

  * Detección de personas (YOLO, CPU fallback).
  * Reconocimiento facial (MediaPipe) asíncrono.
  * Reconocimiento de color (HSV).
* Sensores:

  * Cámara UVC / Astra.
  * **LiDAR RPLIDAR** con mapa MJPEG y utilidades de navegación.
* Bus interno de eventos + WebSocket robusto.

---

## 📑 Índice

1. [Arquitectura](#arquitectura)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Uso rápido](#uso-rápido)
5. [Configuración (.env)](#configuración-env)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Endpoints](#endpoints)

   * [7.1. Básicos](#71-básicos)
   * [7.2. Streaming de Video](#72-streaming-de-video)
   * [7.3. Movimiento](#73-movimiento)
   * [7.4. LIDAR Map](#74-lidar-map)
8. [WebSocket API](#websocket-api)
9. [Para Devs](#para-devs)

   * [9.1. Entorno de desarrollo](#91-entorno-de-desarrollo)
   * [9.2. Estilo, lint y tests](#92-estilo-lint-y-tests)
   * [9.3. Workers asíncronos (patrón latest-wins)](#93-workers-asíncronos-patrón-latest-wins)
   * [9.4. Logs y depuración](#94-logs-y-depuración)
   * [9.5. Despliegue Jetson (systemd)](#95-despliegue-jetson-systemd)
10. [Solución de problemas](#solución-de-problemas)
11. [Roadmap](#roadmap)
12. [Licencia](#licencia)

---

## 🏗️ Arquitectura

* **Lenguaje:** Python 3.9+
* **Framework:** FastAPI (ASGI)
* **IA:** OpenCV + MediaPipe + Ultralytics YOLO (CPU fallback)
* **Streaming:** MJPEG (multipart/x-mixed-replace)
* **Sensores:** cámara y LiDAR con hilos dedicados
* **Patrón:** Modular (IA, sensores, movimiento, core/web)
* **Concurrencia:**

  * Procesamiento IA/LiDAR en **hilos** con **coalescing** (latest-wins).
  * WebSocket con **manejo de CancelledError** para cierres limpios.

---

## 📋 Requisitos

* Python **3.9+**
* `pip` actualizado
* Cámara compatible con OpenCV (UVC) o Astra
* LiDAR **RPLIDAR** (A1/A2/S1/S2) — opcional
* Jetson (opcional, recomendado para despliegue)

---

## ⚙️ Instalación

```bash
# 1) Entrar al proyecto
cd ~/Desktop/hexamind-main/robot-server

# 2) Crear y activar venv
python3 -m venv env
source env/bin/activate

# 3) Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4) Sensores extra
pip install rplidar       # LiDAR backend (RPLIDAR)
pip install PyTurboJPEG   # opcional: acelera MJPEG
```

> Si usas Jetson con versiones específicas de torch/mediapipe, ajusta `requirements.txt` según tu plataforma.

---

## ▶️ Uso rápido

```bash
source env/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* App: `http://<IP>:8000`
* Docs Swagger: `http://<IP>:8000/docs`

Logs esperados al iniciar:

* `[INFO] Cámara abierta`
* `[person_detection] cargando modelo: ... | device=cpu | conf_infer=... | imgsz=...`
* `[lidar] running backend=rplidar port=/dev/ttyUSB* baud=...`

---

## 🔧 Configuración (.env)

Crea/edita `config/.env` (sin `export`). Ejemplo completo:

```ini
# ───── Servidor ─────
HOST=0.0.0.0
PORT=8000
WS_RATE_HZ=10

# ───── Cámara ─────
CAMERA_DEVICE=/dev/video0
CAMERA_WIDTH=640
CAMERA_HEIGHT=360
CAMERA_FPS=15
# o backend Astra:
# CAMERA_BACKEND=astra

# ───── YOLO (person_detection) ─────
HEXAMIND_YOLO_MODEL=/home/jetson/Desktop/hexamind-main/robot-server/app/IA/yolov5/yolov5s.pt
PD_IMGSZ=416
PD_CONF=0.25
PD_SHOW_CONF=0.45
PD_MIN_AREA=0.03
PD_AR_MIN=0.25
PD_AR_MAX=2.5
PD_MIN_SHORT=0.08
PD_PERSIST=3
PD_NMS_IOU=0.55
PD_TEXTURE_MIN=6.0
PD_MOTION_FRAC=0.02
PD_STATIC_CONF=0.60
PD_STATIC_MIN_AREA=0.06

# ───── Face (MediaPipe) ─────
FACE_CONF=0.6
FACE_MAX_SIDE=640
FACE_FPS=12
FACE_COLOR=0,255,0
FACE_BLUR=0

# ───── LiDAR (RPLIDAR) ─────
LIDAR_BACKEND=rplidar
LIDAR_PORT=/dev/ttyUSB0   # AJUSTA
LIDAR_BAUD=115200         # A1/A2=115200; S1=256000; S2≈1000000
LIDAR_TIMEOUT=1.0
LIDAR_RETRY_S=2.0
LIDAR_MIN_DIST=0.10
LIDAR_MAX_DIST=8.0
LIDAR_FPS=12
LIDAR_OFFSET_X=0.0
LIDAR_OFFSET_Y=0.0
LIDAR_YAW_DEG=0.0
LIDAR_MAP_SIZE=640
LIDAR_MAP_SCALE=100.0
LIDAR_MAP_DECAY=0.94
LIDAR_MAP_POINT=2
```

Asegúrate de cargarlo al inicio de la app:

```python
from dotenv import load_dotenv
load_dotenv("config/.env")
```

---

## 📂 Estructura del Proyecto

```bash
robot-server/
├── app/
│   ├── main.py                  # Punto de entrada FastAPI
│   ├── streaming.py             # Generador MJPEG (none|color|face|person)
│   ├── web/
│   │   └── ws.py               # WebSocket robusto (CancelledError-safe)
│   ├── core/
│   │   ├── bus.py              # Bus interno pub/sub
│   │   └── settings.py         # Carga de .env y constantes
│   ├── IA/
│   │   ├── person_detection.py # YOLO asíncrono con filtros anti-FP
│   │   └── face_recognition.py # MediaPipe asíncrono (worker)
│   ├── motion/
│   │   └── controller_vel.py   # Control de velocidad (deadman, etc.)
│   └── sensors/
│       ├── camera.py           # Cámara UVC/Astra
│       └── lidar.py            # LiDAR RPLIDAR + mapa MJPEG
├── config/
│   └── .env                    # Variables de entorno
├── requirements.txt
└── README.md
```

---

## 🔌 Endpoints

### 7.1. Básicos

| Método | Ruta            | Descripción              |
| ------ | --------------- | ------------------------ |
| GET    | `/health`       | Estado del servidor      |
| GET    | `/snapshot.jpg` | Captura de imagen actual |

### 7.2. Streaming de Video

Un único endpoint (según tu router) con modos:

```
GET /stream/video?mode=<none|color|face|person>&fps=15&size=640x360&quality=70&overlay=1
```

* `mode`: `none`, `color`, `face`, `person`
* `fps`: 1–60 (MJPEG)
* `size`: `WxH` (opcional)
* `quality`: 40–95
* `overlay`: 0/1 (dibujar cajas/HUD)
* `color`: para `mode=color` (ej. `red`, `green`)

### 7.3. Movimiento

| Método | Ruta      | Descripción                         |
| ------ | --------- | ----------------------------------- |
| POST   | `/vel`    | Control de velocidad lineal/angular |
| POST   | `/stop`   | Detener movimiento                  |
| GET    | `/status` | Estado del controlador              |
| POST   | `/move`   | Movimiento por `duration_ms`        |

**Ejemplo `/move`:**

```bash
curl -X POST http://localhost:8000/move \
  -H "Content-Type: application/json" \
  -d '{"linear": 0.3, "angular": 0.2, "duration_ms": 800}'
```

### 7.4. LIDAR Map

```
GET /stream/lidar_map?fps=8&size=640x640
```

Devuelve MJPEG del mapa (ocupación simple con *decay*).

---

## 🌐 WebSocket API

```
ws://<host>:<port>/ws/
```

**Suscripciones (servidor → cliente):** `telemetry`, `alert`, `mode`, `ui_event`.

**Ejemplo salida:**

```json
{"topic":"telemetry","data":{"battery":92,"temp":41.2}}
```

**Comandos del cliente:**

1. **Movimiento (latest-wins)**

```json
{"type":"motion_setpoint","x":0,"y":1,"z":0,"speed":40}
```

2. **Publicar en bus**

```json
{"topic":"ui_event","data":{"button":"start","pressed":true}}
```

> El servidor maneja cierres con `CancelledError` sin tracebacks y cancela tareas internas de forma limpia.

---

## 🧑‍💻 Para Devs

### 9.1. Entorno de desarrollo

```bash
# activar venv
source env/bin/activate

# ejecutar con hot-reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# variables (usa config/.env)
python - << 'PY'
import os
keys = ['HEXAMIND_YOLO_MODEL','PD_IMGSZ','FACE_FPS','LIDAR_PORT','LIDAR_BAUD','WS_RATE_HZ']
print({k: os.getenv(k) for k in keys})
PY
```

**Probar LiDAR rápidamente** (detecta puerto/baud): consulta `probe_lidar.py` en la guía.

### 9.2. Estilo, lint y tests

Sugeridos (si no están en `requirements-dev.txt`):

```bash
pip install black isort flake8 pytest pre-commit
pre-commit install
```

* **Formato:** `black .`  | **Imports:** `isort .`
* **Lint:** `flake8 app`
* **Tests:** `pytest -q`

Convenciones:

* PEP8 + type hints.
* Módulos **sin efectos colaterales** al importar (workers arrancan on-demand).
* Evita bloqueos en el hilo principal: usa hilos/`run_in_executor`.

### 9.3. Workers asíncronos (patrón latest-wins)

Ejemplo mínimo de worker (hilo) que procesa el último frame disponible:

```python
class Worker:
    def __init__(self, fps=12):
        self._lock=threading.Lock(); self._new=None; self._last=None
        self._stop=False; self._period=1.0/max(1e-6,fps)
        self._t=None
    def start(self):
        if self._t: return
        self._t=threading.Thread(target=self._loop,daemon=True); self._t.start()
    def submit(self, x):
        with self._lock: self._new=x
    def get(self):
        with self._lock: return self._last
    def _loop(self):
        next_t=0.0
        while not self._stop:
            now=time.time();
            if now<next_t: time.sleep(min(0.005,next_t-now)); continue
            next_t=now+self._period
            with self._lock:
                x=self._new; self._new=None
            if x is None: continue
            y = self._process(x)  # <- pesado
            with self._lock: self._last=y
```

### 9.4. Logs y depuración

* Nivel Uvicorn: `--log-level info|debug`.
* **WebSocket**: patrón try/finally cancelando tareas y suprimiendo `CancelledError`.
* **YOLO**: imprime al iniciar ruta del modelo y device. Si no hay CUDA → CPU automatic.
* **LIDAR**: en errores, reconecta tras `LIDAR_RETRY_S`.

### 9.5. Despliegue Jetson (systemd)

Archivo `/etc/systemd/system/hexamind.service` (ejemplo):

```ini
[Unit]
Description=HexaMind Robot Server
After=network.target

[Service]
User=jetson
WorkingDirectory=/home/jetson/Desktop/hexamind-main/robot-server
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/home/jetson/Desktop/hexamind-main/robot-server/config/.env
ExecStart=/home/jetson/Desktop/hexamind-main/robot-server/env/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Activar:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hexamind.service
journalctl -u hexamind -f
```

---

## 🛠️ Solución de problemas

**YOLO: “Modelo no encontrado”**

* Verifica `HEXAMIND_YOLO_MODEL` apunte a un `.pt` existente.

**CUDA inválido**

* El loader fuerza `device=cpu` si no hay GPU; limpia `CUDA_VISIBLE_DEVICES` si es necesario.

**Cámara**

* `ls -l /dev/video*`, grupo `video` para el usuario.
* Para CSI/GStreamer, adapta `camera.py`.

**LiDAR**

* Grupos: `sudo usermod -aG dialout $USER` y relogin.
* Detecta puerto/baud con `probe_lidar.py`.
* Considera regla udev para alias fijo `/dev/lidar`.

**WebSocket `CancelledError`**

* Esperado al cerrar cliente; el `ws.py` ya lo captura y cancela tareas limpiamente.

**Rendimiento bajo (CPU)**

* `PD_IMGSZ` a 416/384, `FACE_MAX_SIDE` 480.
* Baja FPS de streams; usa TurboJPEG.

---

## 🛣️ Roadmap

* [ ] Mejoras de performance en Jetson.
* [ ] Múltiples cámaras / switching.
* [ ] Enrutamiento de alertas (voz/sonido).
* [ ] SLAM real (gmapping/cartographer) integrable vía ROS/bridge.

---

## 📜 Licencia

MIT © 2025 — HexaMind UH
