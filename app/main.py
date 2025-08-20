# app/main.py
from contextlib import asynccontextmanager
import asyncio, time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2

from .core.bus import Bus
from .core.settings import HTTP_PORT
from .core.alerts import alerts_loop
from .sensors.camera import camera, get_telemetry_snapshot
from .web.routes_stream import router as stream_router
from .web.routes_status import router as status_router
from .web.routes_control import router as control_router
from .web import ws as ws_module
from .motion.controller_vel import MotionControllerVel
from .web.routes_stream_lidar import router as lidar_router
from .web.follow import router as follow_router


bus = Bus()
t0 = time.time()
_bg_tasks: list[asyncio.Task] = []

# ↓↓↓ Mejor en Jetson/ARM: sin hilos internos de OpenCV
try:
    cv2.setNumThreads(0)
except Exception:
    pass

async def telemetry_loop():
    try:
        while True:
            await bus.publish("telemetry", get_telemetry_snapshot())
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        # salir silencioso en shutdown
        return

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[BOOT] starting MotionController + camera + background tasks")

    # Controlador de movimiento
    motion_controller = MotionControllerVel(hz=15.0, deadman_s=0.8)
    motion_controller.start(asyncio.get_event_loop())

    # Inyecta BUS a módulos web
    ws_module.BUS = bus
    from .web import routes_status, routes_control
    routes_status.BUS = bus
    routes_control.BUS = bus

    # Abrir cámara (tu módulo de cámara debería tener hilo de captura)
    try:
        camera.open()
        print("[INFO] Cámara abierta")
    except Exception as e:
        print("[WARN] No se pudo abrir la cámara:", e)

    # Tareas de fondo
    _bg_tasks.append(asyncio.create_task(telemetry_loop()))
    _bg_tasks.append(asyncio.create_task(alerts_loop(bus)))

    try:
        yield
    finally:
        print("[SHUTDOWN] stopping background tasks")
        for t in _bg_tasks:
            t.cancel()
        await asyncio.gather(*_bg_tasks, return_exceptions=True)

        # Si MotionControllerVel tiene stop(), descomenta:
        # try:
        #     motion_controller.stop()
        # except Exception:
        #     pass

        try:
            camera.release()
            print("[INFO] Cámara cerrada")
        except Exception:
            pass

app = FastAPI(title="HexaMind Robot Server (Phase 1)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status_router, prefix="", tags=["status"])
app.include_router(stream_router, prefix="", tags=["stream"])
app.include_router(control_router, prefix="/control", tags=["control"])
app.include_router(lidar_router)
app.mount("/ws", ws_module.ws_app)
app.include_router(follow_router)

