# app/web/ws.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio, time, contextlib
from typing import Optional

ws_app = FastAPI()  

from ..core.bus import Bus
from ..core.settings import WS_RATE_HZ
from ..motion.controller_vel import MotionControllerVel
from ..motion.controller_height import HeightController  # soporte de altura

BUS: Optional[Bus] = None

# -------- Singletons perezosos --------
_controller: Optional[MotionControllerVel] = None
def ctl() -> MotionControllerVel:
    global _controller
    if _controller is None:
        _controller = MotionControllerVel(hz=15.0, deadman_s=0.8)
        # En esta versión con hilo, start() ignora el loop, pero mantenemos la firma
        _controller.start(asyncio.get_event_loop())
    return _controller

_hctl: Optional[HeightController] = None
def hctl() -> HeightController:
    global _hctl
    if _hctl is None:
        _hctl = HeightController(hz=15.0, step=2)
        _hctl.start()
    return _hctl

@ws_app.websocket("/")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if BUS is None:
        await ws.close()
        return

    # Suscríbete a todo lo relevante para enviar al cliente.
    # Añadimos "motion" y "pose" para snapshots de estado.
    sub = BUS.subscribe(["telemetry", "alert", "mode", "ui_event", "pose", "motion"])

    # Coalescer (latest-wins) del setpoint que llega del cliente
    latest_sp = {"x": 0, "y": 0, "z": 0, "speed": None}
    latest_h  = {"h": 0}

    OUT_HZ        = max(1.0, float(WS_RATE_HZ))  # envío coalesced hacia el cliente
    IN_APPLY_HZ   = 30.0                         # aplicar setpoint a controladores
    MIN_PARSE_DT  = 1.0 / 60.0                   # no parsear más de 60 Hz

    last_recv_ts: float = 0.0
    last_send_ts: float = 0.0
    pending_out = {}

    loop = asyncio.get_event_loop()
    stop_evt = asyncio.Event()

    # (Opcional) emitir a BUS solo si cambió: ahorra tráfico
    _last_motion = None
    _last_pose   = None

    # Track para detectar transición activo -> inactivo y re-aplicar pose
    _last_active = False

    async def apply_loop():
        """Aplica el último setpoint de velocidad a tasa fija (latest-wins)."""
        nonlocal _last_motion, _last_active
        period = 1.0 / IN_APPLY_HZ
        try:
            while not stop_evt.is_set():
                t0 = loop.time()
                try:
                    sp = latest_sp.copy()
                    await loop.run_in_executor(None, ctl().set_vel, sp["x"], sp["y"], sp["z"], sp["speed"])

                    # --- NUEVO: detectar activo/inactivo ---
                    active = bool(sp["x"] or sp["y"] or sp["z"])
                    if _last_active and not active:
                        # Transición activo -> inactivo:
                        # el driver suele llamar stay_put() y resetear altura → re-aplica la pose.
                        h_val = int(latest_h["h"])
                        await loop.run_in_executor(None, hctl().set_height, h_val)
                    _last_active = active

                    # Publicar snapshot motion → BUS (solo si cambió)
                    snap = ctl().snapshot()  # dict(x,y,z,speed,ts,gen)
                    if snap != _last_motion:
                        snap["server_ts"] = time.time()
                        await BUS.publish("motion", snap)
                        _last_motion = snap
                except Exception:
                    pass

                dt = period - (loop.time() - t0)
                if dt > 0:
                    try:
                        await asyncio.sleep(dt)
                    except asyncio.CancelledError:
                        break
        except asyncio.CancelledError:
            pass

    async def apply_height_loop():
        """Aplica el último setpoint de altura a tasa fija; el hilo interno suaviza."""
        nonlocal _last_pose
        period = 1.0 / IN_APPLY_HZ
        try:
            while not stop_evt.is_set():
                t0 = loop.time()
                try:
                    h_val = int(latest_h["h"])
                    await loop.run_in_executor(None, hctl().set_height, h_val)
                    # Publicar snapshot pose → BUS (solo si cambió)
                    hsnap = hctl().snapshot()  # dict(h,cur_h,ts,gen)
                    if hsnap != _last_pose:
                        hsnap["server_ts"] = time.time()
                        await BUS.publish("pose", hsnap)
                        _last_pose = hsnap
                except Exception:
                    pass

                dt = period - (loop.time() - t0)
                if dt > 0:
                    try:
                        await asyncio.sleep(dt)
                    except asyncio.CancelledError:
                        break
        except asyncio.CancelledError:
            pass

    applier    = asyncio.create_task(apply_loop())
    h_applier  = asyncio.create_task(apply_height_loop())

    try:
        while True:
            recv_task = asyncio.create_task(ws.receive_json())
            bus_task  = asyncio.create_task(sub.get())

            try:
                done, pending = await asyncio.wait(
                    {recv_task, bus_task},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=max(1.0 / OUT_HZ, 0.02)
                )
            except asyncio.CancelledError:
                break

            now = time.time()

            # ---- Entrada del cliente ----
            if recv_task in done:
                try:
                    msg = recv_task.result()
                except Exception:
                    for p in pending: p.cancel()
                    break

                if (now - last_recv_ts) >= MIN_PARSE_DT:
                    last_recv_ts = now
                    mtype = msg.get("type") or msg.get("topic")

                    # Movimiento XYZ absoluto
                    if mtype == "motion_setpoint":
                        try:
                            latest_sp["x"] = int(msg.get("x", 0))
                            latest_sp["y"] = int(msg.get("y", 0))
                            latest_sp["z"] = int(msg.get("z", 0))
                            spd = msg.get("speed", None)
                            latest_sp["speed"] = int(spd) if spd is not None else None
                        except Exception:
                            pass

                    # Altura absoluta
                    elif mtype == "pose_setpoint":
                        try:
                            h_val = msg.get("h", msg.get("height", 0))
                            latest_h["h"] = max(-30, min(30, int(h_val)))
                        except Exception:
                            pass

                    # Altura incremental (nudges)
                    elif mtype == "pose_nudge":
                        try:
                            dh = int(msg.get("dh", 0))
                            latest_h["h"] = max(-30, min(30, int(latest_h["h"] + dh)))
                        except Exception:
                            pass

                    # (Opcional) cambio de fineza del paso de altura
                    elif mtype == "pose_step":
                        try:
                            step = int(msg.get("step", 2))
                            await loop.run_in_executor(None, hctl().set_step, step)
                        except Exception:
                            pass

                    else:
                        # Passthrough de otros topics al BUS (ui_event, etc.)
                        topic = msg.get("topic"); data = msg.get("data")
                        if topic and (data is not None):
                            await BUS.publish(topic, data)

            # ---- Mensajes del BUS → cliente (coalesced) ----
            if bus_task in done:
                try:
                    topic, data = bus_task.result()
                    pending_out[topic] = data
                except Exception:
                    pass

            # ---- Flush hacia el cliente a OUT_HZ ----
            if (now - last_send_ts) >= (1.0 / OUT_HZ) and pending_out:
                try:
                    for topic, data in list(pending_out.items()):
                        await ws.send_json({"topic": topic, "data": data})
                    pending_out.clear()
                    last_send_ts = now
                except Exception:
                    for p in pending: p.cancel()
                    break

            for p in pending:
                p.cancel()

    except WebSocketDisconnect:
        pass
    finally:
        # parada ordenada
        stop_evt.set()
        applier.cancel()
        h_applier.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await applier
        with contextlib.suppress(asyncio.CancelledError):
            await h_applier
        sub.close()
        with contextlib.suppress(Exception):
            await ws.close()

