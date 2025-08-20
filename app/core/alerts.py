# app/core/alerts.py
import asyncio, time
from .bus import Bus

async def alerts_loop(bus: Bus):
    low_fps_since = None
    while True:
        tel = await bus.last("telemetry") or {}
        fps = tel.get("fps", 0)
        now = time.time()

        if fps < 5:
            low_fps_since = low_fps_since or now
            if now - low_fps_since > 3:  # 3s con FPS bajo
                await bus.publish("alert", {
                    "code": "LOW_FPS",
                    "level": "warn",
                    "message": f"FPS bajo: {fps}",
                    "ts": now
                })
        else:
            low_fps_since = None

        await asyncio.sleep(0.25)
