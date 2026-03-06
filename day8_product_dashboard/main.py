import os
import cv2
import numpy as np
import shutil
import asyncio
import threading
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import json

# --- ABSOLUTE PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMP_DIR = os.path.join(STATIC_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Absolute Path Diagnostics (Importing after BASE_DIR setup)
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import analytics
import factory_twin
import tracker
import ppe_detector
import sam2_segmenter

print("--- CORE ENGINE DIAGNOSTICS ---")
print(f"BASE_DIR: {BASE_DIR}")
print(f"analytics: {os.path.abspath(analytics.__file__)}")
print(f"factory_twin: {os.path.abspath(factory_twin.__file__)}")
print(f"tracker: {os.path.abspath(tracker.__file__)}")
print(f"ppe_detector: {os.path.abspath(ppe_detector.__file__)}")
print(f"sam2_segmenter: {os.path.abspath(sam2_segmenter.__file__)}")
print("-------------------------------")

from factory_twin import FactoryTwin

app = FastAPI(title="FactoryTwin Premium API")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

print("Initializing FactoryTwin Engine...")
engine = FactoryTwin()
print("Engine Ready.")

active_connections = []
main_event_loop = None


def sanitize_filename(filename: str, fallback: str) -> str:
    cleaned = os.path.basename(filename or fallback).strip()
    return cleaned or fallback


@app.on_event("startup")
async def on_startup():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/process-image")
async def api_process_image(
    file: UploadFile = File(...),
    enable_ppe: bool = Form(True),
    enable_sam2: bool = Form(False)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"status": "error", "message": "Invalid image format"}, status_code=400)

    safe_filename = sanitize_filename(file.filename, "upload.jpg")

    engine.tracker.reset()
    engine.analytics.reset()
    processed_img = engine.process_frame(img, enable_ppe, enable_sam2)

    out_filename = f"processed_{safe_filename}"
    out_path = os.path.join(TEMP_DIR, out_filename)
    write_ok = cv2.imwrite(out_path, img if processed_img is None else processed_img)

    if not write_ok:
        return JSONResponse({"status": "error", "message": "Failed to write processed image"}, status_code=500)

    logs = engine.analytics.get_log_text()
    stats = engine.analytics.get_last_stats()

    return {
        "status": "success",
        "processed_url": f"/static/temp/{out_filename}",
        "logs": logs,
        "stats": stats
    }


@app.websocket("/ws/progress/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections.append((task_id, websocket))
    try:
        while True:
            # Keep the socket alive while the background worker pushes updates.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"WS Exception: {exc}")
    finally:
        active_connections[:] = [entry for entry in active_connections if entry[1] != websocket]


async def notify_progress(task_id: str, data: dict):
    dead_connections = []
    for tid, ws in list(active_connections):
        if tid != task_id:
            continue
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead_connections.append((tid, ws))

    for entry in dead_connections:
        if entry in active_connections:
            active_connections.remove(entry)


def schedule_progress_notification(task_id: str, data: dict):
    if main_event_loop is None:
        return
    asyncio.run_coroutine_threadsafe(notify_progress(task_id, data), main_event_loop)


def process_video_task(input_path, output_path, task_id, enable_ppe, enable_sam2):
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Uploaded video could not be opened.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 24.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Uploaded video has invalid frame dimensions.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError("Could not initialize output video writer.")

        engine.tracker.reset()
        engine.analytics.reset()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = engine.process_frame(frame, enable_ppe, enable_sam2)
            out.write(frame if processed_frame is None else processed_frame)
            frame_count += 1

            if frame_count % 5 == 0 or (total_frames and frame_count >= total_frames):
                stats = engine.analytics.get_last_stats()
                logs = engine.analytics.get_log_text()
                log_lines = [line for line in logs.splitlines() if line.strip()]
                progress_msg = {
                    "type": "progress",
                    "current": frame_count,
                    "total": total_frames if total_frames else max(frame_count, 1),
                    "stats": stats,
                    "latest_log": log_lines[-1] if log_lines else ""
                }
                schedule_progress_notification(task_id, progress_msg)

        final_msg = {
            "type": "complete",
            "processed_url": f"/static/temp/{os.path.basename(output_path)}",
            "stats": engine.analytics.get_last_stats(),
            "logs": engine.analytics.get_log_text()
        }
        schedule_progress_notification(task_id, final_msg)
    except Exception as exc:
        print(f"CRITICAL ERROR IN VIDEO TASK: {exc}")
        import traceback
        traceback.print_exc()
        schedule_progress_notification(task_id, {"type": "error", "message": str(exc)})
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()


@app.post("/api/process-video")
async def api_process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_ppe: bool = Form(True),
    enable_sam2: bool = Form(False)
):
    safe_filename = sanitize_filename(file.filename, "upload.mp4")
    stem, _ = os.path.splitext(safe_filename)
    stem = stem or "upload"

    task_id = f"task_{stem}_{np.random.randint(1000, 9999)}"
    input_path = os.path.join(TEMP_DIR, f"raw_{task_id}_{safe_filename}")
    output_path = os.path.join(TEMP_DIR, f"processed_vid_{task_id}.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    thread = threading.Thread(
        target=process_video_task,
        args=(input_path, output_path, task_id, enable_ppe, enable_sam2),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "task_id": task_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


