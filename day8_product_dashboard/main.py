import os
import cv2
import numpy as np
import shutil
import asyncio
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
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

print("Initializing FactoryTwin Engine...")
engine = FactoryTwin()
print("Engine Ready.")

active_connections = []
main_event_loop = None


def sanitize_filename(filename: str, fallback: str) -> str:
    cleaned = os.path.basename(filename or fallback).strip()
    return cleaned or fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="FactoryTwin Premium API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)


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


def to_even(value: int) -> int:
    """Video encoders and browser players are more reliable with even dimensions."""
    value = max(2, int(value))
    return value if value % 2 == 0 else value - 1


def resolve_output_size(src_width: int, src_height: int, preset: str):
    preset = (preset or "original").lower()
    if preset == "original":
        return to_even(src_width), to_even(src_height)

    if preset == "720p":
        target_h = 720
    elif preset == "480p":
        target_h = 480
    else:
        target_h = src_height

    scale = float(target_h) / float(max(1, src_height))
    target_w = int(round(src_width * scale))
    return to_even(target_w), to_even(target_h)


def open_video_writer(path: str, fps: float, width: int, height: int):
    """
    Use mp4v first because some Windows OpenCV builds fail on H.264/OpenH264.
    Returns (writer, codec_name).
    """
    for codec in ("mp4v",):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec
        writer.release()
    return None, None


def process_video_task(input_path, output_path, task_id, enable_ppe, enable_sam2, process_seconds, output_preset):
    cap = None
    out = None
    codec_name = "unknown"
    out_width = 0
    out_height = 0

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

        max_frames = None
        if process_seconds and process_seconds > 0:
            max_frames = max(1, int(round(process_seconds * fps)))

        effective_total = total_frames
        if max_frames is not None:
            effective_total = min(total_frames, max_frames) if total_frames else max_frames

        out_width, out_height = resolve_output_size(width, height, output_preset)
        out, codec_name = open_video_writer(output_path, fps, out_width, out_height)
        if out is None:
            raise RuntimeError("Could not initialize output video writer with supported codecs.")

        engine.tracker.reset()
        engine.analytics.reset()
        frame_count = 0

        while cap.isOpened():
            if max_frames is not None and frame_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = engine.process_frame(frame, enable_ppe, enable_sam2)
            output_frame = frame if processed_frame is None else processed_frame

            if output_frame.shape[1] != out_width or output_frame.shape[0] != out_height:
                output_frame = cv2.resize(output_frame, (out_width, out_height), interpolation=cv2.INTER_AREA)

            out.write(output_frame)
            frame_count += 1

            if frame_count % 5 == 0 or (effective_total and frame_count >= effective_total):
                stats = engine.analytics.get_last_stats()
                logs = engine.analytics.get_log_text()
                log_lines = [line for line in logs.splitlines() if line.strip()]
                progress_msg = {
                    "type": "progress",
                    "current": frame_count,
                    "total": effective_total if effective_total else max(frame_count, 1),
                    "stats": stats,
                    "latest_log": log_lines[-1] if log_lines else ""
                }
                schedule_progress_notification(task_id, progress_msg)

        # Finalize encoded file before notifying client so browser can load it.
        cap.release()
        cap = None
        out.release()
        out = None

        if (not os.path.exists(output_path)) or os.path.getsize(output_path) <= 0:
            raise RuntimeError("Processed output video was not generated correctly.")

        final_msg = {
            "type": "complete",
            "processed_url": f"/static/temp/{os.path.basename(output_path)}",
            "stats": engine.analytics.get_last_stats(),
            "logs": engine.analytics.get_log_text(),
            "codec": codec_name,
            "resolution": f"{out_width}x{out_height}"
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
    enable_sam2: bool = Form(False),
    process_seconds: float = Form(0.0),
    output_preset: str = Form("original")
):
    safe_filename = sanitize_filename(file.filename, "upload.mp4")
    stem, _ = os.path.splitext(safe_filename)
    stem = stem or "upload"

    if process_seconds < 0:
        return JSONResponse({"status": "error", "message": "process_seconds must be 0 or greater"}, status_code=400)

    if process_seconds > 7200:
        return JSONResponse({"status": "error", "message": "process_seconds is too large (max 7200 seconds)"}, status_code=400)

    if output_preset not in {"original", "720p", "480p"}:
        return JSONResponse({"status": "error", "message": "Invalid output_preset"}, status_code=400)

    task_id = f"task_{stem}_{np.random.randint(1000, 9999)}"
    input_path = os.path.join(TEMP_DIR, f"raw_{task_id}_{safe_filename}")
    output_path = os.path.join(TEMP_DIR, f"processed_vid_{task_id}.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    thread = threading.Thread(
        target=process_video_task,
        args=(input_path, output_path, task_id, enable_ppe, enable_sam2, process_seconds, output_preset),
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
