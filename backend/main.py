"""
main.py - FastAPI backend for the Smart Surveillance System.

Exposes REST endpoints that the Streamlit frontend calls to process images and
video frames.  Each endpoint accepts a raw image upload (multipart/form-data),
runs the requested CV pipeline, and returns a JSON payload containing:
  - The processed frame encoded as a base-64 JPEG string
  - Detection / tracking / segmentation metadata

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from background import BackgroundSubtractor
from detection import ObjectDetector
from face_detection import FaceDetector
from segmentation import Segmenter
from tracking import ObjectTracker
from utils import FPSCounter, draw_fps, encode_frame_to_base64, resize_keep_aspect


# --------------------------------------------------------------------------- #
#  Module singletons – loaded once at startup to avoid repeated model loads    #
# --------------------------------------------------------------------------- #
_modules: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all CV modules on startup; release resources on shutdown."""
    print("[startup] Loading CV modules…")
    try:
        _modules["detector"] = ObjectDetector()
        print("  ✓ ObjectDetector (YOLOv8n)")
    except Exception as exc:
        print(f"  ✗ ObjectDetector failed: {exc}")

    try:
        _modules["tracker"] = ObjectTracker()
        print("  ✓ ObjectTracker (YOLOv8n + ByteTrack)")
    except Exception as exc:
        print(f"  ✗ ObjectTracker failed: {exc}")

    try:
        _modules["face_detector"] = FaceDetector()
        print("  ✓ FaceDetector (MediaPipe)")
    except Exception as exc:
        print(f"  ✗ FaceDetector failed: {exc}")

    try:
        _modules["segmenter"] = Segmenter()
        print("  ✓ Segmenter (YOLOv8n-seg)")
    except Exception as exc:
        print(f"  ✗ Segmenter failed: {exc}")

    _modules["bg_subtractor"] = BackgroundSubtractor()
    print("  ✓ BackgroundSubtractor (MOG2)")

    _modules["fps"] = FPSCounter(window=30)
    print("[startup] Ready.\n")
    yield
    # Cleanup
    if "face_detector" in _modules:
        _modules["face_detector"].close()
    print("[shutdown] CV modules released.")


# --------------------------------------------------------------------------- #
#  App                                                                         #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="CSCI435 Smart Surveillance API",
    description="Computer Vision backend for the Smart Surveillance System.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  Helper                                                                      #
# --------------------------------------------------------------------------- #
async def _read_upload(file: UploadFile) -> np.ndarray:
    """Decode an uploaded image file into a BGR numpy array."""
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- #
#  Endpoints                                                                   #
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    """Quick liveness check."""
    return {"status": "ok", "modules": list(_modules.keys())}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    max_dim: int = Form(1280),
):
    """Object detection endpoint (YOLOv8n).

    Accepts an image upload, runs YOLO inference, and returns the annotated
    frame plus detection metadata.
    """
    if "detector" not in _modules:
        raise HTTPException(503, "ObjectDetector not available")

    frame = await _read_upload(file)
    frame = resize_keep_aspect(frame, max_dim)

    detector: ObjectDetector = _modules["detector"]
    detector.conf = conf
    t0 = time.perf_counter()
    result = detector.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    fps = _modules["fps"].tick()
    draw_fps(result["annotated"], fps)

    # Remove non-serialisable numpy array before returning
    result.pop("annotated")
    for d in result["detections"]:
        d["bbox"] = list(d["bbox"])

    return JSONResponse({
        "image": encode_frame_to_base64(detector.detect(frame)["annotated"]),
        **result,
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
    })


@app.post("/track")
async def track(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    max_dim: int = Form(1280),
):
    """Object tracking endpoint (YOLOv8n + ByteTrack)."""
    if "tracker" not in _modules:
        raise HTTPException(503, "ObjectTracker not available")

    frame = await _read_upload(file)
    frame = resize_keep_aspect(frame, max_dim)

    tracker: ObjectTracker = _modules["tracker"]
    tracker.conf = conf
    t0 = time.perf_counter()
    result = tracker.track(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    fps = _modules["fps"].tick()
    annotated = result.pop("annotated")
    draw_fps(annotated, fps)

    for t in result["tracks"]:
        t["bbox"] = list(t["bbox"])

    return JSONResponse({
        "image": encode_frame_to_base64(annotated),
        **result,
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
    })


@app.post("/faces")
async def detect_faces(
    file: UploadFile = File(...),
    max_dim: int = Form(1280),
):
    """Face detection endpoint (MediaPipe)."""
    if "face_detector" not in _modules:
        raise HTTPException(503, "FaceDetector not available")

    frame = await _read_upload(file)
    frame = resize_keep_aspect(frame, max_dim)

    fd: FaceDetector = _modules["face_detector"]
    t0 = time.perf_counter()
    result = fd.detect(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    fps = _modules["fps"].tick()
    annotated = result.pop("annotated")
    draw_fps(annotated, fps)

    for face in result["faces"]:
        face["bbox"] = list(face["bbox"])

    return JSONResponse({
        "image": encode_frame_to_base64(annotated),
        **result,
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
    })


@app.post("/segment")
async def segment(
    file: UploadFile = File(...),
    conf: float = Form(0.35),
    max_dim: int = Form(1280),
):
    """Instance segmentation endpoint (YOLOv8n-seg)."""
    if "segmenter" not in _modules:
        raise HTTPException(503, "Segmenter not available")

    frame = await _read_upload(file)
    frame = resize_keep_aspect(frame, max_dim)

    seg: Segmenter = _modules["segmenter"]
    seg.conf = conf
    t0 = time.perf_counter()
    result = seg.segment(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    fps = _modules["fps"].tick()
    annotated = result.pop("annotated")
    draw_fps(annotated, fps)

    # Masks are large numpy arrays – strip them from JSON response
    serialisable_segs = []
    for s in result["segments"]:
        serialisable_segs.append({
            k: list(v) if k == "bbox" else v
            for k, v in s.items()
            if k != "mask"
        })

    return JSONResponse({
        "image": encode_frame_to_base64(annotated),
        "segments": serialisable_segs,
        "count": result["count"],
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
    })


@app.post("/background")
async def background_subtraction(
    file: UploadFile = File(...),
    max_dim: int = Form(1280),
):
    """Background subtraction / motion detection endpoint (MOG2)."""
    frame = await _read_upload(file)
    frame = resize_keep_aspect(frame, max_dim)

    bg: BackgroundSubtractor = _modules["bg_subtractor"]
    t0 = time.perf_counter()
    result = bg.process(frame)
    latency_ms = (time.perf_counter() - t0) * 1000

    fps = _modules["fps"].tick()
    annotated = result.pop("annotated")
    draw_fps(annotated, fps)

    # Mask is a numpy array – encode separately or drop
    mask_encoded = encode_frame_to_base64(
        cv2.cvtColor(result.pop("mask"), cv2.COLOR_GRAY2BGR)
    )

    return JSONResponse({
        "image": encode_frame_to_base64(annotated),
        "mask_image": mask_encoded,
        "motion_regions": [list(r) for r in result["motion_regions"]],
        "motion_detected": result["motion_detected"],
        "latency_ms": round(latency_ms, 1),
        "fps": round(fps, 1),
    })


@app.post("/pipeline")
async def full_pipeline(
    file: UploadFile = File(...),
    enable_detection: bool = Form(True),
    enable_tracking: bool = Form(False),
    enable_faces: bool = Form(True),
    enable_segmentation: bool = Form(True),
    enable_background: bool = Form(True),
    conf: float = Form(0.35),
    max_dim: int = Form(1280),
):
    """Unified pipeline endpoint – runs all enabled CV modules on one frame.

    Modules are applied sequentially and each annotates the same running frame
    so the final image shows all overlays combined.
    """
    frame = await _read_upload(file)
    frame = resize_keep_aspect(frame, max_dim)

    output: dict = {
        "detections": [],
        "tracks": [],
        "faces": [],
        "segments": [],
        "motion_regions": [],
        "motion_detected": False,
    }

    current = frame.copy()

    # 1. Background subtraction (operates on original frame for clean mask)
    if enable_background and "bg_subtractor" in _modules:
        bg_result = _modules["bg_subtractor"].process(current)
        output["motion_regions"] = [list(r) for r in bg_result["motion_regions"]]
        output["motion_detected"] = bg_result["motion_detected"]
        # Blend motion highlight into running frame
        cv2.addWeighted(bg_result["annotated"], 0.5, current, 0.5, 0, current)

    # 2. Object detection
    if enable_detection and not enable_tracking and "detector" in _modules:
        det_result = _modules["detector"].detect(current)
        for d in det_result["detections"]:
            d["bbox"] = list(d["bbox"])
        output["detections"] = det_result["detections"]
        current = det_result["annotated"]

    # 3. Object tracking (replaces detection if both selected)
    if enable_tracking and "tracker" in _modules:
        trk_result = _modules["tracker"].track(current)
        for t in trk_result["tracks"]:
            t["bbox"] = list(t["bbox"])
        output["tracks"] = trk_result["tracks"]
        current = trk_result["annotated"]

    # 4. Face detection
    if enable_faces and "face_detector" in _modules:
        face_result = _modules["face_detector"].detect(current)
        for f in face_result["faces"]:
            f["bbox"] = list(f["bbox"])
        output["faces"] = face_result["faces"]
        current = face_result["annotated"]

    # 5. Segmentation
    if enable_segmentation and "segmenter" in _modules:
        seg_result = _modules["segmenter"].segment(current)
        serialisable_segs = [
            {k: list(v) if k == "bbox" else v for k, v in s.items() if k != "mask"}
            for s in seg_result["segments"]
        ]
        output["segments"] = serialisable_segs
        current = seg_result["annotated"]

    fps = _modules["fps"].tick()
    draw_fps(current, fps)

    return JSONResponse({
        "image": encode_frame_to_base64(current),
        **output,
        "fps": round(fps, 1),
    })


@app.post("/reset_tracker")
async def reset_tracker():
    """Reset ByteTrack history (call when switching video source)."""
    if "tracker" in _modules:
        _modules["tracker"].reset()
    return {"status": "tracker reset"}


@app.post("/reset_background")
async def reset_background():
    """Reset MOG2 background model (call when switching video source)."""
    if "bg_subtractor" in _modules:
        _modules["bg_subtractor"].reset()
    return {"status": "background model reset"}
