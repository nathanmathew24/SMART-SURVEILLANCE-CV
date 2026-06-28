"""
app.py - Streamlit frontend for the CSCI435 Smart Surveillance System.

This single-page application provides:
  • Webcam live feed processing (via st.camera_input)
  • Image file upload processing
  • Video file upload processing (frame-by-frame)
  • Sidebar controls to enable/disable each CV module
  • Real-time FPS display and detection statistics
"""

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
#  Config                                                                      #
# --------------------------------------------------------------------------- #
API_URL = "http://localhost:8000"
PAGE_TITLE = "Smart Surveillance System — CSCI435"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _check_backend() -> bool:
    """Return True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _b64_to_pil(b64: str) -> Image.Image:
    """Decode a base-64 JPEG string to a PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data))


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Encode a PIL Image to bytes (for multipart upload)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _numpy_to_bytes(arr: np.ndarray) -> bytes:
    """Encode a BGR numpy array to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _call_api(endpoint: str, img_bytes: bytes, extra_data: dict | None = None) -> dict | None:
    """POST an image to *endpoint* and return the parsed JSON response."""
    data = extra_data or {}
    try:
        resp = requests.post(
            f"{API_URL}/{endpoint}",
            files={"file": ("frame.jpg", img_bytes, "image/jpeg")},
            data=data,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        st.warning(f"API error {resp.status_code}: {resp.text[:200]}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend.  Make sure the FastAPI server is running on port 8000.")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def _pipeline_data(opts: dict, conf: float) -> dict:
    """Build the form-data dict for the /pipeline endpoint."""
    return {
        "enable_detection": str(opts["detection"]).lower(),
        "enable_tracking": str(opts["tracking"]).lower(),
        "enable_faces": str(opts["faces"]).lower(),
        "enable_segmentation": str(opts["segmentation"]).lower(),
        "enable_background": str(opts["background"]).lower(),
        "conf": str(conf),
    }


def _stats_box(result: dict, opts: dict) -> None:
    """Render detection statistics below the image."""
    cols = st.columns(5)
    with cols[0]:
        fps_val = result.get("fps", 0)
        color = "green" if fps_val >= 10 else "orange"
        st.markdown(f"**FPS** :{'green' if fps_val >= 10 else 'red'}[{fps_val:.1f}]")
    with cols[1]:
        st.metric("Objects", len(result.get("detections", [])) or len(result.get("tracks", [])))
    with cols[2]:
        st.metric("Faces", len(result.get("faces", [])))
    with cols[3]:
        st.metric("Segments", len(result.get("segments", [])))
    with cols[4]:
        motion = result.get("motion_detected", False)
        st.markdown(f"**Motion** {'🔴 YES' if motion else '🟢 No'}")


# --------------------------------------------------------------------------- #
#  Sidebar                                                                     #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎥 Surveillance Controls")
    st.markdown("---")

    st.subheader("📷 Input Source")
    source = st.radio(
        "Choose input",
        ["Webcam", "Upload Image", "Upload Video"],
        index=0,
    )

    st.markdown("---")
    st.subheader("🧠 CV Modules")

    enable_detection = st.toggle("Object Detection (YOLOv8n)", value=True)
    enable_tracking = st.toggle("Object Tracking (ByteTrack)", value=False)
    enable_faces = st.toggle("Face Detection (MediaPipe)", value=True)
    enable_segmentation = st.toggle("Segmentation (YOLOv8n-seg)", value=False)
    enable_background = st.toggle("Motion Detection (MOG2)", value=True)

    # Tracking and detection are mutually exclusive in the pipeline
    if enable_tracking and enable_detection:
        st.info("Tracking mode replaces plain detection.")

    st.markdown("---")
    st.subheader("⚙️ Parameters")
    conf_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)

    st.markdown("---")
    st.subheader("ℹ️ Backend Status")
    backend_ok = _check_backend()
    if backend_ok:
        st.success("Backend online ✓")
    else:
        st.error("Backend offline ✗")
        st.code("cd backend\nuvicorn main:app --host 0.0.0.0 --port 8000")

    if st.button("🔄 Reset Tracker & BG Model"):
        requests.post(f"{API_URL}/reset_tracker", timeout=3)
        requests.post(f"{API_URL}/reset_background", timeout=3)
        st.success("Reset done.")

# Collect enabled options
cv_opts = {
    "detection": enable_detection,
    "tracking": enable_tracking,
    "faces": enable_faces,
    "segmentation": enable_segmentation,
    "background": enable_background,
}

# --------------------------------------------------------------------------- #
#  Main area                                                                   #
# --------------------------------------------------------------------------- #
st.title("🔍 Smart Surveillance System")
st.caption("CSCI435 – Computer Vision Algorithms and Systems | University of Wollongong in Dubai")

if not backend_ok:
    st.warning("Start the FastAPI backend first (see sidebar), then refresh.")
    st.stop()

# ── Webcam ────────────────────────────────────────────────────────────────── #
if source == "Webcam":
    st.subheader("📷 Live Webcam Feed")
    st.info(
        "Click **Take Photo** to capture a frame.  "
        "For continuous feed, keep clicking or use the loop below."
    )

    cam_frame = st.camera_input("Capture frame")

    col_img, col_stats = st.columns([3, 1])

    if cam_frame is not None:
        img_bytes = cam_frame.getvalue()
        with st.spinner("Processing…"):
            result = _call_api(
                "pipeline",
                img_bytes,
                _pipeline_data(cv_opts, conf_threshold),
            )

        if result and "image" in result:
            with col_img:
                st.image(_b64_to_pil(result["image"]), use_container_width=True)
            with col_stats:
                st.markdown("### Stats")
                st.metric("FPS", f"{result.get('fps', 0):.1f}")
                st.metric("Objects", len(result.get("detections", [])) or len(result.get("tracks", [])))
                st.metric("Faces", len(result.get("faces", [])))
                st.metric("Segments", len(result.get("segments", [])))
                motion = result.get("motion_detected", False)
                st.markdown(f"**Motion:** {'🔴 YES' if motion else '🟢 No'}")

            # Detail expanders
            with st.expander("🔎 Detection details"):
                dets = result.get("detections", []) or result.get("tracks", [])
                if dets:
                    st.dataframe(
                        [{"Label": d.get("class_name", ""), "Conf": f"{d.get('confidence',0):.0%}",
                          "ID": d.get("track_id", "-"), "BBox": str(d.get("bbox"))} for d in dets],
                        use_container_width=True,
                    )
                else:
                    st.write("No objects detected.")

            with st.expander("😊 Face detection details"):
                faces = result.get("faces", [])
                if faces:
                    st.dataframe(
                        [{"Confidence": f"{f['confidence']:.0%}", "BBox": str(f["bbox"])} for f in faces],
                        use_container_width=True,
                    )
                else:
                    st.write("No faces detected.")


# ── Upload Image ──────────────────────────────────────────────────────────── #
elif source == "Upload Image":
    st.subheader("🖼️ Image Upload")
    uploaded = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded:
        img_bytes = uploaded.getvalue()
        col_orig, col_proc = st.columns(2)

        with col_orig:
            st.markdown("**Original**")
            st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)

        with st.spinner("Running CV pipeline…"):
            result = _call_api(
                "pipeline",
                img_bytes,
                _pipeline_data(cv_opts, conf_threshold),
            )

        if result and "image" in result:
            with col_proc:
                st.markdown("**Processed**")
                st.image(_b64_to_pil(result["image"]), use_container_width=True)

            _stats_box(result, cv_opts)

            with st.expander("📋 Full JSON response"):
                display = {k: v for k, v in result.items() if k != "image"}
                st.json(display)


# ── Upload Video ──────────────────────────────────────────────────────────── #
elif source == "Upload Video":
    st.subheader("🎬 Video Upload")
    uploaded_vid = st.file_uploader(
        "Choose a video", type=["mp4", "avi", "mov", "mkv", "webm"]
    )

    if uploaded_vid:
        # Save to a temp file so OpenCV can read it
        import tempfile, os

        suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0

        st.info(f"Video: {total_frames} frames @ {fps_video:.1f} FPS  |  Processing every frame…")

        # Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            frame_skip = st.number_input("Process every N-th frame", 1, 10, 1)
        with c2:
            max_frames = st.number_input("Max frames to process", 10, min(500, total_frames), 60)
        with c3:
            playback_fps = st.number_input("Display FPS", 1, 30, 10)

        run_btn = st.button("▶ Process Video", type="primary")

        if run_btn:
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()
            progress = st.progress(0)

            requests.post(f"{API_URL}/reset_tracker", timeout=3)
            requests.post(f"{API_URL}/reset_background", timeout=3)

            processed = 0
            frame_idx = 0
            delay = 1.0 / playback_fps

            while cap.isOpened() and processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                img_bytes = _numpy_to_bytes(frame)
                result = _call_api(
                    "pipeline",
                    img_bytes,
                    _pipeline_data(cv_opts, conf_threshold),
                )

                if result and "image" in result:
                    frame_placeholder.image(
                        _b64_to_pil(result["image"]),
                        use_container_width=True,
                        caption=f"Frame {frame_idx}/{total_frames} | FPS: {result.get('fps', 0):.1f}",
                    )
                    with stats_placeholder.container():
                        _stats_box(result, cv_opts)

                processed += 1
                progress.progress(min(processed / max_frames, 1.0))
                time.sleep(delay)

            cap.release()
            os.unlink(tmp_path)
            st.success(f"Done – processed {processed} frames.")
