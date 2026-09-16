Smart Surveillance System

**CSCI435 – Computer Vision Algorithms and Systems**
University of Wollongong in Dubai

A real-time smart surveillance web application that combines four computer vision techniques:

1. **Background Subtraction / Motion Detection** – OpenCV MOG2
2. **Object Detection** – YOLOv8 nano (`yolov8n.pt`)
3. **Object Tracking** – YOLOv8 nano + ByteTrack
4. **Face Detection** – MediaPipe
5. **Instance Segmentation** – YOLOv8 nano-seg (`yolov8n-seg.pt`)

Built as a group project for CSCI435. Group member names and student IDs have been omitted from this public copy for privacy.

## Prerequisites

- Python 3.10 or higher (3.11 recommended)
- pip package manager
- A webcam (optional – image/video upload works without one)
- ~2 GB free disk space (for model weights)

## Installation

### 1. Clone / unzip the project
```
cd csci435-surveillance
```

### 2. Create and activate a virtual environment
```
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

Note: PyTorch will be downloaded automatically (~800 MB for the CPU build).
For GPU support install the CUDA version of PyTorch first: https://pytorch.org/get-started/locally/

## Running the Application

The system consists of two processes that must both be running at the same time.

### Step 1 – Start the FastAPI backend

Open a terminal, activate the virtual environment, then run:
```
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first run Ultralytics will automatically download `yolov8n.pt` and `yolov8n-seg.pt` (~6 MB + ~7 MB) from the internet.

### Step 2 – Start the Streamlit frontend

Open a second terminal, activate the virtual environment, then run:
```
cd frontend
streamlit run app.py
```

Streamlit will open the app in your default browser at http://localhost:8501.

### Step 3 – Use the app

Use the sidebar to:
- Choose input source (Webcam / Image Upload / Video Upload)
- Enable / disable individual CV modules
- Adjust confidence threshold
- Capture a webcam frame or upload a file
- View the annotated output and detection statistics

## API Endpoints (FastAPI)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Backend liveness check |
| POST | `/detect` | Object detection only |
| POST | `/track` | Object tracking only |
| POST | `/faces` | Face detection only |
| POST | `/segment` | Instance segmentation only |
| POST | `/background` | Motion detection only |
| POST | `/pipeline` | All enabled modules combined |
| POST | `/reset_tracker` | Reset ByteTrack state |
| POST | `/reset_background` | Reset MOG2 background model |

Interactive API docs: http://localhost:8000/docs

## Folder Structure

```
csci435-surveillance/
├── backend/
│   ├── main.py              ← FastAPI app and all REST endpoints
│   ├── detection.py         ← YOLOv8n object detection module
│   ├── tracking.py          ← YOLOv8n + ByteTrack tracking module
│   ├── face_detection.py    ← MediaPipe face detection module
│   ├── segmentation.py      ← YOLOv8n-seg instance segmentation module
│   ├── background.py        ← OpenCV MOG2 background subtraction module
│   └── utils.py             ← Shared utilities (FPS, drawing, encoding)
├── frontend/
│   └── app.py               ← Streamlit web UI
├── models/                  ← YOLOv8 weights (auto-downloaded on first run)
├── fine_tuning/
│   ├── train.py             ← Fine-tuning script (Colab-compatible)
│   └── dataset_prep.py      ← Dataset download and validation helper
├── samples/                 ← place test images/videos here
├── requirements.txt
└── README.md
```

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Primary language |
| FastAPI | 0.111+ | REST API backend |
| Uvicorn | 0.29+ | ASGI server |
| Streamlit | 1.35+ | Web frontend |
| OpenCV | 4.9+ | Image I/O, MOG2 |
| Ultralytics YOLOv8 | 8.2+ | Detection, tracking, segmentation |
| MediaPipe | 0.10+ | Face detection |
| scikit-image | 0.23+ | Additional image processing utilities |
| PyTorch | 2.3+ | Deep learning backend |
| NumPy | 1.26+ | Numerical arrays |
| Pillow | 10.3+ | Image encoding |

## Performance Notes

- The backend targets ≥10 FPS on CPU using the YOLOv8 nano models.
- Enabling all five modules simultaneously reduces FPS; disable unused modules in the sidebar.
- GPU inference (CUDA) provides 3–5× speedup – set `device="0"` in each module's constructor.

## License

Originally submitted as academic coursework at the University of Wollongong in Dubai (CSCI435). Shared publicly as a portfolio sample.
