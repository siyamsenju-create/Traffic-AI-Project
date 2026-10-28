# Traffic AI Project

An AI-powered computer vision system for traffic monitoring, vehicle detection, and multi-object tracking using YOLO11 and ByteTrack.

## 🚀 Features

- **Real-Time Vehicle Detection:** Powered by Ultralytics YOLO11 (`yolo11n.pt`).
- **Targeted Classification:** Detects Cars, Motorcycles, Buses, and Trucks (COCO vehicle classes).
- **Multi-Object Tracking:** Uses ByteTrack algorithm for persistent track ID assignment across frames.
- **Traffic Analytics:** Tracks unique vehicles observed, frame processing metrics, and on-screen live vehicle counts.
- **Calibration Frame Extraction:** Utility to capture reference frames for geometric calibration and camera alignment.

---

## 📁 Project Structure

```text
Traffic AI Project/
├── data/
│   └── videos/
│       └── traffic.mp4            # Sample traffic video input
├── outputs/
│   ├── calibration_frame.jpg      # Extracted sample calibration frame
│   ├── road_points.json           # Selected road coordinate points
│   ├── road_points_preview.jpg    # Road boundary overlay preview
│   ├── road_warp_preview.jpg      # Perspective warp preview
│   ├── motion_observations.csv    # Frame-by-frame tracked motion data
│   ├── motion_summary.csv         # Aggregated track displacement summary
│   ├── motion_quality_summary.csv # Cleaned & filtered motion metrics
│   ├── motion_analysis.mp4        # Annotated motion video (gitignored)
│   └── tracked_traffic.mp4        # Detection tracking video (gitignored)
├── src/
│   ├── extract_frame.py           # Calibration frame extractor
│   ├── vehicle_tracker.py         # Main YOLO11 + ByteTrack pipeline
│   ├── select_road_points.py      # Interactive road point selector
│   ├── preview_road_points.py     # Road points polygon visualizer
│   ├── road_calibration.py        # Perspective transformation & warp
│   ├── motion_analysis.py         # Motion analysis & displacement tracking
│   └── clean_motion_report.py     # Noise filtering & motion quality report
├── tests/                         # Unit and integration tests
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/siyamsenju-create/Traffic-AI-Project.git
cd "Traffic-AI-Project"
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage Workflow

### 1. Extract Calibration Frame
Extracts a frame from the video (default: at 5.0 seconds) for camera calibration or ROI setup:
```bash
python3 src/extract_frame.py
```
*Output:* `outputs/calibration_frame.jpg`

### 2. Select & Calibrate Road Region
Select 4 corner points on the road to establish the perspective region:
```bash
# Interactive selection (GUI window)
python3 src/select_road_points.py

# Preview the selected polygon
python3 src/preview_road_points.py

# Generate perspective warp preview
python3 src/road_calibration.py
```
*Outputs:* `outputs/road_points.json`, `outputs/road_points_preview.jpg`, `outputs/road_warp_preview.jpg`

### 3. Vehicle Tracking Pipeline
Runs object detection and ByteTrack tracking on `data/videos/traffic.mp4`:
```bash
python3 src/vehicle_tracker.py
```
*Output video saved to:* `outputs/tracked_traffic.mp4`

### 4. Motion Analysis & Tracking
Tracks bottom-center positions of vehicles across frames, recording pixel displacements and motion trajectories:
```bash
python3 src/motion_analysis.py
```
*Outputs:* `outputs/motion_analysis.mp4`, `outputs/motion_observations.csv`, `outputs/motion_summary.csv`

### 5. Clean Motion & Quality Report
Applies rolling median smoothing, filters out short flickers (< 1.0s), and flags outliers:
```bash
python3 src/clean_motion_report.py
```
*Output:* `outputs/motion_quality_summary.csv`

---

## ⚙️ Configuration

Key parameters:
- `MODEL_NAME`: YOLO model weights (default: `yolo11n.pt`)
- `CONFIDENCE_THRESHOLD`: Minimum detection confidence (default: `0.30`)
- `IMAGE_SIZE`: Inference resolution (default: `640`)
- `VEHICLE_CLASSES`: COCO class IDs `[2, 3, 5, 7]` (Car, Motorcycle, Bus, Truck)
- `tracker`: Tracking algorithm config (default: `bytetrack.yaml`)

---

## 📄 License
This project is licensed under the MIT License.
