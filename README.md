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
│   └── tracked_traffic.mp4        # Generated tracked video (gitignored)
├── src/
│   ├── extract_frame.py           # Calibration frame extractor
│   └── vehicle_tracker.py         # Main YOLO11 + ByteTrack pipeline
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

## 🚦 Usage

### 1. Extract Calibration Frame
Extracts a frame from the video (default: at 5.0 seconds) for camera calibration or ROI setup:
```bash
python3 src/extract_frame.py
```
*Output saved to:* `outputs/calibration_frame.jpg`

### 2. Run Vehicle Tracking Pipeline
Runs object detection and ByteTrack tracking on `data/videos/traffic.mp4`:
```bash
python3 src/vehicle_tracker.py
```
*Model weights (`yolo11n.pt`) are downloaded automatically on the first run.*  
*Output video saved to:* `outputs/tracked_traffic.mp4`

---

## ⚙️ Configuration

Key parameters in `src/vehicle_tracker.py`:
- `MODEL_NAME`: YOLO model weights (default: `yolo11n.pt`)
- `CONFIDENCE_THRESHOLD`: Minimum detection confidence (default: `0.30`)
- `IMAGE_SIZE`: Inference resolution (default: `640`)
- `VEHICLE_CLASSES`: COCO class IDs `[2, 3, 5, 7]` (Car, Motorcycle, Bus, Truck)
- `tracker`: Tracking algorithm config (default: `bytetrack.yaml`)

---

## 📄 License
This project is licensed under the MIT License.
