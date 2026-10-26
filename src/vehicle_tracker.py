
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "videos" / "traffic.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "tracked_traffic.mp4"

# Detection settings
MODEL_NAME = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.30
IMAGE_SIZE = 640

# COCO class IDs: car, motorcycle, bus, truck
VEHICLE_CLASSES = [2, 3, 5, 7]


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}\n"
            "Put your video at data/videos/traffic.mp4"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Invalid video metadata.")

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create output video.")

    frame_number = 0
    unique_track_ids = set()
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()

            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )[0]

            annotated_frame = result.plot()
            vehicle_count = 0

            if result.boxes is not None:
                vehicle_count = len(result.boxes)

                if result.boxes.id is not None:
                    ids = result.boxes.id.int().cpu().tolist()
                    unique_track_ids.update(ids)

            cv2.putText(
                annotated_frame,
                f"Vehicles: {vehicle_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            writer.write(annotated_frame)
            frame_number += 1

            if frame_number % 50 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} frames"
                )

    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - start_time

    print("\n--- Results ---")
    print(f"Frames processed: {frame_number}")
    print(f"Unique track IDs observed: {len(unique_track_ids)}")
    print(f"Processing time: {elapsed:.1f} seconds")
    print(f"Output saved to: {OUTPUT_PATH}")

    if frame_number == 0:
        raise RuntimeError("No frames were processed.")

    if not OUTPUT_PATH.exists() or OUTPUT_PATH.stat().st_size == 0:
        raise RuntimeError("Output video is empty.")

    print("SUCCESS: Vehicle tracking video created!")


if __name__ == "__main__":
    main()
