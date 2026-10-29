
import csv
import math
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT / "data" / "videos" / "traffic.mp4"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_VIDEO = OUTPUT_DIR / "motion_analysis.mp4"
OBSERVATIONS_CSV = OUTPUT_DIR / "motion_observations.csv"
SUMMARY_CSV = OUTPUT_DIR / "motion_summary.csv"

OBSERVATIONS_BACKUP_DIR = ROOT / "backups" / "run_history"

MODEL_NAME = "yolo11n.pt"
CONFIDENCE = 0.30
IMAGE_SIZE = 640

# Ignore movements of 2 pixels or less as likely detection jitter.
MOTION_JITTER_THRESHOLD_PX = 2.0

# COCO vehicle classes: car, motorcycle, bus, truck.
VEHICLE_CLASSES = [2, 3, 5, 7]

TRAIL_LENGTH = 30


# --------------------------------------------------
# 2. BACK UP EXISTING OUTPUTS
# --------------------------------------------------

def backup_previous_outputs():
    """Back up existing results before a new run can overwrite them."""

    existing_outputs = [
        OBSERVATIONS_CSV,
        SUMMARY_CSV,
        OUTPUT_VIDEO,
    ]

    files_to_backup = [
        path for path in existing_outputs if path.is_file()
    ]

    if not files_to_backup:
        print("No previous output files to back up.")
        return

    OBSERVATIONS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for source in files_to_backup:
        backup_path = (
            OBSERVATIONS_BACKUP_DIR
            / f"{source.stem}_{timestamp}{source.suffix}"
        )

        shutil.copy2(source, backup_path)

        print(f"Backed up: {source.name} -> {backup_path}")


# --------------------------------------------------
# 3. MAIN MOTION ANALYSIS
# --------------------------------------------------

def main():
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(
            f"Traffic video not found: {VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Preserve previous results before opening any output for writing.
    backup_previous_outputs()

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid video metadata.")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError(
            f"Could not create output video: {OUTPUT_VIDEO}"
        )

    # Last observed bottom-center point for each track ID.
    previous = {}

    # Historical points used to draw vehicle trajectories.
    trails = defaultdict(list)

    # Measurements aggregated separately for each track.
    track_stats = defaultdict(
        lambda: {
            "observations": 0,
            "first_frame": None,
            "last_frame": None,
            "total_pixel_distance": 0.0,
            "pixel_rates": [],
            "filtered_pixel_distance": 0.0,
            "filtered_pixel_rates": [],
        }
    )

    frame_number = 0
    started = time.perf_counter()

    observation_fields = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "vehicle_class",
        "confidence",
        "bottom_center_x_px",
        "bottom_center_y_px",
        "displacement_px",
        "elapsed_seconds",
        "motion_rate_px_per_second",
        "filtered_displacement_px",
        "filtered_motion_rate_px_per_second",
        "measurement_status",
    ]

    try:
        with OBSERVATIONS_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=observation_fields,
            )
            csv_writer.writeheader()

            while True:
                success, frame = cap.read()

                if not success:
                    break

                timestamp = frame_number / fps

                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASSES,
                    conf=CONFIDENCE,
                    imgsz=IMAGE_SIZE,
                    device="cpu",
                    agnostic_nms=True,
                    verbose=False,
                )[0]

                annotated = result.plot()

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                ):
                    boxes = result.boxes.xyxy.cpu().tolist()
                    ids = result.boxes.id.int().cpu().tolist()
                    classes = result.boxes.cls.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()

                    for box, track_id, cls_id, confidence in zip(
                        boxes,
                        ids,
                        classes,
                        confidences,
                    ):
                        x1, y1, x2, y2 = box

                        # Bottom-center approximates the road contact point.
                        x = (x1 + x2) / 2.0
                        y = y2

                        trails[track_id].append(
                            (frame_number, x, y)
                        )
                        trails[track_id] = trails[track_id][
                            -TRAIL_LENGTH:
                        ]

                        displacement = None
                        elapsed = None
                        motion_rate = None

                        if track_id in previous:
                            old_frame, old_x, old_y = previous[track_id]

                            elapsed = (frame_number - old_frame) / fps

                            if elapsed > 0:
                                displacement = math.hypot(
                                    x - old_x,
                                    y - old_y,
                                )
                                motion_rate = displacement / elapsed

                        previous[track_id] = (
                            frame_number,
                            x,
                            y,
                        )

                        vehicle_name = model.names.get(
                            cls_id,
                            str(cls_id),
                        )

                        # Suppress small frame-to-frame movement.
                        filtered_displacement = None
                        filtered_motion_rate = None

                        if (
                            displacement is not None
                            and elapsed is not None
                            and elapsed > 0
                        ):
                            if (
                                displacement
                                <= MOTION_JITTER_THRESHOLD_PX
                            ):
                                filtered_displacement = 0.0
                            else:
                                filtered_displacement = displacement

                            filtered_motion_rate = (
                                filtered_displacement / elapsed
                            )

                        stats = track_stats[track_id]
                        stats["observations"] += 1

                        if stats["first_frame"] is None:
                            stats["first_frame"] = frame_number

                        stats["last_frame"] = frame_number

                        if displacement is not None:
                            stats["total_pixel_distance"] += displacement

                            if motion_rate is not None:
                                stats["pixel_rates"].append(motion_rate)

                        if filtered_displacement is not None:
                            stats["filtered_pixel_distance"] += (
                                filtered_displacement
                            )

                            if filtered_motion_rate is not None:
                                stats["filtered_pixel_rates"].append(
                                    filtered_motion_rate
                                )

                        csv_writer.writerow(
                            {
                                "frame": frame_number,
                                "timestamp_seconds": round(
                                    timestamp, 4
                                ),
                                "track_id": track_id,
                                "vehicle_class": vehicle_name,
                                "confidence": round(confidence, 4),
                                "bottom_center_x_px": x,
                                "bottom_center_y_px": y,
                                "displacement_px": (
                                    round(displacement, 3)
                                    if displacement is not None
                                    else ""
                                ),
                                "elapsed_seconds": (
                                    round(elapsed, 4)
                                    if elapsed is not None
                                    else ""
                                ),
                                "motion_rate_px_per_second": (
                                    round(motion_rate, 3)
                                    if motion_rate is not None
                                    else ""
                                ),
                                "filtered_displacement_px": (
                                    round(filtered_displacement, 3)
                                    if filtered_displacement is not None
                                    else ""
                                ),
                                "filtered_motion_rate_px_per_second": (
                                    round(filtered_motion_rate, 3)
                                    if filtered_motion_rate is not None
                                    else ""
                                ),
                                "measurement_status": (
                                    "UNVALIDATED_IMAGE_PLANE"
                                ),
                            }
                        )

                # Draw the recent trajectory of each tracked vehicle.
                for trail in trails.values():
                    for i in range(1, len(trail)):
                        _, x_a, y_a = trail[i - 1]
                        _, x_b, y_b = trail[i]

                        cv2.line(
                            annotated,
                            (x_a, y_a),
                            (x_b, y_b),
                            (0, 255, 255),
                            2,
                        )

                cv2.putText(
                    annotated,
                    "EXPERIMENTAL: image motion only",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated,
                    f"Time: {timestamp:.2f}s",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                writer.write(annotated)
                frame_number += 1

                if frame_number % 50 == 0:
                    print(
                        f"Processed {frame_number}/{total_frames} frames"
                    )

    finally:
        cap.release()
        writer.release()

    # --------------------------------------------------
    # 4. EXPORT PER-TRACK SUMMARY
    # --------------------------------------------------

    summary_fields = [
        "track_id",
        "observed_frames",
        "first_timestamp_seconds",
        "last_timestamp_seconds",
        "observed_duration_seconds",
        "accumulated_pixel_displacement",
        "mean_motion_rate_px_per_second",
        "max_motion_rate_px_per_second",
        "filtered_accumulated_pixel_displacement",
        "filtered_mean_motion_rate_px_per_second",
        "filtered_max_motion_rate_px_per_second",
        "measurement_status",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as summary_file:

        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_fields,
        )
        summary_writer.writeheader()

        for track_id, stats in sorted(track_stats.items()):
            first = stats["first_frame"] / fps
            last = stats["last_frame"] / fps

            rates = stats["pixel_rates"]
            filtered_rates = stats["filtered_pixel_rates"]

            summary_writer.writerow(
                {
                    "track_id": track_id,
                    "observed_frames": stats["observations"],
                    "first_timestamp_seconds": round(first, 3),
                    "last_timestamp_seconds": round(last, 3),
                    "observed_duration_seconds": round(
                        last - first, 3
                    ),
                    "accumulated_pixel_displacement": round(
                        stats["total_pixel_distance"], 3
                    ),
                    "mean_motion_rate_px_per_second": (
                        round(sum(rates) / len(rates), 3)
                        if rates
                        else ""
                    ),
                    "max_motion_rate_px_per_second": (
                        round(max(rates), 3)
                        if rates
                        else ""
                    ),
                    "filtered_accumulated_pixel_displacement": round(
                        stats["filtered_pixel_distance"], 3
                    ),
                    "filtered_mean_motion_rate_px_per_second": (
                        round(
                            sum(filtered_rates) / len(filtered_rates),
                            3,
                        )
                        if filtered_rates
                        else ""
                    ),
                    "filtered_max_motion_rate_px_per_second": (
                        round(max(filtered_rates), 3)
                        if filtered_rates
                        else ""
                    ),
                    "measurement_status": (
                        "UNVALIDATED_IMAGE_PLANE"
                    ),
                }
            )

    elapsed_time = time.perf_counter() - started

    # --------------------------------------------------
    # 5. FINAL REPORT
    # --------------------------------------------------

    print("\n========== MOTION ANALYSIS REPORT ==========")
    print(f"Input video: {VIDEO_PATH}")
    print(f"Frames processed: {frame_number}")
    print(f"Unique track IDs observed: {len(track_stats)}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print(f"Observations CSV: {OBSERVATIONS_CSV}")
    print(f"Track summary CSV: {SUMMARY_CSV}")
    print("Status: SUCCESS")
    print("Speed in km/h: NOT CALCULATED")
    print("Reason: Real-world distance scale is unknown.")
    print("============================================")

    if frame_number == 0:
        raise RuntimeError("No frames were processed.")

    if not OUTPUT_VIDEO.is_file() or OUTPUT_VIDEO.stat().st_size == 0:
        raise RuntimeError("Output video is missing or empty.")


if __name__ == "__main__":
    main()
