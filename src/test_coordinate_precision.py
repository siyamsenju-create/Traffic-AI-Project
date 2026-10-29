
import cv2
import math
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "videos" / "traffic.mp4"

MODEL_NAME = "yolo11n.pt"
TARGET_FRAMES = {83, 84, 85}
TARGET_TRACK_ID = 2


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    model = YOLO(MODEL_NAME)
    frame_number = 0
    previous_float = None
    previous_int = None

    try:
        while frame_number <= max(TARGET_FRAMES):
            success, frame = cap.read()

            if not success:
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[2, 3, 5, 7],
                conf=0.30,
                imgsz=640,
                device="cpu",
                agnostic_nms=True,
                verbose=False,
            )[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().tolist()
                ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):
                    if track_id != TARGET_TRACK_ID:
                        continue

                    x1, y1, x2, y2 = box
                    current_float = ((x1 + x2) / 2.0, y2)
                    current_int = (
                        int((x1 + x2) / 2),
                        int(y2),
                    )
                    current_box = (x1, y1, x2, y2)

                    if frame_number in TARGET_FRAMES:
                        float_displacement = ""
                        int_displacement = ""

                        if previous_float is not None:
                            float_displacement = round(
                                math.dist(current_float, previous_float), 3
                            )

                        if previous_int is not None:
                            int_displacement = round(
                                math.dist(current_int, previous_int), 3
                            )

                        print(
                            f"Frame {frame_number}\n"
                            f"  Bounding box: {current_box}\n"
                            f"  Width: {x2 - x1:.2f} px\n"
                            f"  Height: {y2 - y1:.2f} px\n"
                            f"  Float position: {current_float}\n"
                            f"  Integer position: {current_int}\n"
                            f"  Float displacement: {float_displacement}\n"
                            f"  Integer displacement: {int_displacement}\n"
                        )

                    if frame_number == 84:
                        annotated_frame = frame.copy()

                        cv2.rectangle(
                            annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            3,
                        )

                        cv2.circle(
                            annotated_frame,
                            (
                                int(current_float[0]),
                                int(current_float[1]),
                            ),
                            8,
                            (0, 0, 255),
                            -1,
                        )

                        output_path = (
                            ROOT
                            / "outputs"
                            / "jitter_inspection"
                            / "track_2_frame_84.jpg"
                        )
                        output_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        cv2.imwrite(str(output_path), annotated_frame)
                        print(f"Saved diagnostic image: {output_path}")

                    previous_float = current_float
                    previous_int = current_int

            frame_number += 1

    finally:
        cap.release()


if __name__ == "__main__":
    main()
