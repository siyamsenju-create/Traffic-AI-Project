
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "outputs" / "calibration_frame.jpg"
POINTS_PATH = ROOT / "outputs" / "road_points.json"
OUTPUT_PATH = ROOT / "outputs" / "road_points_preview.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    data = json.loads(POINTS_PATH.read_text())
    points = np.array(data["points_xy_pixels"], dtype=np.int32)

    if points.shape != (4, 2):
        raise ValueError("Expected exactly four (x, y) points.")

    preview = image.copy()
    cv2.polylines(
        preview, [points.reshape(-1, 1, 2)],
        isClosed=True, color=(0, 255, 255), thickness=4
    )

    for i, (x, y) in enumerate(points, start=1):
        cv2.circle(preview, (int(x), int(y)), 9, (0, 255, 0), -1)
        cv2.putText(
            preview, str(i), (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
        )

    if not cv2.imwrite(str(OUTPUT_PATH), preview):
        raise RuntimeError("Could not save preview.")

    print(f"Preview saved: {OUTPUT_PATH}")
    print("Calibration status: UNVALIDATED")


if __name__ == "__main__":
    main()
