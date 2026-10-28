
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "outputs" / "calibration_frame.jpg"
POINTS_PATH = ROOT / "outputs" / "road_points.json"
OUTPUT_PATH = ROOT / "outputs" / "road_warp_preview.jpg"


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    data = json.loads(POINTS_PATH.read_text())
    points = np.array(
        data["points_xy_pixels"], dtype=np.float32
    )

    if points.shape != (4, 2):
        raise ValueError("Expected four road points.")

    # These dimensions are for visualization only.
    # They are NOT real-world metres.
    output_width = 800
    output_height = 600

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(points, destination)

    warped = cv2.warpPerspective(
        image, matrix, (output_width, output_height)
    )

    if not cv2.imwrite(str(OUTPUT_PATH), warped):
        raise RuntimeError("Could not save perspective preview.")

    print(f"Perspective preview saved: {OUTPUT_PATH}")
    print("Status: EXPERIMENTAL — dimensions are arbitrary.")
    print("Do not interpret warped coordinates as metres or km/h.")


if __name__ == "__main__":
    main()
