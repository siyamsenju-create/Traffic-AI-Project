
import json
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "outputs" / "calibration_frame.jpg"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
POINTS_PATH = OUTPUT_DIR / "road_points.json"
ANNOTATED_PATH = OUTPUT_DIR / "road_points_preview.jpg"

points = []
display_image = None


def handle_mouse(event, x, y, flags, param):
    global points, display_image

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([int(x), int(y)])
        print(f"Point {len(points)}: x={x}, y={y}")


def main():
    global display_image

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Calibration image not found: {IMAGE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError("Could not read calibration image.")

    print("Click FOUR points around a road region.")
    print("Choose points in order around the region.")
    print("Press R to reset, ENTER to save, or ESC to cancel.")

    cv2.namedWindow("Select road region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select road region", 1200, 675)
    cv2.setMouseCallback("Select road region", handle_mouse)

    while True:
        display_image = image.copy()

        for i, (x, y) in enumerate(points):
            cv2.circle(display_image, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(
                display_image,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(
                    display_image,
                    tuple(points[i]),
                    tuple(points[i + 1]),
                    (0, 255, 255),
                    2,
                )

        if len(points) == 4:
            cv2.line(
                display_image,
                tuple(points[-1]),
                tuple(points[0]),
                (0, 255, 255),
                2,
            )

        cv2.imshow("Select road region", display_image)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # Escape
            print("Selection cancelled.")
            break

        if key in (ord("r"), ord("R")):
            points.clear()
            print("Points reset.")

        if key in (10, 13) and len(points) == 4:
            data = {
                "image": str(IMAGE_PATH.name),
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "points_xy_pixels": points,
                "real_world_coordinates_available": False,
                "calibration_status": "UNVALIDATED",
            }

            POINTS_PATH.write_text(json.dumps(data, indent=2))
            cv2.imwrite(str(ANNOTATED_PATH), display_image)

            print(f"Saved points: {POINTS_PATH}")
            print(f"Saved preview: {ANNOTATED_PATH}")
            print("WARNING: These points do not establish metres or km/h.")
            break

        if key in (10, 13) and len(points) != 4:
            print("Please select exactly four points before saving.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
