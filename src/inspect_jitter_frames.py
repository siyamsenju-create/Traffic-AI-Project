
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "data" / "videos" / "traffic.mp4"
OUTPUT_DIR = ROOT / "outputs" / "jitter_inspection"

TARGET_FRAMES = [80, 83, 84, 85, 88]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    for frame_number in TARGET_FRAMES:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

        if not success:
            print(f"Could not read frame {frame_number}")
            continue

        output_path = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"
        cv2.imwrite(str(output_path), frame)
        print(f"Saved: {output_path}")

    video.release()


if __name__ == "__main__":
    main()
