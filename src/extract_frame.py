from pathlib import Path
import cv2

project = Path(__file__).resolve().parents[1]
video_path = project / "data" / "videos" / "traffic.mp4"
output_path = project / "outputs" / "calibration_frame.jpg"

cap = cv2.VideoCapture(str(video_path))


if not cap.isOpened():
    raise RuntimeError(f"Could open video: {video_path}")

# Extract a frame at 5 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not extract frame.")

output_path.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(output_path), frame):
    raise RuntimeError("Could not save the calibration image.")

print(f"Calibration image saved: {output_path}")
print(f"Image dimensions: {frame.shape[1]} x {frame.shape[0]}")
