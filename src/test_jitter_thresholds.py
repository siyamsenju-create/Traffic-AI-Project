
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "motion_observations.csv"

THRESHOLDS = [2.0, 2.5, 3.0, 4.0, 5.0]
TARGET_TRACKS = {"1", "2"}


def main():
    tracks = defaultdict(list)

    with INPUT_CSV.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["track_id"] not in TARGET_TRACKS:
                continue

            value = row["displacement_px"]

            if value:
                tracks[row["track_id"]].append(float(value))

    print("\n--- Jitter Threshold Experiment ---")
    print("Uses raw displacement; does not modify project outputs.")

    for track_id in sorted(tracks, key=int):
        values = tracks[track_id]

        print(f"\nTrack {track_id}")

        for threshold in THRESHOLDS:
            remaining = [value for value in values if value > threshold]

            print(
                f"Threshold: {threshold:.1f} px | "
                f"Moving intervals: {len(remaining)} | "
                f"Accumulated displacement: {sum(remaining):.2f} px"
            )


if __name__ == "__main__":
    main()
