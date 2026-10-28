
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "outputs" / "motion_observations.csv"
OUTPUT_CSV = ROOT / "outputs" / "motion_quality_summary.csv"

MIN_DURATION_SECONDS = 1.0
MAX_GAP_SECONDS = 0.20
WINDOW_SIZE = 5

# A review threshold, not a physical speed limit.
# Values above this rate are flagged for inspection.
REVIEW_RATE_PX_S = 1000.0


def rolling_median(values, window):
    result = []
    half = window // 2

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        result.append(statistics.median(values[start:end]))

    return result


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    tracks = defaultdict(list)

    with INPUT_CSV.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            tracks[row["track_id"]].append({
                "frame": int(row["frame"]),
                "time": float(row["timestamp_seconds"]),
                "x": float(row["bottom_center_x_px"]),
                "y": float(row["bottom_center_y_px"]),
                "vehicle_class": row["vehicle_class"],
            })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "track_id",
        "vehicle_class",
        "observations",
        "duration_seconds",
        "valid_intervals",
        "mean_smoothed_rate_px_s",
        "max_smoothed_rate_px_s",
        "review_flags",
        "measurement_status",
    ]

    kept = 0
    excluded = 0

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for track_id, rows in sorted(
            tracks.items(), key=lambda item: int(item[0])
        ):
            rows.sort(key=lambda r: r["frame"])

            first_time = rows[0]["time"]
            last_time = rows[-1]["time"]
            duration = last_time - first_time

            if duration < MIN_DURATION_SECONDS:
                excluded += 1
                continue

            xs = rolling_median(
                [r["x"] for r in rows], WINDOW_SIZE
            )
            ys = rolling_median(
                [r["y"] for r in rows], WINDOW_SIZE
            )

            rates = []
            flagged = False

            for i in range(1, len(rows)):
                dt = rows[i]["time"] - rows[i - 1]["time"]

                # Do not calculate a rate across a large time gap.
                if dt <= 0 or dt > MAX_GAP_SECONDS:
                    continue

                dx = xs[i] - xs[i - 1]
                dy = ys[i] - ys[i - 1]
                rate = (dx * dx + dy * dy) ** 0.5 / dt

                if rate > REVIEW_RATE_PX_S:
                    flagged = True
                    continue

                rates.append(rate)

            flags = []
            if not rates:
                flags.append("NO_VALID_INTERVALS")
            if flagged:
                flags.append("LARGE_RATE_REQUIRES_REVIEW")

            writer.writerow({
                "track_id": track_id,
                "vehicle_class": rows[0]["vehicle_class"],
                "observations": len(rows),
                "duration_seconds": round(duration, 3),
                "valid_intervals": len(rates),
                "mean_smoothed_rate_px_s": (
                    round(statistics.mean(rates), 3)
                    if rates else ""
                ),
                "max_smoothed_rate_px_s": (
                    round(max(rates), 3) if rates else ""
                ),
                "review_flags": ";".join(flags) or "NONE",
                "measurement_status": "UNVALIDATED_IMAGE_PLANE",
            })

            kept += 1

    print("\n--- Motion Quality Report ---")
    print(f"Tracks found: {len(tracks)}")
    print(f"Tracks retained (>= 1 second): {kept}")
    print(f"Short tracks excluded: {excluded}")
    print(f"Output: {OUTPUT_CSV}")
    print("Status: UNVALIDATED IMAGE-PLANE MOTION")
    print("No real-world speed in km/h was calculated.")


if __name__ == "__main__":
    main()
