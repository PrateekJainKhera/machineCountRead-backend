"""
Generates a synthetic machine-panel test video for end-to-end testing.

The video simulates exactly what the factory camera will see:
  - A counter incrementing ~2/sec (like a running printing machine)
  - A job card with an alphanumeric number placed below the counter
  - The job card is REMOVED for the last few seconds (tests removal detection)

Output: test_machine.mp4 (in this folder)
Prints the exact ROIs to use when registering the camera.
"""

import cv2
import numpy as np

W, H = 960, 540
FPS = 10
DURATION_S = 40
COUNTER_START = 12340
COUNTS_PER_SEC = 2          # counter speed
JOB_NUMBER = "JC-4521"
CARD_REMOVED_AT_S = 25      # job card disappears here (job finished)

# Panel layout
COUNTER_ROI = (280, 120, 400, 110)   # x, y, w, h — where the counter digits sit
JOBCARD_ROI = (280, 300, 400, 120)   # the magnet slot below the counter


def draw_frame(t: float) -> np.ndarray:
    frame = np.full((H, W, 3), 40, dtype=np.uint8)  # dark machine panel

    # Panel bezel
    cv2.rectangle(frame, (200, 60), (760, 480), (70, 70, 70), -1)
    cv2.putText(frame, "MACHINE PANEL - TEST", (340, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    # ── Counter display (green digits on black, LED-style) ──────────
    x, y, w, h = COUNTER_ROI
    cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 10, 10), -1)
    counter = COUNTER_START + int(t * COUNTS_PER_SEC)
    cv2.putText(frame, str(counter), (x + 40, y + 80),
                cv2.FONT_HERSHEY_DUPLEX, 2.4, (0, 255, 0), 5)

    # ── Job card slot (white card, big black job number) ────────────
    x, y, w, h = JOBCARD_ROI
    if t < CARD_REMOVED_AT_S:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (235, 235, 235), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 150), 2)
        cv2.putText(frame, JOB_NUMBER, (x + 60, y + 75),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (20, 20, 20), 4)
    else:
        # Card removed — empty slot (job finished)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)

    return frame


def main():
    out_path = "test_machine.mp4"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)
    )
    total_frames = DURATION_S * FPS
    for i in range(total_frames):
        writer.write(draw_frame(i / FPS))
    writer.release()

    print(f"Wrote {out_path} ({DURATION_S}s @ {FPS}fps)")
    print(f"Counter: starts {COUNTER_START}, +{COUNTS_PER_SEC}/sec")
    print(f"Job card '{JOB_NUMBER}' visible until t={CARD_REMOVED_AT_S}s, then removed")
    print(f"Counter ROI:  x={COUNTER_ROI[0]} y={COUNTER_ROI[1]} w={COUNTER_ROI[2]} h={COUNTER_ROI[3]}")
    print(f"Job card ROI: x={JOBCARD_ROI[0]} y={JOBCARD_ROI[1]} w={JOBCARD_ROI[2]} h={JOBCARD_ROI[3]}")


if __name__ == "__main__":
    main()
