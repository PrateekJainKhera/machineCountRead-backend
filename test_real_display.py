"""
Test the OCR pipeline against the REAL Siemens SIMATIC HMI display video
(Flexo label press, GIDOE). Samples frames across the clip, detects all
numbers, and tracks the 'Labels Prod.' counter (the fast-changing one).
"""

import cv2
from vision.ocr_reader import OCRReader

VIDEO = r"C:\Users\hp\Videos\Screen Recordings\Screen Recording 2026-06-11 160334.mp4"
N_SAMPLES = 12
UPSCALE = 2.5  # video is only 586px wide; digits ~14px tall — upscale before OCR


def main():
    reader = OCRReader(gpu=False)
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n{'t(s)':>5} | detected numbers (conf)")
    print("-" * 75)

    labels_prod_track = []

    for i in range(N_SAMPLES):
        fidx = int(i * (total - 5) / (N_SAMPLES - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        big = cv2.resize(frame, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        dets = reader.read_all_counters(big, min_confidence=0.30, min_height=18, max_digits=9)
        t = fidx / fps
        shown = ", ".join(f"{d.value}({d.confidence:.2f})" for d in dets[:8])
        print(f"{t:5.1f} | {shown}")

        # Track the Labels Prod counter (29900–31500 range in this clip)
        for d in dets:
            if 29800 <= d.value <= 31500:
                labels_prod_track.append((t, d.value, d.confidence))
                break

    cap.release()

    print("\n── Labels Prod. counter track ──")
    prev = None
    for t, v, c in labels_prod_track:
        rate = ""
        if prev:
            dt = t - prev[0]
            dv = v - prev[1]
            if dt > 0:
                rate = f"  Δ{dv:+d} in {dt:.1f}s = {dv/dt:.1f}/sec"
        print(f"t={t:5.1f}s  value={v}  conf={c:.2f}{rate}")
        prev = (t, v)


if __name__ == "__main__":
    main()
