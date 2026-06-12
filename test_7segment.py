"""
Empirical test: can the OCR pipeline read 7-segment LED counter digits
(DRC-161 style: red glowing digits + faint ghost segments on dark window)?

Renders synthetic test images with the DSEG7 font and runs them through
OCRReader.read_counter() in both preprocessing modes.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from vision.ocr_reader import OCRReader

FONT_PATH = "DSEG7Classic-Bold.ttf"
TEST_VALUES = ["12565", "18917", "007403", "999999", "120406"]


def render_7seg(value: str, ghost: bool = True, blur: float = 0.0) -> np.ndarray:
    """Render a DRC-161 style counter: red 7-seg digits on dark maroon window."""
    W, H = 640, 200
    img = Image.new("RGB", (W, H), (28, 8, 10))  # dark tinted window
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 110)

    # DSEG renders '8' for every digit position — use as ghost segments
    pad = value.rjust(6)
    if ghost:
        draw.text((40, 45), "888888", font=font, fill=(70, 18, 20))  # faint unlit segments
    draw.text((40, 45), pad.replace(" ", "!"), font=font, fill=(235, 40, 45))  # lit digits
    # ('!' renders blank in DSEG — keeps spacing for leading blanks)

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # mild glow bloom like a real LED
    glow = cv2.GaussianBlur(frame, (0, 0), 3)
    frame = cv2.addWeighted(frame, 0.8, glow, 0.45, 0)

    if blur > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return frame


def main():
    import time
    from vision.yolo_digit_reader import YOLODigitReader

    readers = {
        "easyocr": OCRReader(gpu=False),
        "yolo": YOLODigitReader("vision/digit_model.pt"),
    }
    print(f"\n{'value':>10} | {'reader':>8} | {'read':>10} | {'conf':>5} | {'ms':>6} | ok?")
    print("-" * 62)
    for val in TEST_VALUES:
        frame = render_7seg(val)
        cv2.imwrite(f"seg_test_{val}.png", frame)
        for name, reader in readers.items():
            t0 = time.perf_counter()
            r = reader.read_counter(frame, roi=None, sharpness_threshold=10.0,
                                    preprocess_mode="led")
            ms = (time.perf_counter() - t0) * 1000
            expected = int(val)
            got = r.value
            ok = "YES" if got == expected else ("part" if got and str(got) in val else "NO")
            print(f"{val:>10} | {name:>8} | {str(got):>10} | {r.confidence:.2f}  | {ms:6.0f} | {ok}")


if __name__ == "__main__":
    main()
