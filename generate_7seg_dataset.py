"""
Synthetic 7-segment LED display dataset generator for YOLOv8 digit detection.

Renders DRC-161-style counter images (red/green/amber 7-seg digits, ghost
segments, glow, glare, blur, rotation, label-text distractors) with exact
auto-generated YOLO bounding-box labels — no manual annotation needed.

Output: datasets/seg7/{train,valid}/{images,labels} + data.yaml

Run:  venv/Scripts/python.exe generate_7seg_dataset.py
"""

import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "DSEG7Classic-Bold.ttf"
OUT_DIR = "datasets/seg7"
TRAIN_N = 2200
VALID_N = 250
CANVAS = 416

# LED colors: (lit RGB, weight)
COLORS = [
    ((235, 40, 45), 0.55),   # red (DRC-161)
    ((60, 230, 90), 0.20),   # green
    ((255, 170, 40), 0.15),  # amber
    ((230, 230, 225), 0.10), # white/blue-ish LCD backlit
]

DISTRACTOR_WORDS = ["DIGITAL COUNTER", "DRC 161", "MCS", "COUNTER", "TOTAL",
                    "RESET", "MODEL XJ-2000", "SPH", "PRESET", "BATCH"]


def pick_color():
    r = random.random()
    acc = 0.0
    for c, w in COLORS:
        acc += w
        if r <= acc:
            return c
    return COLORS[0][0]


def render_sample(rng: random.Random):
    """Render one synthetic display image. Returns (bgr_image, yolo_labels)."""
    # ── background: dark display window with slight hue variation ──
    base = rng.randint(8, 35)
    bg = (base + rng.randint(0, 12), max(4, base - rng.randint(0, 6)), max(4, base - rng.randint(0, 6)))
    img = Image.new("RGB", (CANVAS, CANVAS), bg)
    draw = ImageDraw.Draw(img)

    # ── digit string ──
    n_digits = rng.choices([1, 2, 3, 4, 5, 6], weights=[4, 6, 12, 22, 28, 28])[0]
    value = "".join(rng.choice("0123456789") for _ in range(n_digits))

    font_size = rng.randint(55, 150)
    font = ImageFont.truetype(FONT_PATH, font_size)
    adv = draw.textlength("8", font=font)
    total_w = adv * n_digits
    if total_w > CANVAS - 30:
        font_size = int(font_size * (CANVAS - 30) / total_w)
        font = ImageFont.truetype(FONT_PATH, font_size)
        adv = draw.textlength("8", font=font)
        total_w = adv * n_digits

    bb8 = font.getbbox("8")
    text_h = bb8[3] - bb8[1]
    x0 = rng.uniform(10, CANVAS - total_w - 10)
    y0 = rng.uniform(10 - bb8[1], CANVAS - text_h - 10 - bb8[1])

    color = pick_color()

    # ── ghost (unlit) segments ──
    if rng.random() < 0.65:
        gf = rng.uniform(0.12, 0.32)
        ghost = tuple(int(bg[i] + (color[i] - bg[i]) * gf) for i in range(3))
        draw.text((x0, y0), "8" * n_digits, font=font, fill=ghost)

    # ── lit digits + per-digit bboxes ──
    labels = []
    for i, ch in enumerate(value):
        cx = x0 + i * adv
        draw.text((cx, y0), ch, font=font, fill=color)
        b = font.getbbox(ch)
        bx0, by0 = cx + b[0], y0 + b[1]
        bx1, by1 = cx + b[2], y0 + b[3]
        labels.append((int(ch), bx0, by0, bx1, by1))

    # ── distractor label text (printed, not glowing) — teaches the model to
    #    ignore the bezel text like "DIGITAL COUNTER DRC 161" ──
    if rng.random() < 0.40:
        word = rng.choice(DISTRACTOR_WORDS)
        dfont_size = rng.randint(14, 30)
        try:
            dfont = ImageFont.truetype("arial.ttf", dfont_size)
        except OSError:
            dfont = ImageFont.load_default()
        dcol = (rng.randint(140, 230), rng.randint(130, 215), rng.randint(110, 190))
        dy = rng.choice([rng.uniform(2, max(3, y0 + bb8[1] - 35)),
                         rng.uniform(min(CANVAS - 35, y0 + bb8[3] + 8), CANVAS - 20)])
        draw.text((rng.uniform(5, 150), dy), word, font=dfont, fill=dcol)

    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ── LED glow bloom ──
    glow = cv2.GaussianBlur(frame, (0, 0), rng.uniform(2.0, 4.5))
    frame = cv2.addWeighted(frame, 0.8, glow, rng.uniform(0.3, 0.6), 0)

    # ── rotation (small mounting tilt) + bbox transform ──
    angle = rng.uniform(-6, 6)
    M = cv2.getRotationMatrix2D((CANVAS / 2, CANVAS / 2), angle, 1.0)
    frame = cv2.warpAffine(frame, M, (CANVAS, CANVAS), borderValue=bg[::-1])
    new_labels = []
    for cls, bx0, by0, bx1, by1 in labels:
        corners = np.array([[bx0, by0, 1], [bx1, by0, 1], [bx0, by1, 1], [bx1, by1, 1]]).T
        t = M @ corners
        nx0, ny0 = t[0].min(), t[1].min()
        nx1, ny1 = t[0].max(), t[1].max()
        nx0, ny0 = max(0, nx0), max(0, ny0)
        nx1, ny1 = min(CANVAS, nx1), min(CANVAS, ny1)
        if nx1 - nx0 > 3 and ny1 - ny0 > 3:
            new_labels.append((cls, nx0, ny0, nx1, ny1))
    labels = new_labels

    # ── glare streak (tinted window reflection) ──
    if rng.random() < 0.5:
        overlay = np.zeros_like(frame, dtype=np.float32)
        gx = rng.randint(0, CANVAS)
        cv2.ellipse(overlay, (gx, rng.randint(0, CANVAS // 2)),
                    (rng.randint(60, 180), rng.randint(20, 60)),
                    rng.uniform(-40, 40), 0, 360, (255, 255, 255), -1)
        overlay = cv2.GaussianBlur(overlay, (0, 0), 25)
        frame = cv2.addWeighted(frame, 1.0, overlay.astype(np.uint8), rng.uniform(0.05, 0.18), 0)

    # ── camera blur (the "Indian factory" condition) ──
    blur = rng.uniform(0, 2.2)
    if blur > 0.2:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)

    # ── sensor noise + exposure jitter ──
    noise = np.random.normal(0, rng.uniform(2, 9), frame.shape)
    frame = np.clip(frame.astype(np.float32) * rng.uniform(0.75, 1.2) + noise, 0, 255).astype(np.uint8)

    # YOLO label format: class cx cy w h (normalized)
    yolo = []
    for cls, bx0, by0, bx1, by1 in labels:
        cx = (bx0 + bx1) / 2 / CANVAS
        cy = (by0 + by1) / 2 / CANVAS
        w = (bx1 - bx0) / CANVAS
        h = (by1 - by0) / CANVAS
        yolo.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return frame, yolo


def generate(split: str, n: int, seed: int):
    rng = random.Random(seed)
    img_dir = os.path.join(OUT_DIR, split, "images")
    lbl_dir = os.path.join(OUT_DIR, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for i in range(n):
        frame, yolo = render_sample(rng)
        name = f"{split}_{i:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), frame)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(yolo))
        if (i + 1) % 500 == 0:
            print(f"{split}: {i + 1}/{n}")


def main():
    generate("train", TRAIN_N, seed=42)
    generate("valid", VALID_N, seed=1337)
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(
            "train: train/images\n"
            "val: valid/images\n\n"
            "nc: 10\n"
            "names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']\n"
        )
    print(f"Done. Dataset at {OUT_DIR}/")


if __name__ == "__main__":
    main()
