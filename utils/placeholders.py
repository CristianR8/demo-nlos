from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCENE_SPECS = [
    {
        "id": "scene_001",
        "label": "bunny",
        "choices": ["bunny", "sphere", "chair", "teapot"],
        "difficulty": 2,
        "hint": "tiene orejas",
        "notes": "scene sintetica mitsuba",
    },
    {
        "id": "scene_002",
        "label": "sphere",
        "choices": ["bunny", "sphere", "chair", "teapot"],
        "difficulty": 1,
        "hint": "forma redonda",
        "notes": "scene sintetica mitsuba",
    },
    {
        "id": "scene_003",
        "label": "chair",
        "choices": ["bunny", "sphere", "chair", "teapot"],
        "difficulty": 3,
        "hint": "tiene respaldo",
        "notes": "scene sintetica mitsuba",
    },
    {
        "id": "scene_004",
        "label": "teapot",
        "choices": ["bunny", "sphere", "chair", "teapot"],
        "difficulty": 2,
        "hint": "objeto de cocina",
        "notes": "scene sintetica mitsuba",
    },
]


def ensure_dummy_dataset(data_dir: Path, min_scenes: int = 4) -> None:
    """Create a minimal playable dataset when data is missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in data_dir.glob("scene_*") if p.is_dir()])
    if len(existing) >= min_scenes:
        return

    for spec in SCENE_SPECS[:min_scenes]:
        scene_dir = data_dir / spec["id"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        _create_scene_assets(scene_dir, spec)


def _create_scene_assets(scene_dir: Path, spec: dict) -> None:
    transient = _make_transient_frames(spec["label"], size=(256, 256), n_frames=18)
    recon = _make_reconstruction_frames(spec["label"], size=(256, 256), n_frames=16)
    final_img = _make_final_image(spec["label"], size=(320, 320))

    transient[0].save(
        scene_dir / "transient.gif",
        save_all=True,
        append_images=transient[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    recon[0].save(
        scene_dir / "recon.gif",
        save_all=True,
        append_images=recon[1:],
        duration=100,
        loop=0,
        optimize=False,
    )
    final_img.save(scene_dir / "recon_final.png")

    with (scene_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "label": spec["label"],
                "choices": spec["choices"],
                "difficulty": spec["difficulty"],
                "hint": spec["hint"],
                "notes": spec["notes"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _make_transient_frames(label: str, size: tuple[int, int], n_frames: int) -> list[Image.Image]:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    frames = []

    seed = sum(ord(c) for c in label)
    rng = np.random.default_rng(seed)

    centers = {
        "bunny": [(90, 120), (150, 120), (120, 170)],
        "sphere": [(128, 128)],
        "chair": [(100, 150), (155, 150), (128, 95)],
        "teapot": [(95, 135), (150, 130), (185, 135)],
    }
    active = centers.get(label, [(128, 128)])

    for t in range(n_frames):
        img = np.zeros((h, w), dtype=np.float32)
        for i, (cx, cy) in enumerate(active):
            sigma = 10 + 2 * i
            phase = (t / n_frames) * 2 * np.pi + i
            drift_x = int(8 * np.cos(phase))
            drift_y = int(8 * np.sin(phase))
            g = np.exp(-(((xx - (cx + drift_x)) ** 2 + (yy - (cy + drift_y)) ** 2) / (2 * sigma * sigma)))
            img += g

        img += 0.05 * rng.random((h, w), dtype=np.float32)
        img = img / max(float(img.max()), 1e-6)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(255 * img, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(180 * (img**0.8), 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(80 * img, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(rgb, mode="RGB"))

    return frames


def _make_reconstruction_frames(label: str, size: tuple[int, int], n_frames: int) -> list[Image.Image]:
    w, h = size
    frames = []
    for t in range(n_frames):
        img = Image.new("RGB", size, color=(8, 16, 28))
        draw = ImageDraw.Draw(img)

        alpha = (t + 1) / n_frames
        radius = 22 + int(50 * alpha)
        cx, cy = w // 2, h // 2

        if label == "sphere":
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(120, 220, 255), width=3)
        elif label == "bunny":
            draw.ellipse((cx - radius, cy - radius // 2, cx + radius, cy + radius), outline=(140, 240, 255), width=3)
            ear_h = int(radius * 1.2)
            draw.ellipse((cx - radius, cy - radius - ear_h, cx - radius // 4, cy - radius // 4), outline=(140, 240, 255), width=3)
            draw.ellipse((cx + radius // 4, cy - radius - ear_h, cx + radius, cy - radius // 4), outline=(140, 240, 255), width=3)
        elif label == "chair":
            draw.rectangle((cx - radius, cy, cx + radius, cy + radius), outline=(140, 240, 255), width=3)
            draw.rectangle((cx - radius, cy - radius, cx + radius, cy), outline=(140, 240, 255), width=3)
        elif label == "teapot":
            draw.ellipse((cx - radius, cy - radius // 2, cx + radius, cy + radius // 2), outline=(140, 240, 255), width=3)
            draw.arc((cx + radius - 8, cy - radius // 3, cx + radius + 30, cy + radius // 3), start=30, end=330, fill=(140, 240, 255), width=3)
            draw.polygon([(cx - 6, cy - radius // 2 - 6), (cx + 6, cy - radius // 2 - 6), (cx, cy - radius // 2 - 20)], outline=(140, 240, 255), fill=None)

        txt = f"slice {t + 1:02d}/{n_frames:02d}"
        draw.text((12, 12), txt, fill=(180, 220, 245))
        frames.append(img)

    return frames


def _make_final_image(label: str, size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, color=(20, 26, 38))
    draw = ImageDraw.Draw(img)
    w, h = size
    cx, cy = w // 2, h // 2

    if label == "sphere":
        draw.ellipse((cx - 78, cy - 78, cx + 78, cy + 78), fill=(100, 160, 210), outline=(220, 240, 255), width=3)
    elif label == "bunny":
        draw.ellipse((cx - 80, cy - 35, cx + 80, cy + 85), fill=(110, 170, 220), outline=(220, 240, 255), width=3)
        draw.ellipse((cx - 75, cy - 130, cx - 20, cy - 10), fill=(110, 170, 220), outline=(220, 240, 255), width=3)
        draw.ellipse((cx + 20, cy - 130, cx + 75, cy - 10), fill=(110, 170, 220), outline=(220, 240, 255), width=3)
    elif label == "chair":
        draw.rectangle((cx - 85, cy + 10, cx + 85, cy + 95), fill=(105, 165, 215), outline=(220, 240, 255), width=3)
        draw.rectangle((cx - 85, cy - 95, cx + 85, cy + 10), fill=(105, 165, 215), outline=(220, 240, 255), width=3)
    elif label == "teapot":
        draw.ellipse((cx - 95, cy - 45, cx + 95, cy + 45), fill=(105, 165, 215), outline=(220, 240, 255), width=3)
        draw.arc((cx + 85, cy - 40, cx + 145, cy + 40), start=40, end=320, fill=(220, 240, 255), width=5)
        draw.polygon([(cx - 8, cy - 45), (cx + 8, cy - 45), (cx, cy - 72)], fill=(220, 240, 255))

    draw.text((12, h - 24), f"Final reconstruction: {label}", fill=(220, 240, 255))
    return img
