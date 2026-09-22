from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from utils.io import (
    PREPROCESSED_DIRNAME,
    _load_h5_front_rgb_volume,
    render_inferno_image,
)


BASE_DIR = Path(__file__).parent
DEFAULT_SCENES_DIR = BASE_DIR / "scenes"
DEFAULT_DATA_DIR = BASE_DIR / "data"
GIF_DURATIONS_MS = {
    "facil": 100,
    "medio": 40,
    "dificil": 20,
}
GIF_FRAME_STEPS = {
    "facil": 1,
    "medio": 1,
    "dificil": 2,
}


def preprocess_scene(
    h5_path: Path,
    target_frames: int,
    force: bool,
    output_root: Path | None = None,
) -> None:
    output_dir = (output_root or h5_path.parent) / PREPROCESSED_DIRNAME / h5_path.stem
    gif_paths = {
        "facil": output_dir / "transient_facil.gif",
        "medio": output_dir / "transient.gif",
        "dificil": output_dir / "transient_dificil.gif",
    }
    integrated_path = output_dir / "integrated.png"
    integrated_rgb_path = output_dir / "integrated_rgb.png"

    expected = [*gif_paths.values(), integrated_path, integrated_rgb_path]
    if not force and all(path.exists() for path in expected):
        print(f"[skip] {h5_path.name}: artefactos existentes")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] {h5_path.name}")
    rgb_volume = _load_h5_front_rgb_volume(h5_path, target_frames=target_frames)
    volume = rgb_volume.mean(axis=-1, dtype=np.float32)

    lo = float(np.percentile(volume, 1.0))
    hi = float(np.percentile(volume, 99.5))
    if hi <= lo:
        hi = lo + 1e-6

    print(f"[gif]  {h5_path.stem} ({volume.shape[2]} frames, 3 velocidades)")
    frames = [
        render_inferno_image(volume[:, :, i], lo, hi).transpose(Image.Transpose.ROTATE_180)
        for i in range(volume.shape[2])
    ]
    for level, gif_path in gif_paths.items():
        level_frames = frames[::GIF_FRAME_STEPS[level]]
        level_frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=level_frames[1:],
            duration=GIF_DURATIONS_MS[level],
            loop=0,
            optimize=False,
        )

    integrated = volume.sum(axis=2)
    int_lo = float(np.percentile(integrated, 1.0))
    int_hi = float(np.percentile(integrated, 99.5))
    print(f"[png]  {integrated_path}")
    render_inferno_image(integrated, int_lo, int_hi).transpose(
        Image.Transpose.ROTATE_180
    ).save(integrated_path, format="PNG")

    integrated_rgb = rgb_volume.sum(axis=2)
    rgb_lo = float(np.percentile(integrated_rgb, 1.0))
    rgb_hi = float(np.percentile(integrated_rgb, 99.5))
    if rgb_hi <= rgb_lo:
        rgb_hi = rgb_lo + 1e-6
    integrated_rgb = np.clip((integrated_rgb - rgb_lo) / (rgb_hi - rgb_lo), 0.0, 1.0)
    rgb_image = Image.fromarray((integrated_rgb * 255.0).astype(np.uint8), mode="RGB")
    print(f"[png]  {integrated_rgb_path}")
    rgb_image.transpose(Image.Transpose.ROTATE_180).save(integrated_rgb_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesa escenas HDF5 para que Streamlit solo lea GIF/PNG."
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=None,
        help="Carpeta HDF5. Sin esta opción se revisan ./scenes y ./data.",
    )
    parser.add_argument("--target-frames", type=int, default=300)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Raíz donde se crea preprocessed/. Por defecto se usa la carpeta de cada HDF5.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    roots = [args.scenes_dir] if args.scenes_dir else [DEFAULT_SCENES_DIR, DEFAULT_DATA_DIR]
    paths = sorted(path for root in roots for path in root.glob("*.h5"))
    if not paths:
        raise SystemExit(f"No se encontraron HDF5 en: {', '.join(str(root) for root in roots)}")
    if args.target_frames < 1:
        raise SystemExit("--target-frames debe ser mayor que cero")

    for path in paths:
        preprocess_scene(
            path,
            target_frames=args.target_frames,
            force=args.force,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
