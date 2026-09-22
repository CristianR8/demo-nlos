from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from scipy.io import loadmat

try:
    import h5py
except ImportError:  # pragma: no cover - reported when an HDF5 scene is opened
    h5py = None

try:
    from matplotlib import colormaps
except Exception:  # pragma: no cover - fallback for minimal environments
    colormaps = None


REQUIRED_FILES = ["transient.gif", "recon.gif", "meta.json"]
FRONT_VIEW_INDEX = 24
H5_LABELS = {
    "scene_1": "Rana",
    "scene_10": "Venado",
    "scene_13": "Botella",
    "scene_24": "León",
    "scene_28": "Bailarina",
    "scene_30": "Carro",
    "scene_31": "Logo del Semillero",
    "scene_34": "Mario Bros",
}
EXCLUDED_PREPROCESSED_SCENES = {"scene_23"}
PREPROCESSED_DIRNAME = "preprocessed"


def list_scene_dirs(data_dir: Path) -> list[Path]:
    scenes = []
    for scene_dir in sorted(data_dir.glob("scene_*")):
        if not scene_dir.is_dir():
            continue
        if all((scene_dir / file_name).exists() for file_name in REQUIRED_FILES):
            scenes.append(scene_dir)
    return scenes


def load_meta(scene_dir: Path) -> dict:
    with (scene_dir / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_game_scenes(data_dir: Path) -> list[dict]:
    """Discover playable scenes from precomputed artifacts, then fallback formats."""
    preprocessed_scenes = [
        *_discover_preprocessed_dataset(data_dir.parent / "scenes"),
        *_discover_preprocessed_dataset(data_dir),
    ]
    if preprocessed_scenes:
        choices = [scene["label"] for scene in preprocessed_scenes]
        for scene in preprocessed_scenes:
            scene["choices"] = choices
        return preprocessed_scenes

    real_scenes = _discover_real_dataset(data_dir / "scenes")
    if real_scenes:
        return real_scenes

    scenes = []
    for scene_dir in list_scene_dirs(data_dir):
        meta = load_meta(scene_dir)
        scenes.append(
            {
                "id": scene_dir.name,
                "scene_dir": str(scene_dir),
                "transient_path": str(scene_dir / "transient.gif"),
                "transient_type": "gif",
                "label": meta["label"],
                "choices": meta["choices"],
                "difficulty": int(meta.get("difficulty", 1)),
                "hint": meta.get("hint", "Sin pista"),
                "notes": meta.get("notes", ""),
                "recon_gif": str(scene_dir / "recon.gif") if (scene_dir / "recon.gif").exists() else None,
                "recon_final": _first_existing(
                    scene_dir / "recon_final.png",
                    scene_dir / "depth.png",
                ),
                "render_3d": str(scene_dir / "volume.png") if (scene_dir / "volume.png").exists() else None,
            }
        )
    return scenes


def get_transient_gif_bytes(
    transient_path: Path,
    transient_type: str = "gif",
    only_three_frames: bool = False,
    noise_level: float = 0.0,
    upscale: int = 1,
) -> bytes:
    """Return transient GIF bytes optionally reduced/noisy for gameplay."""
    if transient_type == "h5":
        vol = _load_h5_front_volume(transient_path, target_frames=300)
        frames = _volume_to_frames(vol)
    elif transient_type == "mat":
        frames = _load_mat_transient_frames(transient_path, target_frames=300)
    else:
        with Image.open(transient_path) as img:
            frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]

    if only_three_frames and len(frames) > 3:
        idxs = np.linspace(0, len(frames) - 1, 3, dtype=int)
        frames = [frames[i] for i in idxs]

    if noise_level > 0:
        noisy = []
        for frame in frames:
            arr = np.array(frame).astype(np.float32)
            sigma = noise_level * 255.0
            arr += np.random.normal(0.0, sigma, size=arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            noisy.append(Image.fromarray(arr, mode="RGB"))
        frames = noisy

    if upscale > 1:
        enlarged = []
        for frame in frames:
            w, h = frame.size
            enlarged.append(frame.resize((w * upscale, h * upscale), resample=Image.Resampling.NEAREST))
        frames = enlarged

    # Keep playback around ~10 seconds so long transients behave like a video sweep.
    duration_ms = int(np.clip(10000 / max(len(frames), 1), 20, 100))

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buf.getvalue()


def get_integrated_image(
    transient_path: Path,
    transient_type: str = "gif",
    noise_level: float = 0.0,
) -> Image.Image:
    """Average all transient frames to produce a single integrated measurement."""
    if transient_type == "h5":
        vol = _load_h5_front_volume(transient_path, target_frames=None)
        gray = vol.sum(axis=2)
        avg = _gray_to_rgb(_normalize_scalar_image(gray)).astype(np.float32)
    elif transient_type == "mat":
        vol = _load_mat_volume(transient_path)
        gray = vol.mean(axis=2)
        avg = _gray_to_rgb(gray).astype(np.float32)
    else:
        with Image.open(transient_path) as img:
            frames = [np.array(frame.convert("RGB"), dtype=np.float32) for frame in ImageSequence.Iterator(img)]
        avg = np.mean(np.stack(frames, axis=0), axis=0)

    avg = np.array(avg, dtype=np.float32)

    if noise_level > 0:
        sigma = noise_level * 255.0
        avg += np.random.normal(0.0, sigma, size=avg.shape)

    avg = np.clip(avg, 0, 255).astype(np.uint8)
    return Image.fromarray(avg, mode="RGB")


def get_transient_volume_for_ui(
    transient_path: Path,
    transient_type: str = "gif",
    target_frames: int = 300,
    log_scale: bool = False,
) -> dict:
    """Build a lightweight 3D transient volume and robust display ranges for UI controls."""
    if transient_type == "h5":
        vol = _load_h5_front_volume(transient_path, target_frames=target_frames)
    elif transient_type == "mat":
        vol = _load_mat_volume(transient_path)
    else:
        with Image.open(transient_path) as img:
            frames = [np.array(frame.convert("RGB"), dtype=np.float32) for frame in ImageSequence.Iterator(img)]
        if not frames:
            raise RuntimeError(f"{transient_path.name}: no contiene frames")
        gray_frames = [f.mean(axis=2) / 255.0 for f in frames]
        vol = np.stack(gray_frames, axis=2).astype(np.float32)

    t = vol.shape[2]
    if t > target_frames:
        edges = np.linspace(0, t, target_frames + 1, dtype=int)
        bins = []
        for i in range(target_frames):
            a, b = int(edges[i]), int(edges[i + 1])
            if b <= a:
                b = min(a + 1, t)
            bins.append(vol[:, :, a:b].mean(axis=2))
        vol = np.stack(bins, axis=2)

    # Keep raw data for physically meaningful integrated image.
    vol_raw = np.array(vol, dtype=np.float32)

    if log_scale:
        vol = np.log1p(np.clip(vol_raw, 0.0, None))
    else:
        vol = vol_raw

    lo = float(np.percentile(vol, 1.0))
    hi = float(np.percentile(vol, 99.5))
    if hi <= lo:
        hi = lo + 1e-6

    integrated = vol_raw.sum(axis=2)
    if log_scale:
        integrated = np.log1p(np.clip(integrated, 0.0, None))
    int_lo = float(np.percentile(integrated, 1.0))
    int_hi = float(np.percentile(integrated, 99.5))
    if int_hi <= int_lo:
        int_hi = int_lo + 1e-6

    return {
        "volume": vol.astype(np.float32),
        "lo": lo,
        "hi": hi,
        "integrated": integrated.astype(np.float32),
        "integrated_lo": int_lo,
        "integrated_hi": int_hi,
    }


def render_inferno_image(gray: np.ndarray, lo: float, hi: float) -> Image.Image:
    """Render a 2D scalar image with inferno colormap and robust clipping."""
    if hi <= lo:
        hi = lo + 1e-6
    x = np.array(gray, dtype=np.float32)
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if colormaps is not None:
        rgb = (colormaps["inferno"](x)[..., :3] * 255.0).astype(np.uint8)
    else:
        rgb = _gray_to_rgb(x).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def resolve_reconstruction_outputs(scene_dir: Path) -> dict:
    recon_gif = scene_dir / "recon.gif"
    recon_final = scene_dir / "recon_final.png"
    depth = scene_dir / "depth.png"
    volume = scene_dir / "volume.png"

    return {
        "recon_gif": str(recon_gif) if recon_gif.exists() else None,
        "recon_final": str(recon_final) if recon_final.exists() else (str(depth) if depth.exists() else None),
        "render_3d": str(volume) if volume.exists() else None,
        "mode": "demo",
    }


def _discover_preprocessed_dataset(root: Path) -> list[dict]:
    """Expose precomputed scenes without requiring their source HDF5 files."""
    preprocessed_root = root / PREPROCESSED_DIRNAME
    artifact_dirs = (
        sorted(
            (path for path in preprocessed_root.glob("scene_*") if path.is_dir()),
            key=lambda path: _natural_sort_key(path.name),
        )
        if preprocessed_root.exists()
        else []
    )
    if not artifact_dirs:
        return []

    # Merely existing is not enough: an interrupted/invalid preprocessing run can
    # leave valid image files whose pixels are all black (scene_12 was one such
    # case).  Do not offer those scenes to the player because both the transient
    # and the reconstruction would appear empty.
    playable_dirs = []
    for artifact_dir in artifact_dirs:
        if artifact_dir.name in EXCLUDED_PREPROCESSED_SCENES:
            continue
        transient_path = artifact_dir / "transient.gif"
        integrated_path = artifact_dir / "integrated.png"
        integrated_rgb_path = artifact_dir / "integrated_rgb.png"
        if not transient_path.exists():
            continue
        if not any(
            _image_has_visible_signal(path)
            for path in (integrated_path, integrated_rgb_path)
            if path.exists()
        ):
            continue
        playable_dirs.append(artifact_dir)

    labels = [H5_LABELS.get(path.name, path.name.replace("_", " ")) for path in playable_dirs]
    scenes = []
    for artifact_dir, label in zip(playable_dirs, labels):
        transient_gifs = {
            "Fácil": artifact_dir / "transient_facil.gif",
            "Medio": artifact_dir / "transient.gif",
            "Difícil": artifact_dir / "transient_dificil.gif",
        }
        transient_gif = transient_gifs["Medio"]
        integrated_image = artifact_dir / "integrated.png"
        integrated_rgb_image = artifact_dir / "integrated_rgb.png"
        if not transient_gif.exists():
            # A scene is playable only when its main precomputed animation exists.
            continue
        scenes.append(
            {
                "id": artifact_dir.name,
                "scene_dir": str(artifact_dir),
                "transient_path": str(transient_gif),
                "transient_gifs": {
                    level: str(gif_path)
                    for level, gif_path in transient_gifs.items()
                    if gif_path.exists()
                },
                "transient_type": "precomputed_gif",
                "view_index": FRONT_VIEW_INDEX,
                "label": label,
                "choices": labels,
                "difficulty": 3,
                "hint": "observa la distribución temporal de la energía",
                "notes": f"Medición HDF5, vista frontal {FRONT_VIEW_INDEX}.",
                "recon_gif": None,
                "recon_final": None,
                "render_3d": None,
                "integrated_image": str(integrated_image) if integrated_image.exists() else None,
                "integrated_rgb_image": str(integrated_rgb_image) if integrated_rgb_image.exists() else None,
            }
        )
    return scenes


def _image_has_visible_signal(path: Path) -> bool:
    """Return whether a precomputed still contains non-uniform visible data."""
    try:
        with Image.open(path) as image:
            extrema = image.convert("RGB").getextrema()
    except (OSError, ValueError):
        return False

    return any(high > low for low, high in extrema) and any(high > 0 for _, high in extrema)


def _natural_sort_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _discover_real_dataset(root: Path) -> list[dict]:
    transients_dir = root / "transients"
    blends_dir = root / "blends"
    if not transients_dir.exists() or not blends_dir.exists():
        return []

    mat_files = sorted(transients_dir.glob("*.mat"))
    if not mat_files:
        return []

    gif_map = {_normalize_key(p.stem): p for p in blends_dir.glob("*.gif")}
    png_map = {_normalize_key(p.stem): p for p in blends_dir.glob("*.png")}
    labels = [_display_label(_extract_raw_label(m.stem)) for m in mat_files]

    scenes = []
    for mat_path in mat_files:
        raw_label = _extract_raw_label(mat_path.stem)
        label = _display_label(raw_label)
        key = _normalize_key(raw_label)

        recon_key = _choose_recon_key(raw_label, key, gif_map, png_map)
        recon_gif = gif_map.get(recon_key)
        recon_final = png_map.get(recon_key)
        if recon_final is None and "scene0" in png_map:
            recon_final = png_map["scene0"]
        if recon_final is None and png_map:
            recon_final = next(iter(png_map.values()))

        choices = sorted(set(labels))
        if label not in choices:
            choices.append(label)

        scenes.append(
            {
                "id": f"real_{raw_label}",
                "scene_dir": str(root),
                "transient_path": str(mat_path),
                "transient_type": "mat",
                "label": label,
                "choices": choices,
                "difficulty": 3,
                "hint": _default_hint(label),
                "notes": "Escena real NLOS (transient .mat) con reconstrucción precomputada.",
                "recon_gif": str(recon_gif) if recon_gif else None,
                "recon_final": str(recon_final) if recon_final else None,
                "render_3d": None,
            }
        )
    return scenes


def _extract_raw_label(stem: str) -> str:
    if stem.startswith("data_"):
        return stem[5:]
    return stem


def _display_label(raw: str) -> str:
    raw = raw.replace("_", " ").strip().lower()
    aliases = {
        "s u": "su",
        "exit sign": "exit sign",
    }
    return aliases.get(raw, raw)


def _default_hint(label: str) -> str:
    hint_map = {
        "bunny": "silueta con orejas",
        "exit sign": "texto/señal de salida",
        "mannequin": "forma humanoide",
        "su": "dos letras visibles",
    }
    return hint_map.get(label, "observa patrones de energía en el transient")


def _choose_recon_key(raw_label: str, key: str, gif_map: dict, png_map: dict) -> str | None:
    keys_available = set(gif_map) | set(png_map)
    if key in keys_available:
        return key

    explicit = {
        "exit_sign": "exit",
        "s_u": "su",
        "mannequin": "dummy",
    }
    alias = explicit.get(raw_label)
    if alias and alias in keys_available:
        return alias

    compressed = _normalize_key(raw_label.replace("_", ""))
    if compressed in keys_available:
        return compressed

    for token in re.split(r"[_\\s]+", raw_label):
        tk = _normalize_key(token)
        if tk and tk in keys_available:
            return tk

    return next(iter(keys_available)) if keys_available else None


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _first_existing(*paths: Path) -> str | None:
    for p in paths:
        if p.exists():
            return str(p)
    return None


def _load_mat_volume(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    if "rect_data" not in data:
        raise RuntimeError(f"{mat_path.name} no contiene 'rect_data'")
    vol = np.array(data["rect_data"], dtype=np.float32)
    if vol.ndim != 3:
        raise RuntimeError(f"{mat_path.name}: rect_data debe ser 3D")
    return vol


def _load_h5_front_rgb_volume(h5_path: Path, target_frames: int | None = 300) -> np.ndarray:
    """Load and temporally reduce the RGB transient from the frontal view."""
    if h5py is None:
        raise RuntimeError("Se necesita h5py para leer escenas HDF5")

    with h5py.File(h5_path, "r") as data:
        if "transients" not in data:
            raise RuntimeError(f"{h5_path.name} no contiene 'transients'")
        dataset = data["transients"]
        if dataset.ndim != 5:
            raise RuntimeError(f"{h5_path.name}: transients debe tener forma (vistas, alto, ancho, tiempo, canales)")
        if FRONT_VIEW_INDEX >= dataset.shape[0]:
            raise RuntimeError(f"{h5_path.name} no contiene la vista frontal {FRONT_VIEW_INDEX}")
        # The source is chunked by complete view, so a single read is considerably
        # faster than repeatedly decompressing it in temporal slices.
        volume = np.asarray(dataset[FRONT_VIEW_INDEX], dtype=np.float32)

    if target_frames is not None and volume.shape[2] > target_frames:
        group = int(np.ceil(volume.shape[2] / target_frames))
        usable = (volume.shape[2] // group) * group
        # Sum, rather than average, so collapsing the reduced time axis still
        # yields the physical temporal integral of the original measurement.
        reduced = volume[:, :, :usable, :].reshape(
            volume.shape[0], volume.shape[1], -1, group, volume.shape[3]
        ).sum(axis=3)
        if usable < volume.shape[2]:
            tail = volume[:, :, usable:, :].sum(axis=2, keepdims=True)
            reduced = np.concatenate((reduced, tail), axis=2)
        volume = reduced
    return volume


def _load_h5_front_volume(h5_path: Path, target_frames: int | None = 300) -> np.ndarray:
    """Load the frontal transient and collapse its RGB channels to intensity."""
    rgb = _load_h5_front_rgb_volume(h5_path, target_frames=target_frames)
    return rgb.mean(axis=-1, dtype=np.float32)


def _normalize_scalar_image(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, (1.0, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((gray - lo) / (hi - lo), 0.0, 1.0)


def _volume_to_frames(vol: np.ndarray) -> list[Image.Image]:
    lo = float(np.percentile(vol, 1.0))
    hi = float(np.percentile(vol, 99.5))
    if hi <= lo:
        hi = lo + 1e-6
    return [render_inferno_image(vol[:, :, i], lo, hi) for i in range(vol.shape[2])]


def _load_mat_transient_frames(mat_path: Path, target_frames: int = 300) -> list[Image.Image]:
    vol = _load_mat_volume(mat_path)
    t = vol.shape[2]
    if t <= target_frames:
        sampled = vol
    else:
        # Temporal binning preserves the full time axis while keeping playback lightweight.
        edges = np.linspace(0, t, target_frames + 1, dtype=int)
        bins = []
        for i in range(target_frames):
            a, b = int(edges[i]), int(edges[i + 1])
            if b <= a:
                b = min(a + 1, t)
            bins.append(vol[:, :, a:b].mean(axis=2))
        sampled = np.stack(bins, axis=2)

    lo = float(np.percentile(sampled, 1))
    hi = float(np.percentile(sampled, 99.5))
    if hi <= lo:
        hi = lo + 1e-6

    frames = []
    total = sampled.shape[2]
    for i in range(total):
        x = (sampled[:, :, i] - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        if colormaps is not None:
            rgb = (colormaps["inferno"](x)[..., :3] * 255.0).astype(np.uint8)
        else:
            rgb = _gray_to_rgb(x).astype(np.uint8)
        frames.append(Image.fromarray(rgb.astype(np.uint8), mode="RGB"))
    return frames


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    x = np.array(gray, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    r = 255.0 * (x ** 0.85)
    g = 235.0 * (x ** 0.65)
    b = 180.0 * (x ** 0.35)
    return np.stack([r, g, b], axis=-1)
