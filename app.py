from __future__ import annotations

import base64
import io
import importlib.util
import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter

from utils.io import (
    discover_game_scenes,
    get_transient_volume_for_ui,
    get_transient_gif_bytes,
    render_inferno_image,
)
from utils.placeholders import ensure_dummy_dataset
from utils.scoring import compute_round_score


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REAL_SCENES_DIR = DATA_DIR / "scenes"
RECON_SCRIPT = BASE_DIR / "reconstruct.py"
ILLUSTRATION_PATH = BASE_DIR / "illustration.png"
HINT_BLUR_SIGMA = 100.0
HINT_POISSON_SCALE = 100.0


@st.cache_data
def cached_game_scenes(data_dir: str) -> list[dict]:
    return discover_game_scenes(Path(data_dir))


@st.cache_data
def cached_transient(
    transient_path: str,
    transient_type: str,
    upscale: int,
) -> bytes:
    return get_transient_gif_bytes(
        Path(transient_path),
        transient_type=transient_type,
        only_three_frames=False,
        upscale=upscale,
    )


@st.cache_data
def cached_transient_volume(
    transient_path: str,
    transient_type: str,
) -> dict:
    return get_transient_volume_for_ui(
        Path(transient_path),
        transient_type=transient_type,
        target_frames=300,
        log_scale=False,
    )


@st.cache_data
def cached_choice_images(data_dir: str) -> dict[str, str]:
    """Return reference image path for each class option in the real dataset."""
    blends = Path(data_dir) / "scenes" / "blends"
    if not blends.exists():
        return {}

    pngs = {p.stem.lower(): str(p) for p in blends.glob("*.png")}
    option_map = {
        "bunny": ["bunny", "scene0"],
        "mannequin": ["mannequin", "dummy"],
        "su": ["su"],
        "exit sign": ["exitsign", "exit_sign", "exit"],
    }

    resolved: dict[str, str] = {}
    for label, keys in option_map.items():
        for k in keys:
            if k in pngs:
                resolved[label] = pngs[k]
                break
    return resolved


@st.cache_data
def cached_poisson_hint_image(image_path: str) -> bytes:
    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    blurred = gaussian_filter(arr, sigma=(HINT_BLUR_SIGMA, HINT_BLUR_SIGMA, 0))

    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(image_path)) % (2**32)
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(np.clip(blurred, 0.0, 1.0) * HINT_POISSON_SCALE).astype(np.float32) / HINT_POISSON_SCALE
    noisy = np.clip(noisy, 0.0, 1.0)

    out = io.BytesIO()
    Image.fromarray((noisy * 255.0).astype(np.uint8), mode="RGB").save(out, format="PNG")
    return out.getvalue()


@st.cache_data
def cached_image_data_uri(image_path: str) -> str:
    path = Path(image_path)
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def resolve_hint_image_path(scene: dict, choice_images: dict[str, str]) -> str | None:
    label_path = choice_images.get(str(scene.get("label", "")).lower())
    if label_path:
        return label_path

    recon_final = scene.get("recon_final")
    if recon_final and Path(recon_final).exists():
        return recon_final

    return None


@st.cache_data
def real_reconstruct(scene_dir: str, scene_id: str, script_mtime: float) -> dict:
    """Run real reconstruction function from reconstruct.py with cache."""
    del scene_id
    del script_mtime

    spec = importlib.util.spec_from_file_location("nlos_reconstruct", str(RECON_SCRIPT))
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar reconstruct.py")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "reconstruct") or not callable(mod.reconstruct):
        raise RuntimeError("reconstruct.py no expone reconstruct(scene_dir)")

    outputs = mod.reconstruct(Path(scene_dir))
    if not isinstance(outputs, dict):
        raise RuntimeError("reconstruct(scene_dir) debe devolver un dict")

    return {
        "recon_gif": outputs.get("recon_gif"),
        "recon_final": outputs.get("recon_final") or outputs.get("depth"),
        "render_3d": outputs.get("render_3d") or outputs.get("volume"),
        "mode": "real",
    }


def init_state() -> None:
    defaults = {
        "game_started": False,
        "round_idx": 1,
        "total_score": 0,
        "history": [],
        "current_scene": None,
        "revealed": False,
        "recon_outputs": None,
        "last_points": 0,
        "locked_guess": None,
        "locked_correct": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_start_screen() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f2ea 0%, #ece3d2 100%);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            max-width: 100%;
            min-height: 100vh;
            padding-top: 0;
            padding-bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .start-copy h1 {
            margin: 1.25rem 0 0.75rem;
            color: #3b2414;
            font-size: clamp(2.4rem, 5vw, 4rem);
            line-height: 1;
            text-align: center;
        }
        .start-copy p {
            margin: 0 auto 1.5rem;
            max-width: 38rem;
            color: #5d4330;
            font-size: 1.05rem;
            text-align: center;
        }
        div[data-testid="stForm"] {
            width: min(1320px, calc(100vw - 4rem));
            margin: 0;
            padding: 2.5rem 2rem 2rem;
            border: 1px solid rgba(83, 49, 22, 0.14);
            border-radius: 28px;
            background: rgba(255, 250, 243, 0.88);
            box-shadow: 0 24px 70px rgba(80, 52, 26, 0.10);
        }
        div[data-testid="stForm"] img {
            border-radius: 20px;
            box-shadow: 0 16px 40px rgba(80, 52, 26, 0.14);
        }
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            max-width: 240px;
            margin: 0 auto;
            border: none;
            border-radius: 999px;
            background: #c62828;
            color: #fff7f7;
            font-weight: 700;
            font-size: 1rem;
            padding: 0.85rem 1.2rem;
            box-shadow: 0 12px 26px rgba(198, 40, 40, 0.28);
        }
        div[data-testid="stFormSubmitButton"] {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("start_game_form", clear_on_submit=False):
        if ILLUSTRATION_PATH.exists():
            st.image(str(ILLUSTRATION_PATH), use_container_width=True)
        st.markdown(
            """
            <div class="start-copy">
                <h1>NLOS Guess Demo</h1>
                <p>The illustration shows the NLOS setup before the game starts, so the player can understand how the hidden-object measurement works.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        started = st.form_submit_button("Start Game", use_container_width=True)

    if started:
        st.session_state.game_started = True
        st.rerun()


def pick_new_scene(scene_pool: list[str]) -> None:
    if not scene_pool:
        st.session_state.current_scene = None
        return

    previous = st.session_state.current_scene
    candidates = [s for s in scene_pool if s != previous] or scene_pool

    st.session_state.current_scene = random.choice(candidates)
    st.session_state.revealed = False
    st.session_state.recon_outputs = None
    st.session_state.last_points = 0
    st.session_state.locked_guess = None
    st.session_state.locked_correct = None


def run_reconstruction(scene: dict, mode: str) -> dict:
    if mode == "DEMO":
        return {
            "recon_gif": scene.get("recon_gif"),
            "recon_final": scene.get("recon_final"),
            "render_3d": scene.get("render_3d"),
            "mode": "demo",
        }

    if not RECON_SCRIPT.exists():
        st.warning("No existe reconstruct.py. Se usa modo DEMO.")
        return {
            "recon_gif": scene.get("recon_gif"),
            "recon_final": scene.get("recon_final"),
            "render_3d": scene.get("render_3d"),
            "mode": "demo",
        }

    try:
        with st.spinner("Ejecutando reconstruccion real..."):
            progress = st.progress(0)
            progress.progress(10)
            out = real_reconstruct(
                scene.get("scene_dir", str(DATA_DIR)),
                scene.get("id", "unknown"),
                RECON_SCRIPT.stat().st_mtime,
            )
            progress.progress(100)
        return out
    except Exception as ex:
        st.warning(f"Fallo reconstruccion REAL ({ex}). Se usa modo DEMO.")
        return {
            "recon_gif": scene.get("recon_gif"),
            "recon_final": scene.get("recon_final"),
            "render_3d": scene.get("render_3d"),
            "mode": "demo",
        }


def render_score_panel() -> None:
    st.metric("Ronda #", st.session_state.round_idx)
    st.metric("Puntaje total", st.session_state.total_score)

    if st.session_state.history:
        st.caption("Historial")
        st.dataframe(st.session_state.history, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="NLOS Guess Demo", layout="wide")

    init_state()

    if not st.session_state.game_started:
        render_start_screen()
        return

    st.title("¿Qué objeto se ecuentra oculto?")

    force_real_data = (REAL_SCENES_DIR / "transients").exists()
    scenes = cached_game_scenes(str(DATA_DIR))
    if force_real_data:
        scenes = [s for s in scenes if s.get("transient_type") == "mat"]
    if not scenes and not force_real_data:
        ensure_dummy_dataset(DATA_DIR)
        cached_game_scenes.clear()
        scenes = cached_game_scenes(str(DATA_DIR))
    if force_real_data and not scenes:
        st.error("No se detectaron escenas reales válidas en data/scenes/transients/*.mat")
        return
    if not scenes:
        st.error("No hay escenas disponibles en ./data")
        return

    scene_map = {s["id"]: s for s in scenes}
    scene_ids = list(scene_map.keys())

    if st.session_state.current_scene not in scene_map:
        st.session_state.current_scene = None

    if st.session_state.current_scene is None:
        pick_new_scene(scene_ids)

    scene_id = st.session_state.current_scene
    scene = scene_map[scene_id]
    meta = {
        "label": scene["label"],
        "choices": scene["choices"],
        "difficulty": scene["difficulty"],
        "notes": scene["notes"],
    }

    with st.sidebar:
        st.subheader("Marcador")
        render_score_panel()

    col_left, col_right = st.columns(2)
    choice_images = cached_choice_images(str(DATA_DIR))

    with col_left:
        st.subheader("Medición")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Nueva escena", use_container_width=True):
                pick_new_scene(scene_ids)
                st.rerun()
        with c2:
            if st.session_state.revealed and st.button("Siguiente", use_container_width=True):
                st.session_state.round_idx += 1
                pick_new_scene(scene_ids)
                st.rerun()

        if scene["transient_type"] == "mat":
            ui_data = cached_transient_volume(
                scene["transient_path"],
                scene["transient_type"],
            )
            vol = ui_data["volume"]
            h, w, t_total = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
            display_size = 512
            img_width = display_size
            img_height = display_size
            gif_upscale = max(1, int(np.ceil(display_size / max(w, 1))))

            t_idx = st.slider("Frame temporal", min_value=0, max_value=t_total - 1, value=t_total // 2, step=1)
            frame_img = render_inferno_image(vol[:, :, t_idx], float(ui_data["lo"]), float(ui_data["hi"]))
            integrated_img = render_inferno_image(
                ui_data["integrated"],
                float(ui_data["integrated_lo"]),
                float(ui_data["integrated_hi"]),
            )
            frame_img = frame_img.resize((img_width, img_height), resample=Image.Resampling.NEAREST)
            integrated_img = integrated_img.resize((img_width, img_height), resample=Image.Resampling.NEAREST)

            tab_anim, tab_frame, tab_sum = st.tabs(["Animación", "Frame temporal", "Suma temporal"])
            with tab_anim:
                st.image(
                    cached_transient(scene["transient_path"], scene["transient_type"], gif_upscale),
                    caption="Transient NLOS (barrido temporal)",
                    width=img_width,
                )
            with tab_frame:
                st.image(
                    frame_img,
                    caption=f"Frame temporal t={t_idx}/{t_total - 1} | resolución {w}x{h}",
                    width=img_width,
                )
            with tab_sum:
                st.image(
                    integrated_img,
                    caption=f"Suma temporal de energía | resolución {w}x{h}",
                    width=img_width,
                )
        else:
            st.image(
                cached_transient(scene["transient_path"], scene["transient_type"], 1),
                caption="Transient NLOS",
                use_container_width=True,
            )

        st.caption("Opciones de objeto")
        c1, c2 = st.columns(2)
        for i, label in enumerate(meta["choices"]):
            target_col = c1 if i % 2 == 0 else c2
            with target_col:
                img_path = choice_images.get(label.lower())
                if img_path:
                    st.image(img_path, caption=label, use_container_width=True)
                else:
                    st.caption(label)

        hint_image_path = resolve_hint_image_path(scene, choice_images)
        show_visual_hint = st.checkbox(
            "Mostrar pista visual",
            value=False,
            key=f"visual_hint_{scene_id}_{st.session_state.round_idx}",
            help=(
                "Muestra la imagen de referencia degradada con blur_sigma=100 "
                "y poisson_scale=100."
            ),
        )
        if show_visual_hint:
            if hint_image_path:
                st.image(
                    cached_poisson_hint_image(hint_image_path),
                    caption=(
                        f"Pista visual | blur_sigma={int(HINT_BLUR_SIGMA)} | "
                        f"poisson_scale={int(HINT_POISSON_SCALE)}"
                    ),
                    use_container_width=True,
                )
            else:
                st.caption("No hay imagen disponible para generar la pista visual.")

        guess = st.radio(
            "Tu predicción",
            options=meta["choices"],
            index=0,
            key=f"guess_{scene_id}_{st.session_state.round_idx}",
        )

        st.caption(f"Dificultad: {meta.get('difficulty', 1)}")

    with col_right:
        st.subheader("Reconstrucción")
        reconstruct_clicked = st.button("Reconstruct", type="primary", use_container_width=True)

        if reconstruct_clicked:
            st.session_state.recon_outputs = run_reconstruction(scene, "DEMO")

            if not st.session_state.revealed:
                correct = guess == meta["label"]
                st.session_state.locked_guess = guess
                st.session_state.locked_correct = correct
                score = compute_round_score(correct=correct)
                st.session_state.last_points = score["points"]
                st.session_state.total_score += score["points"]
                st.session_state.history.append(
                    {
                        "round": st.session_state.round_idx,
                        "scene": scene_id,
                        "guess": guess,
                        "correct_label": meta["label"],
                        "correct": correct,
                        "points": score["points"],
                    }
                )
                st.session_state.revealed = True

        outputs = st.session_state.recon_outputs
        if outputs:
            if outputs.get("recon_gif"):
                st.image(outputs["recon_gif"], caption=f"Animación de reconstrucción ({outputs.get('mode', 'demo')})", use_container_width=True)

            if outputs.get("render_3d"):
                st.image(outputs["render_3d"], caption="Render 3D", use_container_width=True)

        if st.session_state.revealed:
            correct = bool(st.session_state.locked_correct)
            locked_guess = st.session_state.locked_guess
            if correct:
                st.success(
                    f"Correcto: {meta['label']} (tu respuesta: {locked_guess}) | "
                    f"+{st.session_state.last_points} puntos"
                )
            else:
                st.error(
                    f"Respuesta correcta: {meta['label']} (tu respuesta: {locked_guess}) | "
                    f"{st.session_state.last_points} puntos"
                )

            st.caption(meta.get("notes", ""))


if __name__ == "__main__":
    main()
