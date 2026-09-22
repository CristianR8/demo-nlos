from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

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
        "scene_queue": [],
        "last_scene": None,
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


def draw_scene_without_repeats(
    scene_pool: list[str],
    scene_queue: list[str],
    last_scene: str | None,
) -> tuple[str | None, list[str]]:
    """Draw from a shuffled bag so every available scene appears once per cycle."""
    if not scene_pool:
        return None, []

    available = set(scene_pool)
    queue = [scene_id for scene_id in scene_queue if scene_id in available]
    if not queue:
        queue = list(scene_pool)
        random.shuffle(queue)
        # Avoid an immediate repeat at the boundary between two complete cycles.
        if len(queue) > 1 and queue[-1] == last_scene:
            queue[0], queue[-1] = queue[-1], queue[0]

    return queue.pop(), queue


def pick_new_scene(scene_pool: list[str]) -> None:
    queue = list(st.session_state.scene_queue)
    last_scene = st.session_state.last_scene
    # When upgrading an already-running session, count the scene currently on
    # screen as the first draw of the new cycle.
    if not queue and last_scene is None and st.session_state.current_scene in scene_pool:
        last_scene = st.session_state.current_scene
        queue = [scene_id for scene_id in scene_pool if scene_id != last_scene]
        random.shuffle(queue)

    selected, remaining = draw_scene_without_repeats(
        scene_pool,
        queue,
        last_scene,
    )
    if selected is None:
        st.session_state.current_scene = None
        st.session_state.scene_queue = []
        return

    st.session_state.current_scene = selected
    st.session_state.scene_queue = remaining
    st.session_state.last_scene = selected
    st.session_state.revealed = False
    st.session_state.recon_outputs = None
    st.session_state.last_points = 0
    st.session_state.locked_guess = None
    st.session_state.locked_correct = None


def run_reconstruction(scene: dict, mode: str) -> dict:
    if mode == "DEMO":
        integrated_image = scene.get("integrated_image")
        integrated_rgb_image = scene.get("integrated_rgb_image")
        if scene.get("transient_type") == "h5":
            ui_data = cached_transient_volume(scene["transient_path"], "h5")
            integrated_image = render_inferno_image(
                ui_data["integrated"],
                float(ui_data["integrated_lo"]),
                float(ui_data["integrated_hi"]),
            )
        return {
            "recon_gif": scene.get("recon_gif"),
            "recon_final": scene.get("recon_final"),
            "render_3d": scene.get("render_3d"),
            "integrated_image": integrated_image,
            "integrated_rgb_image": integrated_rgb_image,
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

    st.subheader("Medición transitoria")
    level = st.selectbox(
        "Nivel",
        options=["Fácil", "Medio", "Difícil"],
        index=1,
        key=f"level_{scene_id}",
    )

    if scene["transient_type"] == "precomputed_gif":
        transient_path = scene.get("transient_gifs", {}).get(level, scene["transient_path"])
        st.image(
            transient_path,
            caption=f"Transient NLOS · nivel {level.lower()}",
            use_container_width=True,
        )
    elif scene["transient_type"] in {"mat", "h5"}:
        st.image(
            cached_transient(scene["transient_path"], scene["transient_type"], 1),
            caption="Transient NLOS",
            use_container_width=True,
        )
    else:
        st.image(scene["transient_path"], caption="Transient NLOS", use_container_width=True)

    guess = st.radio(
        "Tu predicción",
        options=meta["choices"],
        index=0,
        key=f"guess_{scene_id}_{st.session_state.round_idx}",
        horizontal=True,
    )

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
                    "level": level,
                }
            )
            st.session_state.revealed = True

    outputs = st.session_state.recon_outputs
    if outputs:
        st.subheader("Reconstrucción")
        rgb_col, intensity_col = st.columns(2)
        with rgb_col:
            if outputs.get("integrated_rgb_image"):
                st.image(
                    outputs["integrated_rgb_image"],
                    caption="Reconstrucción RGB colapsada en el tiempo",
                    use_container_width=True,
                )
        with intensity_col:
            if outputs.get("integrated_image"):
                st.image(
                    outputs["integrated_image"],
                    caption="Intensidad colapsada en el tiempo",
                    use_container_width=True,
                )

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

    st.divider()
    if st.button("Nueva escena", use_container_width=True):
        if st.session_state.revealed:
            st.session_state.round_idx += 1
        pick_new_scene(scene_ids)
        st.rerun()


if __name__ == "__main__":
    main()
