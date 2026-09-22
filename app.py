from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random

from flask import Flask, abort, jsonify, render_template, request, send_file

from utils.io import discover_game_scenes
from utils.placeholders import ensure_dummy_dataset
from utils.scoring import compute_round_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)


@lru_cache(maxsize=1)
def _scene_catalog() -> tuple[dict, ...]:
    scenes = discover_game_scenes(DATA_DIR)
    if not scenes:
        ensure_dummy_dataset(DATA_DIR)
        scenes = discover_game_scenes(DATA_DIR)
    return tuple(scenes)


def _media_url(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(BASE_DIR)
    except ValueError:
        return None
    return f"/media/{relative.as_posix()}" if candidate.is_file() else None


def _public_scene(scene: dict) -> dict:
    transient_gifs = scene.get("transient_gifs") or {}
    distractors = [choice for choice in scene["choices"] if choice != scene["label"]]
    choices = [scene["label"], *random.sample(distractors, min(3, len(distractors)))]
    random.shuffle(choices)
    return {
        "id": scene["id"],
        "choices": choices,
        "difficulty": scene.get("difficulty", 1),
        "hint": scene.get("hint", "Observa el patrón de energía"),
        "transients": {
            "Fácil": _media_url(transient_gifs.get("Fácil") or scene["transient_path"]),
            "Medio": _media_url(transient_gifs.get("Medio") or scene["transient_path"]),
            "Difícil": _media_url(transient_gifs.get("Difícil") or scene["transient_path"]),
        },
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/scenes")
def list_scenes():
    scenes = [_public_scene(scene) for scene in _scene_catalog()]
    if not scenes:
        return jsonify({"error": "No hay escenas disponibles"}), 503
    return jsonify({"scenes": scenes})


@app.post("/api/answer")
def check_answer():
    payload = request.get_json(silent=True) or {}
    scene_id = str(payload.get("scene_id", ""))
    guess = str(payload.get("guess", ""))
    scene = next((item for item in _scene_catalog() if item["id"] == scene_id), None)
    if scene is None or guess not in scene["choices"]:
        return jsonify({"error": "Escena o respuesta inválida"}), 400

    correct = guess == scene["label"]
    score = compute_round_score(correct)
    return jsonify(
        {
            "correct": correct,
            "correct_label": scene["label"],
            "points": score["points"],
            "notes": scene.get("notes", ""),
            "reconstruction": {
                "rgb": _media_url(scene.get("integrated_rgb_image")),
                "intensity": _media_url(scene.get("integrated_image")),
                "final": _media_url(scene.get("recon_final")),
                "animation": _media_url(scene.get("recon_gif")),
            },
        }
    )


@app.get("/media/<path:relative_path>")
def media(relative_path: str):
    candidate = (BASE_DIR / relative_path).resolve()
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError:
        abort(404)
    if not candidate.is_file():
        abort(404)
    return send_file(candidate, conditional=True, max_age=3600)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=True)
