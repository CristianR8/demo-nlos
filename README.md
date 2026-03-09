# Demo NLOS: ¿Qué hay detrás de la esquina?

App de Streamlit tipo juego para adivinar la escena oculta a partir de una medición transient NLOS.

## Estructura

- `app.py`
- `utils/io.py`
- `utils/scoring.py`
- `utils/placeholders.py`
- `data/` (se genera automáticamente si no existe)
- `reconstruct.py` (opcional, modo REAL)

## Requisitos

- Python 3.10+

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar

```bash
streamlit run app.py
```

## Datos

Formato esperado por escena:

```text
data/scene_001/
  transient.gif
  recon.gif
  recon_final.png   # o depth.png
  meta.json
```

`meta.json` ejemplo:

```json
{
  "label": "bunny",
  "choices": ["bunny", "sphere", "chair", "teapot"],
  "difficulty": 2,
  "hint": "tiene orejas",
  "notes": "scene sintetica mitsuba"
}
```

Si `./data` no tiene escenas válidas, la app crea escenas dummy automáticamente.

También soporta dataset real en:

```text
data/scenes/
  transients/*.mat
  blends/*.gif
  blends/*.png
```

La app asocia transient y reconstrucción usando los nombres de archivo (con reglas para casos como `data_exit_sign.mat -> exit.gif` y `data_mannequin.mat -> dummy.gif`).

## Modo REAL (opcional)

Si existe `reconstruct.py` en la raíz del proyecto, la app intentará usarlo cuando selecciones modo `REAL`.

Interfaz esperada:

```python
# reconstruct.py
from pathlib import Path

def reconstruct(scene_dir: Path) -> dict:
    return {
        "recon_gif": str(scene_dir / "recon.gif"),
        "recon_final": str(scene_dir / "recon_final.png"),
        "render_3d": None,
    }
```

Si falla o no existe, la app vuelve automáticamente al modo `DEMO`.
