# Transient · Juego de reconocimiento

Interfaz web tipo juego para observar y clasificar imágenes transitorias preprocesadas.

## Ejecutar

```bash
python -m pip install -r requirements.txt
python app.py
```

Abre `http://127.0.0.1:8501` en el navegador.

La aplicación usa Flask para servir la interfaz y validar las respuestas. El juego, las animaciones y la interacción están implementados en HTML, CSS y JavaScript; no requiere Streamlit.
