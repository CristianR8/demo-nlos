# Transient · Juego de reconocimiento

Interfaz web tipo juego para observar y clasificar imágenes transitorias preprocesadas.

## Ejecutar

```bash
python -m pip install -r requirements.txt
python app.py
```

Abre `http://127.0.0.1:8501` en el navegador.

La aplicación usa Flask para servir la interfaz y validar las respuestas. El juego, las animaciones y la interacción están implementados en HTML, CSS y JavaScript; no requiere Streamlit.

Al iniciar una partida, concede permiso para usar la cámara. MediaPipe Gesture Recognizer detecta el gesto `Open_Palm`: la medición transitoria rápida aparece mientras mantienes la mano abierta y se oculta al cerrarla o retirarla. El modelo y la librería se descargan al iniciar la cámara, así que se necesita conexión a Internet. La cámara requiere `localhost` o HTTPS en el navegador.
