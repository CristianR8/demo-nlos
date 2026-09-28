const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const MAX_ROUNDS = 5;
const MEDIAPIPE_VERSION = "1.0.1";
const MEDIAPIPE_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const GESTURE_MODEL = "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task";
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const GESTURE_NAMES = {
  Closed_Fist: "Puño cerrado", Pointing_Up: "Índice arriba",
  Thumb_Down: "Pulgar abajo", Thumb_Up: "Pulgar arriba",
  Victory: "Victoria", ILoveYou: "Te quiero",
};

const state = {
  scenes: [], queue: [], current: null, selected: null,
  round: 1, score: 0, streak: 0, correct: 0, answered: false,
};
let recognizer = null;
let cameraStream = null;
let cameraRequest = null;
let cameraGeneration = 0;
let trackingFrame = null;
let lastVideoTime = -1;
let lastDetectionTime = 0;
let openFrames = 0;
let closedFrames = 0;
let handOpen = false;

function shuffle(values) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

async function loadScenes() {
  try {
    const response = await fetch("/api/scenes");
    if (!response.ok) throw new Error("No fue posible cargar las mediciones");
    const payload = await response.json();
    state.scenes = payload.scenes;
  } catch (error) {
    toast(error.message);
    $("#start-button").disabled = true;
  }
}

function startGame() {
  if (!state.scenes.length) return;
  state.queue = shuffle(state.scenes);
  state.round = 1; state.score = 0; state.streak = 0; state.correct = 0;
  $("#start-screen").classList.add("hidden");
  $("#end-overlay").classList.add("hidden");
  $("#game-screen").classList.remove("hidden");
  nextScene();
  startCamera();
}

function nextScene() {
  if (!state.queue.length) state.queue = shuffle(state.scenes);
  state.current = state.queue.pop();
  state.selected = null; state.answered = false;
  handOpen = false; openFrames = 0; closedFrames = 0;
  $("#camera-frame").classList.remove("open-hand");
  $("#result-overlay").classList.add("hidden");
  $("#submit-answer").disabled = true;
  $("#hint-text").classList.add("hidden");
  $("#hint-button").classList.remove("hidden");
  hideTransient();
  renderChoices();
  updateHud();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showTransient() {
  if (!state.current || state.answered) return;
  const image = $("#transient-image");
  if (!image.classList.contains("hidden")) return;
  image.src = `${state.current.transient}?play=${Date.now()}`;
  image.classList.remove("hidden");
  $("#hand-prompt").classList.add("hidden");
}

function hideTransient() {
  const image = $("#transient-image");
  image.classList.add("hidden");
  image.removeAttribute("src");
  $("#hand-prompt").classList.remove("hidden");
}

function setCameraStatus(message, canRetry = false) {
  $("#camera-status").textContent = message;
  $("#retry-camera").classList.toggle("hidden", !canRetry);
}

function stopCamera() {
  cameraGeneration += 1;
  cameraRequest = null;
  if (trackingFrame !== null) cancelAnimationFrame(trackingFrame);
  trackingFrame = null;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  $("#camera-preview").srcObject = null;
  handOpen = false;
  openFrames = 0;
  closedFrames = 0;
  lastVideoTime = -1;
  lastDetectionTime = 0;
  clearHandOverlay();
  $("#gesture-badge").textContent = "Sin mano detectada";
  $("#camera-frame").classList.remove("open-hand");
  hideTransient();
}

function clearHandOverlay() {
  const canvas = $("#hand-overlay");
  const context = canvas.getContext("2d");
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function drawHandOverlay(hands, video) {
  const canvas = $("#hand-overlay");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * pixelRatio);
  const pixelHeight = Math.round(height * pixelRatio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!hands?.length || !video.videoWidth || !video.videoHeight) return;

  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
  const offsetX = (width - video.videoWidth * scale) / 2;
  const offsetY = (height - video.videoHeight * scale) / 2;
  const point = (landmark) => ({
    x: offsetX + landmark.x * video.videoWidth * scale,
    y: offsetY + landmark.y * video.videoHeight * scale,
  });
  context.strokeStyle = handOpen ? "#c9f35b" : "#3fe0dc";
  context.fillStyle = handOpen ? "#c9f35b" : "#3fe0dc";
  context.lineWidth = 2;
  for (const landmarks of hands) {
    context.beginPath();
    for (const [from, to] of HAND_CONNECTIONS) {
      const start = point(landmarks[from]);
      const end = point(landmarks[to]);
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
    }
    context.stroke();
    for (const landmark of landmarks) {
      const { x, y } = point(landmark);
      context.beginPath();
      context.arc(x, y, 3.5, 0, Math.PI * 2);
      context.fill();
    }
  }
}

async function startCamera() {
  if (cameraRequest || cameraStream) return;
  const generation = ++cameraGeneration;
  setCameraStatus("Solicitando cámara…");
  const request = (async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Cámara no disponible en este navegador o conexión.");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      if (generation !== cameraGeneration) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      cameraStream = stream;
      stream.getVideoTracks()[0]?.addEventListener("ended", () => {
        if (generation !== cameraGeneration) return;
        stopCamera();
        setCameraStatus("La cámara se desconectó.", true);
      });
      const video = $("#camera-preview");
      video.srcObject = stream;
      await video.play();
      if (generation !== cameraGeneration) return;
      setCameraStatus("Cargando detector de gestos…");
      if (!recognizer) {
        const { FilesetResolver, GestureRecognizer } = await import(`${MEDIAPIPE_CDN}/vision_bundle.mjs`);
        const vision = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
        recognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: { modelAssetPath: GESTURE_MODEL },
          runningMode: "VIDEO",
          numHands: 1,
        });
      }
      if (generation !== cameraGeneration) return;
      setCameraStatus("Esperando mano abierta");
      trackHands(generation);
    } catch (error) {
      if (generation !== cameraGeneration) return;
      stopCamera();
      setCameraStatus(error.name === "NotAllowedError"
        ? "Permite el acceso a la cámara para ver la medición."
        : "No se pudo iniciar la cámara o MediaPipe.", true);
    }
  })();
  cameraRequest = request;
  try { await request; } finally { if (cameraRequest === request) cameraRequest = null; }
}

function trackHands(generation) {
  if (generation !== cameraGeneration || !cameraStream) return;
  const video = $("#camera-preview");
  const now = performance.now();
  if (now - lastDetectionTime >= 100 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    try {
      const result = recognizer.recognizeForVideo(video, now);
      const detectedGesture = result.gestures?.[0]?.[0];
      const open = detectedGesture?.categoryName === "Open_Palm" && detectedGesture.score >= 0.6;
      openFrames = open ? openFrames + 1 : 0;
      closedFrames = open ? 0 : closedFrames + 1;
      if (!handOpen && openFrames >= 2) {
        handOpen = true;
        showTransient();
      } else if (handOpen && closedFrames >= 2) {
        handOpen = false;
        hideTransient();
      }
      const hasHand = Boolean(result.landmarks?.length);
      const gestureName = open ? "Palma abierta" : GESTURE_NAMES[detectedGesture?.categoryName];
      $("#gesture-badge").textContent = hasHand ? gestureName || "Mano detectada" : "Sin mano detectada";
      $("#camera-frame").classList.toggle("open-hand", handOpen);
      setCameraStatus(handOpen
        ? state.answered ? "Palma abierta" : "Palma abierta · medición visible"
        : hasHand ? "Mano detectada · abre la palma" : "Sin mano detectada");
      drawHandOverlay(result.landmarks, video);
      lastVideoTime = video.currentTime;
      lastDetectionTime = now;
    } catch (error) {
      stopCamera();
      setCameraStatus("Se interrumpió el detector de gestos.", true);
      return;
    }
  }
  trackingFrame = requestAnimationFrame(() => trackHands(generation));
}

function renderChoices() {
  const container = $("#choices");
  container.replaceChildren();
  state.current.choices.forEach((choice, index) => {
    const button = document.createElement("button");
    button.className = "choice";
    button.dataset.choice = choice;
    button.innerHTML = `<span class="choice-index">${index + 1}</span><strong>${escapeHtml(choice)}</strong><span class="choice-check">✓</span>`;
    button.addEventListener("click", () => selectChoice(choice));
    container.append(button);
  });
}

function selectChoice(choice) {
  if (state.answered) return;
  state.selected = choice;
  $$(".choice").forEach((button) => button.classList.toggle("selected", button.dataset.choice === choice));
  $("#submit-answer").disabled = false;
}

async function submitAnswer() {
  if (!state.selected || state.answered) return;
  state.answered = true;
  hideTransient();
  $("#submit-answer").disabled = true;
  try {
    const response = await fetch("/api/answer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scene_id: state.current.id, guess: state.selected }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "No fue posible validar la respuesta");
    state.score += result.points;
    state.streak = result.correct ? state.streak + 1 : 0;
    if (result.correct) state.correct += 1;
    showResult(result);
    updateHud();
  } catch (error) {
    state.answered = false;
    if (handOpen) showTransient();
    $("#submit-answer").disabled = false;
    toast(error.message);
  }
}

function showResult(result) {
  const card = $(".result-card");
  card.classList.toggle("incorrect", !result.correct);
  $("#result-icon").textContent = result.correct ? "✓" : "×";
  $("#result-kicker").textContent = result.correct ? "LECTURA CONFIRMADA" : "LECTURA CORREGIDA";
  $("#result-title").textContent = result.correct ? "¡Imagen identificada!" : "Casi lo tienes";
  $("#result-copy").textContent = result.correct
    ? `La firma coincide con “${result.correct_label}”. Has interpretado correctamente el retorno.`
    : `Elegiste “${state.selected}”, pero la firma corresponde a “${result.correct_label}”.`;
  $("#earned-points").textContent = `+${result.points}`;
  $("#next-button").firstChild.textContent = state.round >= MAX_ROUNDS
    ? "Ver resultado final "
    : "Siguiente imagen ";

  const grid = $("#reconstruction-grid");
  grid.replaceChildren();
  const views = [
    [result.reconstruction.rgb, "COLOR INTEGRADO"],
    [result.reconstruction.intensity, "INTENSIDAD INTEGRADA"],
    [result.reconstruction.animation || result.reconstruction.final, "RECONSTRUCCIÓN"],
  ].filter(([source]) => source);
  views.forEach(([source, label]) => {
    const figure = document.createElement("div");
    figure.className = "reconstruction";
    const image = document.createElement("img"); image.src = source; image.alt = label;
    const caption = document.createElement("span"); caption.textContent = label;
    figure.append(image, caption); grid.append(figure);
  });
  grid.classList.toggle("hidden", views.length === 0);
  $("#result-overlay").classList.remove("hidden");
}

function updateHud() {
  $("#score").textContent = String(state.score).padStart(4, "0");
  $("#streak").textContent = `×${state.streak}`;
  $("#round-number").textContent = String(state.round).padStart(2, "0");
  $("#progress-label").textContent = `RONDA ${String(state.round).padStart(2, "0")}`;
  $("#progress-bar").style.width = `${(state.round / MAX_ROUNDS) * 100}%`;
}

function advanceGame() {
  if (state.round >= MAX_ROUNDS) {
    stopCamera();
    $("#result-overlay").classList.add("hidden");
    $("#final-score").textContent = String(state.score).padStart(4, "0");
    $("#final-correct").textContent = `${state.correct} / ${MAX_ROUNDS}`;
    $("#end-overlay").classList.remove("hidden");
    return;
  }
  state.round += 1;
  nextScene();
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message; element.classList.remove("hidden");
  window.setTimeout(() => element.classList.add("hidden"), 3500);
}

function escapeHtml(text) {
  const node = document.createElement("span"); node.textContent = text; return node.innerHTML;
}

$("#start-button").addEventListener("click", startGame);
$("#submit-answer").addEventListener("click", submitAnswer);
$("#retry-camera").addEventListener("click", startCamera);
$("#next-button").addEventListener("click", advanceGame);
$("#close-result").addEventListener("click", advanceGame);
$("#restart-button").addEventListener("click", startGame);
$("#hint-button").addEventListener("click", () => {
  $("#hint-text").textContent = state.current.hint;
  $("#hint-text").classList.remove("hidden");
  $("#hint-button").classList.add("hidden");
});
$("#logo-home").addEventListener("click", (event) => {
  event.preventDefault(); stopCamera(); $("#game-screen").classList.add("hidden"); $("#start-screen").classList.remove("hidden");
});
document.addEventListener("keydown", (event) => {
  if (!$("#result-overlay").classList.contains("hidden") && event.key === "Enter") {
    advanceGame(); return;
  }
  const number = Number(event.key);
  if (number >= 1 && number <= state.current?.choices.length) selectChoice(state.current.choices[number - 1]);
  if (event.key === "Enter") submitAnswer();
});

loadScenes();
