const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const MAX_ROUNDS = 5;

const state = {
  scenes: [], queue: [], current: null, selected: null,
  level: "Medio", round: 1, score: 0, streak: 0, correct: 0, answered: false,
};

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
}

function nextScene() {
  if (!state.queue.length) state.queue = shuffle(state.scenes);
  state.current = state.queue.pop();
  state.selected = null; state.answered = false;
  $("#result-overlay").classList.add("hidden");
  $("#submit-answer").disabled = true;
  $("#hint-text").classList.add("hidden");
  $("#hint-button").classList.remove("hidden");
  updateTransient();
  renderChoices();
  updateHud();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function updateTransient() {
  const image = $("#transient-image");
  const source = state.current.transients[state.level] || state.current.transients.Medio;
  image.classList.add("loading");
  image.onload = () => image.classList.remove("loading");
  image.src = `${source}?play=${Date.now()}`;
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
$("#reload-gif").addEventListener("click", updateTransient);
$("#next-button").addEventListener("click", advanceGame);
$("#close-result").addEventListener("click", advanceGame);
$("#restart-button").addEventListener("click", startGame);
$("#hint-button").addEventListener("click", () => {
  $("#hint-text").textContent = state.current.hint;
  $("#hint-text").classList.remove("hidden");
  $("#hint-button").classList.add("hidden");
});
$("#logo-home").addEventListener("click", (event) => {
  event.preventDefault(); $("#game-screen").classList.add("hidden"); $("#start-screen").classList.remove("hidden");
});
$$('[data-level]').forEach((button) => button.addEventListener("click", () => {
  state.level = button.dataset.level;
  $$('[data-level]').forEach((item) => item.classList.toggle("active", item === button));
  updateTransient();
}));
document.addEventListener("keydown", (event) => {
  if (!$("#result-overlay").classList.contains("hidden") && event.key === "Enter") {
    advanceGame(); return;
  }
  const number = Number(event.key);
  if (number >= 1 && number <= state.current?.choices.length) selectChoice(state.current.choices[number - 1]);
  if (event.key === "Enter") submitAnswer();
});

loadScenes();
