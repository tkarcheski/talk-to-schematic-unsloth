"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  example: null,
  image: null,
  history: [],
  health: null,
  pending: false,
  selection: 0,
  notice: "",
  zoom: 1,
  fit: true,
};
function setZoom(value, preserveCenter = true) {
  const image = $("schematic-image");
  const stage = $("image-stage");
  if (!state.image || !image.naturalWidth || !image.naturalHeight) return;
  const centerX =
    (stage.scrollLeft + stage.clientWidth / 2) / stage.scrollWidth;
  const centerY =
    (stage.scrollTop + stage.clientHeight / 2) / stage.scrollHeight;
  state.fit = value === "fit";
  state.zoom = state.fit
    ? Math.min(
        1,
        (stage.clientWidth - 32) / image.naturalWidth,
        (stage.clientHeight - 32) / image.naturalHeight,
      )
    : Math.max(0.05, Math.min(4, value));
  image.style.width = `${image.naturalWidth * state.zoom}px`;
  $("zoom-level").textContent = `${Math.round(state.zoom * 100)}%`;
  $("zoom-fit").setAttribute("aria-pressed", String(state.fit));
  $("zoom-actual").setAttribute(
    "aria-pressed",
    String(!state.fit && state.zoom === 1),
  );
  if (state.fit || !preserveCenter) {
    stage.scrollLeft = 0;
    stage.scrollTop = 0;
  } else {
    stage.scrollLeft = centerX * stage.scrollWidth - stage.clientWidth / 2;
    stage.scrollTop = centerY * stage.scrollHeight - stage.clientHeight / 2;
  }
  $("zoom-out").disabled = state.zoom <= 0.05;
  $("zoom-in").disabled = state.zoom >= 4;
  $("zoom-fit").disabled = false;
  $("zoom-actual").disabled = false;
}
$("zoom-in").addEventListener("click", () => setZoom(state.zoom * 1.35));
$("zoom-out").addEventListener("click", () => setZoom(state.zoom / 1.35));
$("zoom-fit").addEventListener("click", () => setZoom("fit"));
$("zoom-actual").addEventListener("click", () => setZoom(1));
$("schematic-image").addEventListener("load", () => setZoom("fit", false));
$("image-stage").addEventListener("keydown", (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const zoomKeys = {
    "+": state.zoom * 1.35,
    "=": state.zoom * 1.35,
    "-": state.zoom / 1.35,
    0: "fit",
    1: 1,
  };
  if (Object.hasOwn(zoomKeys, event.key)) {
    event.preventDefault();
    setZoom(zoomKeys[event.key]);
  }
});
new ResizeObserver(() => {
  if (state.fit) setZoom("fit", false);
}).observe($("image-stage"));
function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  const text = await response.text();
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(
      `The local server returned ${response.status}. Please retry when it is available.`,
    );
  }
  if (!response.ok)
    throw new Error(
      value.error?.message || `Request failed (${response.status}).`,
    );
  return value;
}
function controls() {
  const ready = Boolean(
    state.example &&
    state.image &&
    state.health?.loaded &&
    !state.health.busy &&
    !state.pending,
  );
  $("question").disabled = !ready;
  $("send").disabled = !ready;
  $("example").disabled = state.pending || !$("example").options.length;
  $("mode").disabled = state.pending;
  $("reset").disabled = state.pending || !state.history.length;
  for (const button of $("suggestions").children) button.disabled = !ready;
  let status;
  if (state.pending) status = "Local model is thinking…";
  else if (!state.example || !state.image) status = "Waiting for a schematic";
  else if (!state.health) status = "Waiting for local server";
  else if (!state.health.loaded) status = "Waiting for model to load";
  else if (state.health.busy) status = "Model is busy · waiting";
  else status = state.notice || "Ready for your question";
  $("chat-status").textContent = status;
}
async function health() {
  try {
    state.health = await api("/health");
    const available = state.health.loaded;
    $("model-status").textContent =
      `${state.health.model} · ${state.health.busy ? "generating" : available ? "ready locally" : "model not loaded"}`;
    $("model-status").classList.toggle(
      "ready",
      available && !state.health.busy,
    );
  } catch {
    state.health = null;
    $("model-status").textContent = "Local model unavailable";
    $("model-status").classList.remove("ready");
  }
  controls();
}
function reset() {
  state.history = [];
  state.notice = "";
  $("question").value = "";
  $("conversation").replaceChildren();
  const note = document.createElement("p");
  note.className = "empty-state";
  note.textContent =
    "Ask about a visible component or connection. Answers will appear here.";
  $("conversation").append(note);
  $("mode-note").textContent =
    $("mode").value === "evidence"
      ? "The model receives source-extracted values and connections with the image."
      : "The model receives only the image. Unreadable labels and hidden facts should remain uncertain.";
  showError("");
  controls();
}
function message(role, text) {
  $("conversation").querySelector(".empty-state")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "speaker";
  label.textContent =
    role === "user" ? "You" : state.health?.model || "Local model";
  const content = document.createElement("p");
  content.className = "text";
  content.textContent = text;
  article.append(label, content);
  $("conversation").append(article);
  article.scrollIntoView({ block: "nearest" });
  return article;
}
async function asDataURL(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () =>
      reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(blob);
  });
}
async function selectExample() {
  const selection = ++state.selection;
  state.example = null;
  state.image = null;
  for (const id of ["zoom-in", "zoom-out", "zoom-fit", "zoom-actual"])
    $(id).disabled = true;
  $("zoom-level").textContent = "Fit";
  reset();
  $("schematic-image").hidden = true;
  $("image-link").hidden = true;
  $("source").hidden = true;
  $("image-placeholder").hidden = false;
  $("image-placeholder").textContent = "Loading verified schematic…";
  $("suggestions").replaceChildren();
  try {
    const example = await api(
      `/api/examples/${encodeURIComponent($("example").value)}`,
    );
    const response = await fetch(example.image_url, { cache: "no-store" });
    if (!response.ok)
      throw new Error(
        "The example image is missing or failed its integrity check.",
      );
    const data = await asDataURL(await response.blob());
    if (selection !== state.selection) return;
    state.example = example;
    state.image = data;
    $("drawing-heading").textContent = example.title;
    $("sheet-label").textContent = `Sheet ${example.page}`;
    $("schematic-image").src = data;
    $("schematic-image").alt =
      `${example.title}, schematic sheet ${example.page}`;
    $("schematic-image").hidden = false;
    $("image-link").href = example.image_url;
    $("image-link").hidden = false;
    $("image-placeholder").hidden = true;
    $("source-link").href = example.source_url;
    $("source-meta").textContent =
      `${example.attribution} · ${example.license} · ${example.revision.slice(0, 8)}`;
    $("source").hidden = false;
    for (const question of example.suggested_questions) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = question;
      button.addEventListener("click", () => {
        $("question").value = question;
        $("question").focus();
      });
      $("suggestions").append(button);
    }
    controls();
  } catch (error) {
    if (selection === state.selection) {
      $("image-placeholder").textContent = "Schematic unavailable";
      showError(error.message);
      controls();
    }
  }
}
$("chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = $("question").value.trim();
  if (
    !question ||
    state.pending ||
    !state.example ||
    !state.image ||
    !state.health?.loaded ||
    state.health.busy
  )
    return;
  if (state.history.length >= 60) {
    showError(
      "This conversation has reached its turn limit. Start a new chat to continue.",
    );
    return;
  }
  state.pending = true;
  controls();
  showError("");
  const pendingHistory = [...state.history];
  if (!pendingHistory.length) {
    const system =
      $("mode").value === "evidence"
        ? "Answer factual questions using only the attached schematic image and supplied native source evidence. Identify connections as REFDES.PAD using physical package pads. Do not invent specifications, measurements, or safety findings."
        : "Answer only from the attached schematic image. If a label, connection, part, or measurement is not visible, say so. Do not invent facts.";
    pendingHistory.push({ role: "system", content: system });
    let text = question;
    if ($("mode").value === "evidence")
      text = `Extracted native schematic evidence (not an electrical review):\n${JSON.stringify(state.example.source_evidence)}\n\n${question}`;
    pendingHistory.push({
      role: "user",
      content: [
        { type: "image_url", image_url: { url: state.image } },
        { type: "text", text },
      ],
    });
  } else pendingHistory.push({ role: "user", content: question });
  const bubble = message("user", question);
  try {
    const result = await api("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.health.model,
        messages: pendingHistory,
      }),
    });
    const answer = result.choices?.[0]?.message?.content;
    if (typeof answer !== "string" || !answer.trim())
      throw new Error(
        "The model did not return a complete answer. Your question has been kept for retry.",
      );
    state.history = [...pendingHistory, { role: "assistant", content: answer }];
    message("assistant", answer);
    $("question").value = "";
    state.notice = "Answer received · continue the conversation";
  } catch (error) {
    bubble.remove();
    showError(error.message);
    state.notice = "Question kept · retry when ready";
  } finally {
    state.pending = false;
    await health();
    $("question").focus();
  }
});
$("example").addEventListener("change", selectExample);
$("mode").addEventListener("change", reset);
$("reset").addEventListener("click", reset);
async function init() {
  await health();
  try {
    const catalog = await api("/api/examples");
    $("example").replaceChildren();
    if (!catalog.examples.length)
      throw new Error(
        "No example corpus is configured. Start the server with --examples data/real after building the corpus.",
      );
    for (const example of catalog.examples) {
      const option = document.createElement("option");
      option.value = example.id;
      option.textContent = `${example.title} · sheet ${example.page}`;
      $("example").append(option);
    }
    controls();
    await selectExample();
  } catch (error) {
    showError(error.message);
    $("image-placeholder").textContent = "No examples available";
    controls();
  }
  window.setInterval(health, 5000);
}
init();
