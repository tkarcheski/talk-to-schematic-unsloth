"use strict";
/* Talk to the Schematic: a schematic viewer the local model can drive.
 * The model receives the published viewer functions (/api/view-tools) and
 * may answer with tool calls. Each call moves this viewer, and its result is
 * returned to the model before it answers. Coordinates are 0-1000 relative
 * to the current page image, [x1, y1, x2, y2].
 */
const $ = (id) => document.getElementById(id);
const MAX_TOOL_ROUNDS = 4;
const MAX_HISTORY = 60;
const PDF_LONG_SIDE = 2400;
const PDF_MAX_PIXELS = 12_000_000;
const state = {
  doc: null,
  history: [],
  health: null,
  healthRequest: 0,
  tools: null,
  pending: false,
  selection: 0,
  notice: "",
  catalog: [],
  pdf: null,
  view: { scale: 1, x: 0, y: 0 },
  fit: true,
  drag: null,
  pilotTimer: 0,
  preferredMode: "evidence",
};

/* ---------- geometry ---------- */
function clampBox(box) {
  return box.map((value) => Math.max(0, Math.min(1000, Math.round(value))));
}
/* Fit leaves room for the source card above and the controls below. */
const FIT_INSETS = { top: 112, right: 24, bottom: 64, left: 24 };
function fitView(stageWidth, stageHeight, width, height) {
  const inner = [
    stageWidth - FIT_INSETS.left - FIT_INSETS.right,
    stageHeight - FIT_INSETS.top - FIT_INSETS.bottom,
  ];
  const scale = Math.max(0.02, Math.min(inner[0] / width, inner[1] / height));
  return {
    scale,
    x: FIT_INSETS.left + (inner[0] - width * scale) / 2,
    y: FIT_INSETS.top + (inner[1] - height * scale) / 2,
  };
}
/** View that centers a normalized box with margin, never zooming past 4x. */
function viewForBox(box, stageWidth, stageHeight, width, height) {
  const [x1, y1, x2, y2] = box;
  const boxWidth = Math.max(((x2 - x1) / 1000) * width, width * 0.08);
  const boxHeight = Math.max(((y2 - y1) / 1000) * height, height * 0.08);
  const fit = fitView(stageWidth, stageHeight, width, height).scale;
  const scale = Math.max(
    fit,
    Math.min(4, (stageWidth * 0.5) / boxWidth, (stageHeight * 0.5) / boxHeight),
  );
  const centerX = (((x1 + x2) / 2) * width) / 1000;
  const centerY = (((y1 + y2) / 2) * height) / 1000;
  return {
    scale,
    x: stageWidth / 2 - centerX * scale,
    y: stageHeight / 2 - centerY * scale,
  };
}
function multiply(a, b) {
  return [
    a[0] * b[0] + a[2] * b[1],
    a[1] * b[0] + a[3] * b[1],
    a[0] * b[2] + a[2] * b[3],
    a[1] * b[2] + a[3] * b[3],
    a[0] * b[4] + a[2] * b[5] + a[4],
    a[1] * b[4] + a[3] * b[5] + a[5],
  ];
}
/** Split PDF text items into word boxes in 0-1000 page-image coordinates. */
function textWords(items, viewportTransform, scale, width, height) {
  const words = [];
  for (const item of items) {
    if (typeof item.str !== "string" || !item.str.trim()) continue;
    const m = multiply(viewportTransform, item.transform);
    const along = Math.hypot(m[0], m[1]) || 1;
    const direction = [m[0] / along, m[1] / along];
    const up = [m[2], m[3]];
    const total = item.width * scale;
    const length = item.str.length || 1;
    const pattern = /\S+/gu;
    let match;
    while ((match = pattern.exec(item.str))) {
      const start = (match.index / length) * total;
      const end = ((match.index + match[0].length) / length) * total;
      const origin = [m[4] + direction[0] * start, m[5] + direction[1] * start];
      const across = [
        direction[0] * (end - start),
        direction[1] * (end - start),
      ];
      const corners = [
        origin,
        [origin[0] + across[0], origin[1] + across[1]],
        [origin[0] + up[0], origin[1] + up[1]],
        [origin[0] + across[0] + up[0], origin[1] + across[1] + up[1]],
      ];
      const xs = corners.map((point) => point[0]);
      const ys = corners.map((point) => point[1]);
      words.push({
        text: match[0].replace(/[,;:]+$/u, ""),
        box: clampBox([
          (Math.min(...xs) / width) * 1000,
          (Math.min(...ys) / height) * 1000,
          (Math.max(...xs) / width) * 1000,
          (Math.max(...ys) / height) * 1000,
        ]),
      });
    }
  }
  return words;
}
/** Find a part, net or text label on the current page. */
function locateLabel(doc, label) {
  const wanted = label.trim().toUpperCase();
  const regions = doc?.regions;
  for (const kind of ["parts", "nets"]) {
    const table = regions?.[kind] || {};
    const name = Object.keys(table).find((key) => key.toUpperCase() === wanted);
    if (name)
      return {
        label: name,
        kind: kind === "parts" ? "part" : "net",
        box: table[name],
      };
  }
  const word = (doc?.words || []).find(
    (item) => item.text.toUpperCase() === wanted,
  );
  return word ? { label: word.text, kind: "text", box: word.box } : null;
}

/* ---------- viewer ---------- */
function applyView(view, animate) {
  state.view = view;
  const world = $("world");
  world.classList.toggle("flying", Boolean(animate));
  world.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
  world.style.setProperty("--scale", String(view.scale));
  $("zoom-level").textContent = `${Math.round(view.scale * 100)}%`;
}
function stageSize() {
  const stage = $("stage");
  return [stage.clientWidth || 800, stage.clientHeight || 600];
}
function showFull(animate = false) {
  if (!state.doc) return;
  state.fit = true;
  applyView(
    fitView(...stageSize(), state.doc.width, state.doc.height),
    animate,
  );
}
function zoomBy(factor, pointX, pointY) {
  if (!state.doc) return;
  const [width, height] = stageSize();
  const px = pointX ?? width / 2;
  const py = pointY ?? height / 2;
  const { scale, x, y } = state.view;
  const next = Math.max(0.02, Math.min(6, scale * factor));
  state.fit = false;
  applyView(
    {
      scale: next,
      x: px - (px - x) * (next / scale),
      y: py - (py - y) * (next / scale),
    },
    false,
  );
}
function mark(box, label) {
  const reticle = $("reticle");
  if (!box || !state.doc) {
    reticle.hidden = true;
    return;
  }
  const [x1, y1, x2, y2] = box;
  Object.assign(reticle.style, {
    left: `${(x1 / 1000) * state.doc.width}px`,
    top: `${(y1 / 1000) * state.doc.height}px`,
    width: `${((x2 - x1) / 1000) * state.doc.width}px`,
    height: `${((y2 - y1) / 1000) * state.doc.height}px`,
  });
  $("reticle-label").textContent = label || "";
  $("reticle-label").hidden = !label;
  reticle.hidden = false;
}
function pilot(text) {
  $("pilot-text").textContent = text;
  $("pilot").hidden = false;
  window.clearTimeout(state.pilotTimer);
  state.pilotTimer = window.setTimeout(() => {
    $("pilot").hidden = true;
  }, 2600);
}
/** Run one validated viewer call and return the result sent to the model. */
function executeCall(name, args) {
  const doc = state.doc;
  if (!doc) return { status: "error", reason: "no page is open" };
  state.fit = false;
  if (name === "show_full_sheet") {
    mark(null);
    showFull(true);
    pilot("AI · showing the full sheet");
    return { status: "shown", view: "full_sheet" };
  }
  if (name === "show_label") {
    const found = locateLabel(doc, String(args.label || ""));
    if (!found) {
      pilot(`AI · ${args.label} not found on this page`);
      return { status: "not_found", label: String(args.label || "") };
    }
    applyView(
      viewForBox(found.box, ...stageSize(), doc.width, doc.height),
      true,
    );
    mark(found.box, found.label);
    pilot(`AI · showing ${found.label}`);
    return { status: "shown", ...found };
  }
  if (name === "show_region") {
    const box = args.box;
    if (
      !Array.isArray(box) ||
      box.length !== 4 ||
      box.some(
        (value) => !Number.isInteger(value) || value < 0 || value > 1000,
      ) ||
      box[0] >= box[2] ||
      box[1] >= box[3]
    )
      return {
        status: "error",
        reason: "box must be four increasing integers 0-1000",
      };
    applyView(viewForBox(box, ...stageSize(), doc.width, doc.height), true);
    mark(box, args.label || "");
    pilot(`AI · showing ${args.label || "region"}`);
    return { status: "shown", box };
  }
  return { status: "error", reason: `unknown viewer function ${name}` };
}

/* ---------- server ---------- */
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
function ready() {
  return Boolean(
    state.doc &&
    state.tools &&
    state.health?.loaded &&
    !state.health.busy &&
    !state.pending,
  );
}
function controls() {
  const ok = ready();
  $("question").disabled = !ok;
  $("send").disabled = !ok;
  $("mode").disabled =
    state.pending || !state.doc || state.doc.kind !== "example";
  $("reset").disabled = state.pending || !state.history.length;
  $("open-library").disabled = state.pending;
  for (const id of ["zoom-in", "zoom-out", "zoom-fit"])
    $(id).disabled = !state.doc;
  for (const button of $("suggestions").children) button.disabled = !ok;
  for (const button of $("pages").children) button.disabled = state.pending;
  let status;
  if (state.pending) status = "Local model is working…";
  else if (!state.doc) status = "Waiting for a schematic";
  else if (!state.health) status = "Waiting for local server";
  else if (!state.health.loaded) status = "Waiting for model to load";
  else if (state.health.busy) status = "Model is busy · waiting";
  else if (!state.tools) status = "Viewer functions unavailable";
  else status = state.notice || "Ready";
  $("chat-status").textContent = status;
}
async function health() {
  const request = ++state.healthRequest;
  try {
    const result = await api("/health");
    if (request !== state.healthRequest) return;
    state.health = result;
    $("model-status").textContent =
      `${result.model} · ${result.busy ? "generating" : result.loaded ? "ready locally" : "model not loaded"}`;
    $("model-status").classList.toggle("ready", result.loaded && !result.busy);
  } catch {
    if (request !== state.healthRequest) return;
    state.health = null;
    $("model-status").textContent = "Local model unavailable";
    $("model-status").classList.remove("ready");
  }
  controls();
}
async function loadTools() {
  try {
    state.tools = await api("/api/view-tools");
  } catch {
    state.tools = null;
  }
  controls();
}

/* ---------- conversation ---------- */
function mode() {
  return state.doc?.kind === "example" ? $("mode").value : "vision";
}
function reset() {
  state.history = [];
  state.notice = "";
  $("question").value = "";
  const empty = document.createElement("div");
  empty.className = "empty-state";
  const heading = document.createElement("h3");
  heading.textContent = "Ask about what is on the page.";
  const body = document.createElement("p");
  body.textContent =
    "Read a value, trace a net, or ask the model to show you a part. It can pan and zoom the drawing.";
  empty.append(heading, body);
  $("conversation").replaceChildren(empty);
  const vision = mode() === "vision";
  $("gate-notice").hidden = !state.doc || !vision;
  $("mode-note").textContent = !state.doc
    ? ""
    : state.doc.kind !== "example"
      ? "Uploads have no native source evidence, so the model sees only the page image."
      : vision
        ? "The model sees only the page image."
        : "The model sees the image plus values and connections extracted from the EAGLE source.";
  mark(null);
  showError("");
  controls();
}
function message(role, text, actions = []) {
  $("conversation").querySelector(".empty-state")?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const speaker = document.createElement("div");
  speaker.className = "speaker";
  speaker.textContent =
    role === "user" ? "You" : state.health?.model || "Model";
  article.append(speaker);
  if (actions.length) {
    const row = document.createElement("div");
    row.className = "actions";
    for (const action of actions) row.append(actionChip(action));
    article.append(row);
  }
  if (text) {
    const content = document.createElement("p");
    content.className = "text";
    content.textContent = text;
    article.append(content);
  }
  $("conversation").append(article);
  article.scrollIntoView?.({ block: "nearest" });
  return article;
}
function actionChip(action) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = `action${action.result.status === "shown" ? "" : " missing"}`;
  const target =
    action.name === "show_full_sheet"
      ? "full sheet"
      : action.args.label || action.result.label || "region";
  chip.textContent =
    action.result.status === "shown" ? `◎ ${target}` : `✕ ${target} not found`;
  chip.title = `${action.name} ${JSON.stringify(action.args)}`;
  chip.addEventListener("click", () => {
    if (action.result.status !== "shown") return;
    if (action.name === "show_full_sheet") {
      mark(null);
      showFull(true);
    } else {
      const box = action.result.box;
      applyView(
        viewForBox(box, ...stageSize(), state.doc.width, state.doc.height),
        true,
      );
      mark(box, target);
    }
  });
  return chip;
}
function firstMessages(question) {
  const doc = state.doc;
  const vision = mode() === "vision";
  const base = vision
    ? "Answer only from the attached schematic image. If a part, label or measurement is not shown, say so. Do not invent facts."
    : "Answer factual questions using only the attached schematic image and supplied native source evidence. Identify connections as REFDES.PAD using physical package pads. Do not invent specifications, measurements, or safety findings.";
  let text = question;
  if (!vision && doc.evidence)
    text = `Extracted native schematic evidence (not an electrical review):\n${JSON.stringify(doc.evidence)}\n\n${question}`;
  return [
    { role: "system", content: `${base}\n\n${state.tools.instructions}` },
    {
      role: "user",
      content: [
        { type: "image_url", image_url: { url: doc.image } },
        { type: "text", text },
      ],
    },
  ];
}
/** One user question: model ↔ viewer rounds until a plain answer arrives. */
async function ask(question) {
  const selection = state.selection;
  const pending = state.history.length
    ? [...state.history, { role: "user", content: question }]
    : firstMessages(question);
  const actions = [];
  for (let round = 0; round <= MAX_TOOL_ROUNDS; round += 1) {
    const result = await api("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.health.model,
        messages: pending,
        tools: state.tools.tools,
      }),
    });
    if (selection !== state.selection) return null;
    const reply = result.choices?.[0]?.message;
    const calls = Array.isArray(reply?.tool_calls) ? reply.tool_calls : [];
    if (!calls.length) {
      if (typeof reply?.content !== "string" || !reply.content.trim())
        throw new Error(
          "The model did not return a complete answer. Your question has been kept for retry.",
        );
      pending.push({ role: "assistant", content: reply.content });
      return { history: pending, answer: reply.content, actions };
    }
    if (round === MAX_TOOL_ROUNDS) break;
    pending.push({
      role: "assistant",
      content: reply.content || null,
      tool_calls: calls,
    });
    for (const call of calls) {
      let args;
      try {
        args = JSON.parse(call.function.arguments);
      } catch {
        args = {};
      }
      const outcome = executeCall(call.function.name, args);
      actions.push({ name: call.function.name, args, result: outcome });
      pending.push({
        role: "tool",
        tool_call_id: call.id,
        content: JSON.stringify(outcome),
      });
    }
  }
  throw new Error(
    "The model kept moving the view without answering. Your question has been kept for retry.",
  );
}
async function submit(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question || !ready()) return;
  if (state.history.length >= MAX_HISTORY) {
    showError("This conversation is full. Start a new chat to continue.");
    return;
  }
  state.pending = true;
  showError("");
  controls();
  const bubble = message("user", question);
  try {
    const outcome = await ask(question);
    if (!outcome) return;
    state.history = outcome.history;
    message("assistant", outcome.answer, outcome.actions);
    $("question").value = "";
    state.notice = "Answer received";
  } catch (error) {
    bubble.remove();
    showError(error.message);
    state.notice = "Question kept · retry when ready";
  } finally {
    state.pending = false;
    await health();
    $("question").focus();
  }
}

/* ---------- documents ---------- */
async function asDataURL(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the file."));
    reader.readAsDataURL(blob);
  });
}
async function decode(data) {
  const image = $("sheet");
  image.src = data;
  try {
    await image.decode();
  } catch {
    throw new Error("The schematic image could not be decoded.");
  }
  if (!image.naturalWidth || !image.naturalHeight)
    throw new Error("The decoded image is empty.");
  return [image.naturalWidth, image.naturalHeight];
}
function present(doc) {
  state.doc = doc;
  // Uploads have no native evidence; examples restore the user's last choice.
  $("mode").value = doc.kind === "example" ? state.preferredMode : "vision";
  $("sheet").hidden = false;
  $("sheet").alt = `${doc.title}, schematic page ${doc.page}`;
  $("placeholder").hidden = true;
  $("doc-title").textContent = doc.title;
  $("source-title").textContent = doc.title;
  $("source-meta").textContent = doc.meta;
  $("source-link").hidden = !doc.sourceUrl;
  if (doc.sourceUrl) $("source-link").href = doc.sourceUrl;
  $("source").hidden = false;
  $("suggestions").replaceChildren();
  for (const question of doc.suggestions) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = question;
    button.addEventListener("click", () => {
      $("question").value = question;
      $("question").focus();
    });
    $("suggestions").append(button);
  }
  for (const button of $("library-list").querySelectorAll(".board"))
    button.setAttribute("aria-current", String(button.dataset.id === doc.id));
  reset();
  showFull(false);
}
function loading(text) {
  state.selection += 1;
  state.doc = null;
  $("sheet").hidden = true;
  $("source").hidden = true;
  $("placeholder").hidden = false;
  $("placeholder-text").textContent = text;
  mark(null);
  controls();
  return state.selection;
}
async function openExample(id) {
  const selection = loading("Loading verified schematic…");
  state.pdf = null;
  renderPages();
  showError("");
  try {
    const example = await api(`/api/examples/${encodeURIComponent(id)}`);
    const response = await fetch(example.image_url, { cache: "no-store" });
    if (!response.ok)
      throw new Error(
        "The example image is missing or failed its integrity check.",
      );
    const data = await asDataURL(await response.blob());
    if (selection !== state.selection) return;
    const [width, height] = await decode(data);
    if (selection !== state.selection) return;
    present({
      kind: "example",
      id: example.id,
      title: example.title,
      page: example.page,
      meta: `${example.attribution} · ${example.license} · ${example.revision.slice(0, 8)}`,
      sourceUrl: example.source_url,
      image: data,
      width,
      height,
      regions: example.regions,
      words: [],
      evidence: example.source_evidence,
      suggestions: example.suggested_questions,
    });
  } catch (error) {
    if (selection === state.selection) {
      $("placeholder-text").textContent = "Schematic unavailable";
      showError(error.message);
      controls();
    }
  }
}
let pdfLibrary = null;
async function pdfjs() {
  if (!pdfLibrary) {
    pdfLibrary = await import("/assets/vendor/pdf.min.mjs");
    pdfLibrary.GlobalWorkerOptions.workerSrc =
      "/assets/vendor/pdf.worker.min.mjs";
  }
  return pdfLibrary;
}
async function openFile(file) {
  if (!file) return;
  $("library").close?.();
  const isPdf = file.type === "application/pdf" || /\.pdf$/iu.test(file.name);
  if (
    !isPdf &&
    !["image/png", "image/jpeg", "image/webp"].includes(file.type)
  ) {
    showError("Choose a PDF, PNG, JPEG or WebP file.");
    return;
  }
  if (file.size > 40 * 1024 * 1024) {
    showError("Files over 40 MB are not supported.");
    return;
  }
  const selection = loading(`Opening ${file.name}…`);
  showError("");
  try {
    if (!isPdf) {
      state.pdf = null;
      renderPages();
      const data = await asDataURL(file);
      const [width, height] = await decode(data);
      if (selection !== state.selection) return;
      present(uploadDoc(file.name, 1, data, width, height, []));
      return;
    }
    const library = await pdfjs();
    const pdfDocument = await library.getDocument({
      data: new Uint8Array(await file.arrayBuffer()),
      isEvalSupported: false,
      enableXfa: false,
    }).promise;
    if (selection !== state.selection) return;
    state.pdf = {
      document: pdfDocument,
      name: file.name,
      pages: pdfDocument.numPages,
      page: 1,
    };
    await openPdfPage(1);
  } catch (error) {
    if (selection === state.selection) {
      $("placeholder-text").textContent = "Could not open that file";
      showError(error.message || "The PDF could not be read.");
      controls();
    }
  }
}
function uploadDoc(name, page, image, width, height, words) {
  const pages = state.pdf ? ` · page ${page} of ${state.pdf.pages}` : "";
  return {
    kind: state.pdf ? "pdf" : "image",
    id: `upload:${name}:${page}`,
    title: name,
    page,
    meta: `Your upload · stays on this computer${pages}`,
    sourceUrl: "",
    image,
    width,
    height,
    regions: null,
    words,
    evidence: null,
    suggestions: [
      "What is on this page?",
      "Show me U1.",
      "Zoom back out to the whole sheet.",
    ],
  };
}
async function openPdfPage(number) {
  const pdf = state.pdf;
  const selection = loading(`Rendering page ${number}…`);
  const page = await pdf.document.getPage(number);
  const base = page.getViewport({ scale: 1 });
  let scale = PDF_LONG_SIDE / Math.max(base.width, base.height);
  scale = Math.min(
    scale,
    Math.sqrt(PDF_MAX_PIXELS / (base.width * base.height)),
  );
  const viewport = page.getViewport({ scale });
  const canvas = document.createElement("canvas");
  canvas.width = Math.floor(viewport.width);
  canvas.height = Math.floor(viewport.height);
  const context = canvas.getContext("2d");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  await page.render({ canvas, canvasContext: context, viewport }).promise;
  const text = await page.getTextContent();
  if (selection !== state.selection) return;
  const words = textWords(
    text.items,
    viewport.transform,
    viewport.scale,
    canvas.width,
    canvas.height,
  );
  const data = canvas.toDataURL("image/png");
  const [width, height] = await decode(data);
  if (selection !== state.selection) return;
  pdf.page = number;
  renderPages();
  present(uploadDoc(pdf.name, number, data, width, height, words));
}
function renderPages() {
  const nav = $("pages");
  nav.replaceChildren();
  nav.hidden = !state.pdf || state.pdf.pages < 2;
  if (nav.hidden) return;
  for (let number = 1; number <= Math.min(state.pdf.pages, 200); number += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(number);
    button.setAttribute("aria-label", `Page ${number}`);
    if (number === state.pdf.page) button.setAttribute("aria-current", "page");
    button.addEventListener("click", () => {
      if (number !== state.pdf.page && !state.pending)
        openPdfPage(number).catch((error) => showError(error.message));
    });
    nav.append(button);
  }
}

/* ---------- library ---------- */
function renderLibrary() {
  const filter = $("library-search").value.trim().toLowerCase();
  const list = $("library-list");
  list.replaceChildren();
  const groups = new Map();
  for (const example of state.catalog) {
    if (filter && !example.title.toLowerCase().includes(filter)) continue;
    if (!groups.has(example.publisher)) groups.set(example.publisher, []);
    groups.get(example.publisher).push(example);
  }
  for (const [publisher, examples] of groups) {
    const heading = document.createElement("h3");
    heading.textContent = `${publisher} · ${examples.length} sheets`;
    if (publisher.startsWith("SparkFun")) {
      const badge = document.createElement("span");
      badge.className = "new";
      badge.textContent = "New";
      heading.append(badge);
    }
    const grid = document.createElement("div");
    grid.className = "board-grid";
    for (const example of examples) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "board";
      button.dataset.id = example.id;
      button.textContent =
        example.page > 1
          ? `${example.title} · sheet ${example.page}`
          : example.title;
      button.setAttribute("aria-current", String(state.doc?.id === example.id));
      button.addEventListener("click", () => {
        $("library").close?.();
        openExample(example.id);
      });
      grid.append(button);
    }
    list.append(heading, grid);
  }
  if (!groups.size) {
    const none = document.createElement("p");
    none.className = "empty-state";
    none.textContent = state.catalog.length
      ? "No boards match that search."
      : "No example corpus is configured.";
    list.append(none);
  }
}
async function loadCatalog() {
  $("retry-examples").hidden = true;
  showError("");
  try {
    const catalog = await api("/api/examples");
    state.catalog = catalog.examples;
    renderLibrary();
    if (!state.catalog.length)
      throw new Error(
        "No example corpus is configured. Start the server with --examples, or upload a PDF.",
      );
    await openExample(state.catalog[0].id);
  } catch (error) {
    $("retry-examples").hidden = false;
    $("placeholder-text").textContent = "Open a schematic to begin";
    showError(error.message);
    controls();
  }
}

/* ---------- input ---------- */
function bind() {
  $("chat-form").addEventListener("submit", submit);
  $("reset").addEventListener("click", reset);
  $("mode").addEventListener("change", () => {
    state.preferredMode = $("mode").value;
    reset();
  });
  $("open-library").addEventListener("click", () => {
    renderLibrary();
    $("library").showModal?.();
  });
  $("library-close").addEventListener("click", () => $("library").close?.());
  $("library-search").addEventListener("input", renderLibrary);
  $("retry-examples").addEventListener("click", loadCatalog);
  $("upload").addEventListener("change", (event) => {
    openFile(event.target.files?.[0]);
    event.target.value = "";
  });
  $("zoom-in").addEventListener("click", () => zoomBy(1.35));
  $("zoom-out").addEventListener("click", () => zoomBy(1 / 1.35));
  $("zoom-fit").addEventListener("click", () => {
    mark(null);
    showFull(true);
  });
  $("paper").addEventListener("click", () => {
    const paper = !document.body.classList.contains("paper");
    document.body.classList.toggle("paper", paper);
    $("paper").setAttribute("aria-pressed", String(paper));
  });
  const stage = $("stage");
  stage.addEventListener("pointerdown", (event) => {
    if (!state.doc || event.button !== 0) return;
    stage.setPointerCapture?.(event.pointerId);
    state.drag = {
      x: event.clientX,
      y: event.clientY,
      view: { ...state.view },
    };
    stage.classList.add("dragging");
    $("world").classList.remove("flying");
  });
  stage.addEventListener("pointermove", (event) => {
    if (!state.drag) return;
    state.fit = false;
    applyView(
      {
        ...state.drag.view,
        x: state.drag.view.x + event.clientX - state.drag.x,
        y: state.drag.view.y + event.clientY - state.drag.y,
      },
      false,
    );
  });
  const release = () => {
    state.drag = null;
    stage.classList.remove("dragging");
  };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  stage.addEventListener(
    "wheel",
    (event) => {
      if (!state.doc) return;
      event.preventDefault();
      const rect = stage.getBoundingClientRect();
      zoomBy(
        Math.exp(-event.deltaY * 0.0015),
        event.clientX - rect.left,
        event.clientY - rect.top,
      );
    },
    { passive: false },
  );
  stage.addEventListener("keydown", (event) => {
    if (!state.doc || event.ctrlKey || event.metaKey || event.altKey) return;
    const step = 60;
    const moves = {
      ArrowLeft: [step, 0],
      ArrowRight: [-step, 0],
      ArrowUp: [0, step],
      ArrowDown: [0, -step],
    };
    if (event.key === "+" || event.key === "=") zoomBy(1.35);
    else if (event.key === "-") zoomBy(1 / 1.35);
    else if (event.key === "0") showFull(true);
    else if (moves[event.key]) {
      const [dx, dy] = moves[event.key];
      applyView(
        { ...state.view, x: state.view.x + dx, y: state.view.y + dy },
        false,
      );
    } else return;
    event.preventDefault();
  });
  stage.addEventListener("dragover", (event) => {
    event.preventDefault();
    $("drop-hint").hidden = false;
  });
  stage.addEventListener("dragleave", () => {
    $("drop-hint").hidden = true;
  });
  stage.addEventListener("drop", (event) => {
    event.preventDefault();
    $("drop-hint").hidden = true;
    if (!state.pending) openFile(event.dataTransfer?.files?.[0]);
  });
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit?.();
    }
  });
  new ResizeObserver(() => {
    if (state.fit) showFull(false);
  }).observe(stage);
}
async function init() {
  bind();
  reset();
  await Promise.all([health(), loadTools()]);
  await loadCatalog();
  window.setInterval(health, 5000);
}
init();
