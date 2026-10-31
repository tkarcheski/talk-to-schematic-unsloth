/* Saved comparisons are local, session-only evidence review. */
"use strict";
const savedComparison = { pair: null, index: 0, filtered: [], pending: false };
function savedElement(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function savedDetails(title, text) {
  const details = savedElement("details");
  details.append(savedElement("summary", title), savedElement("pre", text));
  return details;
}
function savedMetricText(lane) {
  return Object.entries(lane.report.metrics)
    .map(
      ([metric, value]) =>
        `${metric}: ${(value * 100).toFixed(1)}% (n=${lane.report.counts[metric]})`,
    )
    .join(" · ");
}
function renderSavedCase() {
  const pair = savedComparison.pair,
    item = savedComparison.filtered[savedComparison.index];
  $("saved-case").replaceChildren();
  $("saved-prev").disabled = !item || savedComparison.index === 0;
  $("saved-next").disabled =
    !item || savedComparison.index === savedComparison.filtered.length - 1;
  $("saved-count").textContent =
    `${savedComparison.filtered.length} of ${pair.cases.length} turns match`;
  $("saved-case-select").value = String(savedComparison.index);
  $("saved-image-file").value = "";
  $("saved-image-error").textContent = "";
  $("saved-image-tools").hidden = !item || !item.imageRefs.length;
  $("saved-images").replaceChildren();
  if (!item) {
    $("saved-case").append(savedElement("p", "No turns match this filter."));
    return;
  }
  const heading = savedElement(
    "h3",
    `Conversation ${item.row + 1} · Turn ${item.turn + 1} · ${item.type}`,
  );
  const question = savedElement("p", item.question, "saved-question");
  $("saved-case").append(heading, question);
  const reference = savedDetails(
    "Dataset reference answer (not a new human review)",
    item.reference,
  );
  $("saved-case").append(reference);
  const lanes = savedElement("div", undefined, "comparison-lanes saved-lanes");
  for (const [index, lane] of pair.lanes.entries()) {
    const record = lane.records.get(item.key),
      failed = record.status === "failure";
    const card = savedElement("article", undefined, "comparison-card");
    card.dataset.lane = String(index);
    card.append(
      savedElement(
        "h4",
        `${index ? "B" : "A"} · ${pair.labels[index]} · ${failed ? (record.attempted ? "Generation failed" : "Blocked by earlier failure") : "Recorded success (stop)"}`,
      ),
    );
    card.append(
      savedElement(
        "p",
        failed
          ? "No scored answer. Partial output is not included in this benchmark record."
          : record.output,
        "comparison-answer",
      ),
    );
    const seconds = record.elapsed_seconds;
    const tokens = record.usage?.completion_tokens;
    const metrics = [
      `Finish: ${record.finish_reason}`,
      `Time: ${typeof seconds === "number" ? `${seconds.toFixed(2)}s` : "unavailable"}`,
      `Output tokens: ${tokens ?? "unavailable"}`,
      `Rate: ${typeof seconds === "number" && seconds > 0 && Number.isInteger(tokens) ? `${(tokens / seconds).toFixed(1)} tokens/s` : "unavailable"}`,
    ];
    if (failed) metrics.push(`Error: ${record.error.code}`);
    card.append(savedElement("p", metrics.join(" · "), "mode-note"));
    const history = item.history.slice(0, item.historyEnd).map((message) => {
      if (message.role !== "assistant" || pair.history === "gold")
        return `${message.role}: ${message.text}`;
      const prior = lane.records.get(message.key);
      return `assistant: ${prior.status === "success" ? prior.output : "[failed answer; dependent turns were not attempted]"}`;
    });
    card.append(
      savedDetails(
        pair.history === "gold"
          ? "Prior context: identical dataset reference history"
          : "Prior context: this run's generated answers",
        history.join("\n\n") || "First question: no preceding conversation.",
      ),
    );
    lanes.append(card);
  }
  $("saved-case").append(lanes);
  const hashes = [
    ...new Set(item.imageRefs.map((ref) => pair.imageHashes[ref])),
  ];
  $("saved-image-note").textContent =
    `${pair.images.size} source previews loaded. Select original PNGs for this dataset once to browse matching images across questions. Only matching SHA-256 images are shown; no source paths are fetched.`;
  $("saved-clear-images").disabled =
    savedComparison.imagePending || !pair.images.size;
  for (const fingerprint of hashes) {
    const image = pair.images.get(fingerprint);
    if (image) {
      const img = savedElement("img");
      img.src = image.url;
      img.alt = `Verified source image for conversation ${item.row + 1}`;
      $("saved-images").append(img);
    } else
      $("saved-images").append(
        savedElement(
          "p",
          `Image ${fingerprint.slice(0, 12)}… is not loaded; source pixels have not been verified in this session.`,
          "mode-note",
        ),
      );
  }
}
function filterSavedCases() {
  if (!savedComparison.pair) return;
  const pair = savedComparison.pair,
    filter = $("saved-filter").value;
  savedComparison.filtered = pair.cases.filter((item) => {
    const [a, b] = pair.lanes.map((lane) => lane.records.get(item.key));
    if (filter === "failure")
      return a.status === "failure" || b.status === "failure";
    if (filter === "b-failure")
      return a.status === "success" && b.status === "failure";
    if (filter === "a-failure")
      return a.status === "failure" && b.status === "success";
    if (filter === "different")
      return (
        a.status === "success" &&
        b.status === "success" &&
        a.output !== b.output
      );
    return true;
  });
  savedComparison.index = 0;
  $("saved-case-select").replaceChildren();
  for (const [index, item] of savedComparison.filtered.entries()) {
    const option = savedElement(
      "option",
      `C${item.row + 1} · T${item.turn + 1}: ${item.question.slice(0, 90)}`,
    );
    option.value = String(index);
    $("saved-case-select").append(option);
  }
  renderSavedCase();
}
function renderSavedPair() {
  const pair = savedComparison.pair;
  $("saved-review").hidden = false;
  $("saved-image-status").textContent = "";
  $("saved-summary").replaceChildren();
  $("saved-binding").textContent =
    `${pair.cases.length} turns · ${pair.conversations} conversations · ${pair.profile} · ${pair.history === "gold" ? "identical reference history" : "independent generated histories"}. Dataset and prediction hashes match their reports; every expected key is present.`;
  $("saved-warning").textContent = pair.warnings.join(" ");
  for (const [index, lane] of pair.lanes.entries()) {
    const card = savedElement("article", undefined, "comparison-card");
    const fingerprint = lane.identity.adapter || lane.identity.artifact;
    card.append(
      savedElement("h3", `${index ? "B" : "A"} · ${pair.labels[index]}`),
    );
    card.append(
      savedElement(
        "p",
        `Operator label · recorded ${lane.identity.adapter ? "adapter" : "artifact"} ${fingerprint.slice(0, 12)}…`,
        "mode-note",
      ),
    );
    card.append(
      savedElement(
        "p",
        `${lane.report.successful_turns}/${lane.report.turns} recorded stop · ${lane.report.failed_turns} failed or blocked`,
      ),
    );
    card.append(savedElement("p", savedMetricText(lane), "mode-note"));
    card.append(
      savedDetails(
        "Recorded identity and settings",
        JSON.stringify(
          {
            identity: lane.identity,
            dataset_sha256: pair.datasetHash,
            profile: pair.profile,
            history: pair.history,
            temperature: lane.meta.temperature,
            ...lane.settings,
            request_admission_limits: Object.fromEntries(
              Object.entries(lane.admission).map(([key, value]) => [
                key,
                value ?? "unavailable",
              ]),
            ),
          },
          null,
          2,
        ),
      ),
    );
    $("saved-summary").append(card);
  }
  filterSavedCases();
}
async function importSavedPair(event) {
  event.preventDefault();
  if (savedComparison.pending) return;
  savedComparison.pending = true;
  $("saved-error").textContent = "";
  $("saved-import-status").textContent =
    "Checking file integrity and exact turn coverage…";
  for (const element of $("saved-import").elements) element.disabled = true;
  $("saved-close").disabled = true;
  try {
    const pair = await savedBenchmark.pair($("saved-dataset").files[0], [
      [...$("saved-run-a").files],
      [...$("saved-run-b").files],
    ]);
    pair.labels = ["a", "b"].map(
      (lane) =>
        $("saved-label-" + lane)
          .value.trim()
          .slice(0, 40) || (lane === "a" ? "Run A" : "Run B"),
    );
    savedComparison.pair = pair;
    renderSavedPair();
    $("saved-files").open = false;
    $("saved-filter").focus();
    $("saved-import-status").textContent =
      "Imported locally. No files were uploaded and no models were called.";
  } catch (error) {
    $("saved-error").textContent =
      error instanceof Error
        ? error.message
        : "Could not import the selected files.";
    $("saved-import-status").textContent =
      "Import failed. The previous valid comparison remains available.";
  } finally {
    savedComparison.pending = false;
    for (const element of $("saved-import").elements) element.disabled = false;
    $("saved-close").disabled = false;
  }
}
$("compare-saved").addEventListener("click", () => {
  if (comparison.pending) return;
  $("comparison").close();
  $("saved-comparison").showModal();
});
$("saved-close").addEventListener("click", () => $("saved-comparison").close());
$("saved-comparison").addEventListener("close", () =>
  $("open-comparison").focus(),
);
$("saved-comparison").addEventListener("cancel", (event) => {
  if (savedComparison.pending) event.preventDefault();
});
$("saved-import").addEventListener("submit", importSavedPair);
$("saved-filter").addEventListener("change", filterSavedCases);
$("saved-case-select").addEventListener("change", () => {
  savedComparison.index = Number($("saved-case-select").value);
  renderSavedCase();
});
for (const [id, delta] of [
  ["saved-prev", -1],
  ["saved-next", 1],
])
  $(id).addEventListener("click", () => {
    savedComparison.index += delta;
    renderSavedCase();
  });
$("saved-clear-images").addEventListener("click", () => {
  if (!savedComparison.pair || savedComparison.imagePending) return;
  savedComparison.pair.images.clear();
  $("saved-image-status").textContent = "Source previews cleared.";
  renderSavedCase();
});
$("saved-image-file").addEventListener("change", async () => {
  const pair = savedComparison.pair;
  if (!pair || savedComparison.imagePending) return;
  savedComparison.imagePending = true;
  $("saved-image-error").textContent = "";
  $("saved-image-status").textContent = "Checking selected image hashes…";
  $("saved-image-file").disabled = true;
  $("saved-clear-images").disabled = true;
  try {
    const images = await savedBenchmark.pngBatch(
      [...$("saved-image-file").files],
      Object.values(pair.imageHashes),
      pair.images,
    );
    // Byte/hash checks cannot establish that the browser can display the PNG.
    // Decode the entire batch before changing the last valid preview cache.
    for (const [hash, image] of images) {
      if (pair.images.get(hash)?.url === image.url) continue;
      const preview = new Image();
      preview.src = image.url;
      try {
        await preview.decode();
      } catch {
        throw new Error(
          "Selected PNG could not be decoded as a source preview.",
        );
      }
      if (
        !preview.naturalWidth ||
        !preview.naturalHeight ||
        Math.max(preview.naturalWidth, preview.naturalHeight) > 32768 ||
        preview.naturalWidth * preview.naturalHeight !== image.pixels
      )
        throw new Error(
          "Decoded PNG dimensions differ from its bounded header.",
        );
    }
    if (savedComparison.pair !== pair) return;
    pair.images = images;
    renderSavedCase();
    $("saved-image-status").textContent =
      `${images.size} matching source previews loaded locally.`;
  } catch (error) {
    if (savedComparison.pair === pair) {
      $("saved-image-error").textContent = error.message;
      $("saved-image-status").textContent =
        "Image batch rejected. Existing previews are unchanged.";
    }
  } finally {
    savedComparison.imagePending = false;
    $("saved-image-file").disabled = false;
    $("saved-clear-images").disabled = !savedComparison.pair?.images.size;
  }
});
