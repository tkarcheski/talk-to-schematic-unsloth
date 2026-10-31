/* Local benchmark import: no network, model calls, storage or path resolution. */
"use strict";
// cspell:words IHDR
const savedBenchmark = (() => {
  const fail = (message) => {
    throw new Error(message);
  };
  const ensure = (condition, message) => {
    if (!condition) fail(message);
  };
  const object = (value) =>
    value !== null && typeof value === "object" && !Array.isArray(value);
  const hash = (value) =>
    typeof value === "string" && /^[a-f0-9]{64}$/u.test(value);
  const admissionBounds = {
    max_images: [1, 32],
    max_text_chars: [1, 4000000],
    max_messages: [1, 512],
  };
  const canonical = (value) =>
    JSON.stringify(value, function (_, item) {
      return object(item)
        ? Object.fromEntries(
            Object.keys(item)
              .sort()
              .map((key) => [key, item[key]]),
          )
        : item;
    });
  const equal = (left, right) => canonical(left) === canonical(right);

  // JSON.parse alone silently accepts repeated keys. Reject them, excessive
  // nesting and non-finite numbers before accepting any imported evidence.
  function parse(text) {
    let cursor = 0,
      nodes = 0;
    const bad = () =>
      fail("Invalid JSON, duplicate field, or excessive nesting.");
    const space = () => {
      while (/[\t\r\n ]/u.test(text[cursor] || "x")) cursor++;
    };
    function string() {
      const start = cursor++;
      while (cursor < text.length) {
        const char = text[cursor++];
        if (char === "\\") cursor++;
        else if (char === '"') {
          try {
            return JSON.parse(text.slice(start, cursor));
          } catch {
            bad();
          }
        }
      }
      bad();
    }
    function value(depth) {
      if (depth > 40 || ++nodes > 100000) bad();
      space();
      const char = text[cursor];
      if (char === '"') return string();
      if (char === "{" || char === "[") {
        const isObject = char === "{",
          end = isObject ? "}" : "]";
        const result = isObject ? Object.create(null) : [],
          keys = new Set();
        cursor++;
        space();
        if (text[cursor] === end) {
          cursor++;
          return result;
        }
        while (cursor < text.length) {
          space();
          let key;
          if (isObject) {
            if (text[cursor] !== '"') bad();
            key = string();
            if (keys.has(key)) bad();
            keys.add(key);
            space();
            if (text[cursor++] !== ":") bad();
          }
          const item = value(depth + 1);
          if (isObject) result[key] = item;
          else result.push(item);
          space();
          const next = text[cursor++];
          if (next === end) return result;
          if (next !== ",") bad();
        }
        bad();
      }
      const token =
        /^(?:null|true|false|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/u.exec(
          text.slice(cursor),
        );
      if (!token) bad();
      cursor += token[0].length;
      const result = JSON.parse(token[0]);
      if (typeof result === "number" && !Number.isFinite(result)) bad();
      return result;
    }
    const result = value(0);
    space();
    if (cursor !== text.length) bad();
    return result;
  }
  function jsonl(text) {
    const lines = text.trimEnd().split("\n");
    ensure(
      lines.length > 0 && lines.length <= 5000,
      "Expected 1–5000 JSONL records.",
    );
    return lines.map((line) => {
      ensure(
        line.length <= 1000000,
        "A JSONL record exceeds the import limit.",
      );
      const result = parse(line);
      ensure(object(result), "JSONL records must be objects.");
      return result;
    });
  }
  async function digest(bytes) {
    ensure(
      globalThis.crypto?.subtle,
      "Integrity checks require HTTPS or a localhost address.",
    );
    return Array.from(
      new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
      (item) => item.toString(16).padStart(2, "0"),
    ).join("");
  }
  async function read(file, limit = 16 * 1024 * 1024) {
    ensure(
      file && file.size > 0 && file.size <= limit,
      "Select a nonempty file of at most 16 MiB.",
    );
    const bytes = await file.arrayBuffer();
    let text;
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      fail("Import files must contain valid UTF-8 text.");
    }
    return { text, hash: await digest(bytes) };
  }
  function content(message) {
    if (typeof message.content === "string")
      return { text: message.content, images: [] };
    ensure(
      Array.isArray(message.content),
      "Unsupported dataset message content.",
    );
    const texts = [],
      images = [];
    for (const block of message.content) {
      ensure(object(block), "Invalid dataset content block.");
      if (block.type === "text" && typeof block.text === "string")
        texts.push(block.text);
      else if (
        block.type === "image" &&
        typeof block.image === "string" &&
        block.image
      )
        images.push(block.image);
      else
        fail(
          "Only text and local image references are supported in saved datasets.",
        );
    }
    return { text: texts.join("\n"), images };
  }
  function dataset(text) {
    const rows = jsonl(text),
      cases = [],
      ids = new Set(),
      imageRefs = new Set();
    for (const [rowIndex, row] of rows.entries()) {
      ensure(
        typeof row.id === "string" &&
          row.id.trim() === row.id &&
          row.id &&
          !row.id.includes("#") &&
          !ids.has(row.id),
        "Dataset IDs must be nonempty and unique, without '#'.",
      );
      ids.add(row.id);
      ensure(
        Array.isArray(row.gold) &&
          row.gold.length &&
          row.gold.length <= 64 &&
          Array.isArray(row.messages),
        "Dataset needs reference labels and full conversations.",
      );
      const messages = row.messages;
      const start = messages[0]?.role === "system" ? 1 : 0;
      ensure(
        messages.length === start + row.gold.length * 2,
        "Dataset conversation and label counts disagree.",
      );
      const history = [],
        images = [];
      if (start) {
        const item = content(messages[0]);
        ensure(!item.images.length, "System images are unsupported.");
        history.push({ role: "system", text: item.text });
      }
      for (let turn = 0; turn < row.gold.length; turn++) {
        const user = messages[start + turn * 2],
          assistant = messages[start + turn * 2 + 1],
          label = row.gold[turn];
        ensure(
          user?.role === "user" &&
            assistant?.role === "assistant" &&
            object(label) &&
            typeof label.type === "string" &&
            /^[a-z_]{1,40}$/u.test(label.type),
          "Dataset turns must alternate user and assistant with typed labels.",
        );
        const question = content(user),
          reference = content(assistant);
        ensure(
          question.text.trim() &&
            reference.text.trim() &&
            !reference.images.length,
          "Each turn needs a text question and text reference answer.",
        );
        for (const ref of question.images) {
          imageRefs.add(ref);
          if (!images.includes(ref)) images.push(ref);
        }
        const item = {
          key: `${row.id}#${turn}`,
          row: rowIndex,
          turn,
          type: label.type,
          question: question.text,
          reference: reference.text,
          history,
          historyEnd: history.length,
          imageRefs: [...images],
        };
        cases.push(item);
        history.push(
          { role: "user", text: question.text },
          { role: "assistant", text: reference.text, key: item.key },
        );
      }
    }
    ensure(cases.length <= 5000, "Dataset exceeds 5000 turns.");
    return { cases, conversations: rows.length, imageRefs };
  }
  function bundleFiles(files) {
    ensure(
      files.length === 3,
      "For each run, choose predictions JSONL, its .meta.json and its .report.json together.",
    );
    const predictions = files.filter((file) => file.name.endsWith(".jsonl"));
    ensure(
      predictions.length === 1,
      "Each run needs exactly one predictions JSONL file.",
    );
    const meta = files.find(
      (file) => file.name === predictions[0].name + ".meta.json",
    );
    const report = files.find(
      (file) => file.name === predictions[0].name + ".report.json",
    );
    ensure(
      meta && report,
      "Run sidecar filenames must match the predictions filename.",
    );
    return [predictions[0], meta, report];
  }
  async function run(files, data, dataHash) {
    const inputs = await Promise.all(
      bundleFiles(files).map((file) => read(file)),
    );
    const records = new Map(),
      meta = parse(inputs[1].text),
      report = parse(inputs[2].text);
    ensure(
      object(meta) &&
        object(report) &&
        meta.schema_version === 1 &&
        report.schema_version === 1 &&
        meta.runner === "failure-accounted-benchmark",
      "Only version 1 failure-accounted benchmark runs are supported.",
    );
    ensure(
      meta.dataset_sha256 === dataHash &&
        report.predictions_sha256 === inputs[0].hash &&
        equal(meta, report.provenance),
      "Dataset, predictions, metadata or report hashes do not match.",
    );
    ensure(
      ["gold", "generated"].includes(meta.history) &&
        typeof meta.profile === "string" &&
        /^[a-z0-9-]{1,50}$/u.test(meta.profile) &&
        report.profile === meta.profile &&
        Number.isInteger(meta.max_tokens) &&
        meta.max_tokens > 0 &&
        Number.isFinite(meta.temperature),
      "Invalid benchmark settings or history mode.",
    );
    ensure(
      object(meta.images_sha256) &&
        equal(
          Object.keys(meta.images_sha256).sort(),
          [...data.imageRefs].sort(),
        ) &&
        Object.values(meta.images_sha256).every(hash),
      "Image hash coverage does not match the dataset.",
    );
    const deployment = meta.deployment;
    ensure(
      object(deployment) &&
        object(deployment.model) &&
        hash(deployment.model.sha256) &&
        object(deployment.settings) &&
        object(deployment.packages),
      "Benchmark model identity or settings are missing.",
    );
    for (const record of jsonl(inputs[0].text)) {
      ensure(
        typeof record.key === "string" &&
          !records.has(record.key) &&
          typeof record.attempted === "boolean" &&
          typeof record.output === "string",
        "Duplicate or malformed prediction record.",
      );
      if (record.status === "success")
        ensure(
          record.attempted &&
            record.error == null &&
            record.finish_reason === "stop" &&
            record.output.trim(),
          "Successful records must contain complete attempted answers.",
        );
      else
        ensure(
          record.status === "failure" &&
            record.output === "" &&
            object(record.error) &&
            ["generation_failed", "dependency_failed"].includes(
              record.error.code,
            ) &&
            record.finish_reason ===
              (record.attempted ? "error" : "dependency_failed") &&
            record.error.code ===
              (record.attempted ? "generation_failed" : "dependency_failed"),
          "Failed or partial answers must have empty scored output and an explicit failure status.",
        );
      if (record.elapsed_seconds !== undefined)
        ensure(
          Number.isFinite(record.elapsed_seconds) &&
            record.elapsed_seconds >= 0,
          "Invalid recorded inference time.",
        );
      if (record.usage !== undefined && record.usage !== null)
        ensure(
          object(record.usage) &&
            Object.values(record.usage).every(
              (value) => Number.isInteger(value) && value >= 0,
            ),
          "Invalid recorded token usage.",
        );
      if (record.model_fingerprint !== undefined)
        ensure(
          hash(record.model_fingerprint),
          "Invalid recorded model fingerprint.",
        );
      records.set(record.key, record);
    }
    ensure(
      records.size === data.cases.length &&
        data.cases.every((item) => records.has(item.key)),
      "Prediction keys must exactly cover every dataset turn.",
    );
    let blockedBy = null,
      previousRow = null;
    for (const item of data.cases) {
      if (previousRow !== item.row) blockedBy = null;
      previousRow = item.row;
      const record = records.get(item.key);
      if (meta.history === "generated" && blockedBy)
        ensure(
          !record.attempted && record.error?.caused_by === blockedBy,
          "Generated history contradicts an earlier failed answer.",
        );
      else
        ensure(
          record.attempted,
          "A turn is blocked without a prior generated-history failure.",
        );
      if (record.status === "failure") blockedBy ||= item.key;
    }
    const values = [...records.values()],
      failed = values.filter((record) => record.status === "failure");
    ensure(
      report.turns === records.size &&
        report.conversations === data.conversations &&
        report.attempted_turns ===
          values.filter((record) => record.attempted).length &&
        report.failed_turns === failed.length &&
        report.successful_turns === records.size - failed.length &&
        report.failure_rate === failed.length / records.size,
      "Report totals disagree with prediction records.",
    );
    ensure(
      object(report.metrics) &&
        object(report.counts) &&
        equal(
          Object.keys(report.metrics).sort(),
          Object.keys(report.counts).sort(),
        ) &&
        Object.entries(report.metrics).every(
          ([key, value]) =>
            /^[a-z0-9_.-]{1,60}$/u.test(key) &&
            Number.isFinite(value) &&
            value >= 0 &&
            value <= 1 &&
            Number.isInteger(report.counts[key]) &&
            report.counts[key] > 0 &&
            report.counts[key] <= records.size,
        ),
      "Invalid recorded aggregate scores.",
    );
    const fingerprints = new Set(
      values.map((record) => record.model_fingerprint).filter(Boolean),
    );
    ensure(
      fingerprints.size <= 1,
      "One run contains conflicting model fingerprints.",
    );
    const filesList = deployment.model.files;
    ensure(Array.isArray(filesList), "Model file identity is missing.");
    const adapter = filesList.find(
      (file) => file.name === "adapter_model.safetensors",
    )?.sha256;
    ensure(adapter === undefined || hash(adapter), "Invalid adapter identity.");
    const identity = {
      artifact: deployment.model.sha256,
      adapter,
      base: deployment.model.base_model?.sha256,
      response: [...fingerprints][0],
    };
    ensure(
      identity.base === undefined || hash(identity.base),
      "Invalid base model identity.",
    );
    // Only allowlisted fields reach the view. Metadata paths and raw errors do not.
    const settings = Object.fromEntries(
      ["max_seq", "max_image_size", "max_tokens", "seed", "four_bit"].map(
        (key) => [key, deployment.settings[key]],
      ),
    );
    settings.agent_tools = Object.hasOwn(deployment.settings, "agent_tools")
      ? deployment.settings.agent_tools
      : false;
    ensure(
      typeof settings.agent_tools === "boolean" &&
        Number.isInteger(settings.max_seq) &&
        Number.isInteger(settings.max_image_size) &&
        settings.max_tokens === meta.max_tokens &&
        Number.isInteger(settings.seed) &&
        typeof settings.four_bit === "boolean",
      "Required generation settings are missing.",
    );
    const admission = Object.fromEntries(
      Object.entries(admissionBounds).map(([key, [minimum, maximum]]) => {
        if (!Object.hasOwn(deployment.settings, key)) return [key, null];
        const value = deployment.settings[key];
        ensure(
          Number.isInteger(value) && value >= minimum && value <= maximum,
          "Invalid recorded request admission limit.",
        );
        return [key, value];
      }),
    );
    return { records, meta, report, identity, settings, admission };
  }
  async function pair(dataFile, laneFiles) {
    const allFiles = [dataFile, ...laneFiles.flat()];
    ensure(
      allFiles.every(Boolean) &&
        allFiles.reduce((sum, file) => sum + file.size, 0) <= 64 * 1024 * 1024,
      "Select the dataset and two complete runs, totalling at most 64 MiB.",
    );
    const source = await read(dataFile),
      data = dataset(source.text);
    const lanes = await Promise.all(
      laneFiles.map((files) => run(files, data, source.hash)),
    );
    ensure(lanes.length === 2, "Select exactly two runs.");
    const [a, b] = lanes;
    ensure(
      a.meta.profile === b.meta.profile &&
        a.meta.history === b.meta.history &&
        a.meta.max_tokens === b.meta.max_tokens &&
        a.meta.temperature === b.meta.temperature &&
        equal(a.meta.images_sha256, b.meta.images_sha256) &&
        equal(a.settings, b.settings),
      "Runs must use the same dataset, image hashes, history mode and generation settings.",
    );
    const warnings = [];
    for (const key of Object.keys(admissionBounds)) {
      if (a.admission[key] !== null && b.admission[key] !== null)
        ensure(
          a.admission[key] === b.admission[key],
          "Runs must use the same recorded request admission limits.",
        );
    }
    for (const [index, lane] of lanes.entries()) {
      const unavailable = Object.keys(admissionBounds).filter(
        (key) => lane.admission[key] === null,
      );
      if (unavailable.length)
        warnings.push(
          `Run ${index ? "B" : "A"} request admission limits unavailable: ${unavailable.join(", ")}. Equal admission limits cannot be established; completion differences may reflect these limits.`,
        );
    }
    if (
      (a.identity.adapter && a.identity.adapter === b.identity.adapter) ||
      a.identity.artifact === b.identity.artifact
    )
      warnings.push(
        "Both runs identify the same model artifact; this is not a distinct-model comparison.",
      );
    if (
      a.identity.base !== b.identity.base ||
      !equal(a.meta.deployment.packages, b.meta.deployment.packages) ||
      a.meta.deployment.backend !== b.meta.deployment.backend
    )
      warnings.push(
        "Base model or runtime differs. Differences cannot be attributed to training alone.",
      );
    return {
      ...data,
      lanes,
      datasetHash: source.hash,
      imageHashes: a.meta.images_sha256,
      history: a.meta.history,
      profile: a.meta.profile,
      warnings,
      images: new Map(),
    };
  }
  async function png(file, expected) {
    ensure(
      file && file.size > 0 && file.size <= 16 * 1024 * 1024,
      "Choose a PNG image of at most 16 MiB.",
    );
    const bytes = new Uint8Array(await file.arrayBuffer()),
      view = new DataView(bytes.buffer);
    ensure(
      bytes.length >= 33 &&
        [137, 80, 78, 71, 13, 10, 26, 10].every(
          (item, index) => bytes[index] === item,
        ) &&
        view.getUint32(8) === 13 &&
        String.fromCharCode(...bytes.slice(12, 16)) === "IHDR",
      "Choose an original PNG image.",
    );
    const width = view.getUint32(16),
      height = view.getUint32(20);
    ensure(
      width > 0 &&
        height > 0 &&
        width <= 32768 &&
        height <= 32768 &&
        width * height <= 40000000,
      "Image dimensions exceed the 40-megapixel preview limit.",
    );
    const fingerprint = await digest(bytes);
    ensure(
      expected.includes(fingerprint),
      "Selected image does not match the dataset's recorded image hashes.",
    );
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 8192)
      binary += String.fromCharCode(...bytes.slice(offset, offset + 8192));
    return {
      hash: fingerprint,
      url: `data:image/png;base64,${btoa(binary)}`,
      bytes: file.size,
      pixels: width * height,
    };
  }
  async function pngBatch(files, expected, existing = new Map()) {
    ensure(
      files.length > 0 &&
        files.length <= 32 &&
        files.reduce((sum, file) => sum + file.size, 0) <= 64 * 1024 * 1024,
      "Choose 1–32 PNG files, totalling at most 64 MiB.",
    );
    const images = new Map(existing);
    for (const file of files) {
      const image = await png(file, expected);
      images.set(image.hash, image);
      ensure(
        images.size <= 32 &&
          [...images.values()].reduce((sum, item) => sum + item.bytes, 0) <=
            64 * 1024 * 1024 &&
          [...images.values()].reduce((sum, item) => sum + item.pixels, 0) <=
            80000000,
        "Source preview cache is limited to 32 images, 64 MiB and 80 megapixels. Clear previews before choosing another batch.",
      );
    }
    return images;
  }
  return { parse, jsonl, digest, dataset, pair, png, pngBatch };
})();
