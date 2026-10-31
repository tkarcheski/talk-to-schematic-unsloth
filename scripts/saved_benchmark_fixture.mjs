/** Artificial benchmark fixtures for import tests; no model quality claims. */
// cspell:words IHDR IDAT IEND
import { createHash } from "node:crypto";
import { crc32, deflateSync } from "node:zlib";
export const fingerprint = (bytes) =>
  createHash("sha256").update(bytes).digest("hex");
export const png = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2ioAAAAASUVORK5CYII=",
  "base64",
);
// A second valid PNG makes cross-conversation preview tests distinguishable.
const chunk = (name, data) => {
  const type = Buffer.from(name),
    length = Buffer.alloc(4),
    check = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  check.writeUInt32BE(crc32(Buffer.concat([type, data])));
  return Buffer.concat([length, type, data, check]);
};
const header = Buffer.from("00000001000000010806000000", "hex");
export const secondPng = Buffer.concat([
  png.subarray(0, 8),
  chunk("IHDR", header),
  chunk("IDAT", deflateSync(Buffer.from([0, 0, 255, 0, 255]))),
  chunk("IEND", Buffer.alloc(0)),
]);
export const hash = (letter) => letter.repeat(64);
export const text = (value) => JSON.stringify(value);
export const jsonl = (rows) => rows.map(text).join("\n") + "\n";
export const file = (name, value) => new File([value], name);
export const failure = (key, attempted = true, causedBy) => ({
  key,
  attempted,
  status: "failure",
  output: "",
  finish_reason: attempted ? "error" : "dependency_failed",
  error: {
    code: attempted ? "generation_failed" : "dependency_failed",
    message: "/private/must-not-render/model",
    ...(causedBy ? { caused_by: causedBy } : {}),
  },
});
export function fixture() {
  const messages = [
    {
      role: "system",
      content: [{ type: "text", text: "Read the schematic carefully." }],
    },
    {
      role: "user",
      content: [
        { type: "image", image: "/private/must-not-fetch/source.png" },
        { type: "text", text: "Read R1's value." },
      ],
    },
    { role: "assistant", content: [{ type: "text", text: "R1 is 10K." }] },
    {
      role: "user",
      content: [{ type: "text", text: "And what value is R2?" }],
    },
    { role: "assistant", content: [{ type: "text", text: "R2 is 20K." }] },
  ];
  const dataset = jsonl([
    {
      id: "case",
      gold: [
        { type: "value", value: "10K" },
        { type: "value", value: "20K" },
      ],
      messages,
    },
  ]);
  const record = (key, output) => ({
    key,
    output,
    status: "success",
    attempted: true,
    finish_reason: "stop",
    elapsed_seconds: 2,
    usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
    model_fingerprint: hash("a"),
  });
  const a = [record("case#0", "R1 is 10K."), record("case#1", "R2 is 20K.")];
  const b = [
    { ...record("case#0", "R1 reads 10K."), model_fingerprint: hash("b") },
    failure("case#1"),
  ];
  const meta = {
    schema_version: 1,
    runner: "failure-accounted-benchmark",
    dataset_sha256: fingerprint(dataset),
    images_sha256: { "/private/must-not-fetch/source.png": fingerprint(png) },
    profile: "real-vision",
    history: "gold",
    max_tokens: 1024,
    temperature: 0,
    deployment: {
      schema_version: 1,
      backend: "test",
      settings: {
        model: "/private/must-not-render/model",
        max_seq: 8192,
        max_image_size: 2048,
        max_tokens: 1024,
        seed: 3407,
        four_bit: false,
        max_images: 4,
        max_text_chars: 200000,
        max_messages: 64,
      },
      model: {
        path: "/private/must-not-render/model",
        sha256: hash("a"),
        files: [{ name: "adapter_model.safetensors", sha256: hash("a") }],
        base_model: { sha256: hash("c") },
      },
      packages: { engine: "test" },
    },
  };
  const other = structuredClone(meta);
  other.deployment.model.sha256 = hash("b");
  other.deployment.model.files[0].sha256 = hash("b");
  return { dataset, a, b, meta, other };
}
export function bundle(records, meta, prefix = "run", mutateReport = () => {}) {
  const predictions = jsonl(records),
    failed = records.filter((record) => record.status === "failure").length;
  const report = {
    schema_version: 1,
    profile: meta.profile,
    provenance: structuredClone(meta),
    predictions_sha256: fingerprint(predictions),
    turns: records.length,
    conversations: new Set(records.map((record) => record.key.split("#")[0]))
      .size,
    attempted_turns: records.filter((record) => record.attempted).length,
    successful_turns: records.length - failed,
    failed_turns: failed,
    failure_rate: failed / records.length,
    metrics: { "value.acc": 0.5 },
    counts: { "value.acc": records.length },
  };
  mutateReport(report);
  return [
    file(`${prefix}.jsonl`, predictions),
    file(`${prefix}.jsonl.meta.json`, text(meta)),
    file(`${prefix}.jsonl.report.json`, text(report)),
  ];
}
export function inputs(value = fixture()) {
  return [
    file("dataset.jsonl", value.dataset),
    [bundle(value.a, value.meta, "a"), bundle(value.b, value.other, "b")],
  ];
}

export function twoBoards() {
  const value = fixture(),
    first = JSON.parse(value.dataset),
    second = structuredClone(first);
  second.id = "case2";
  second.messages[1].content[0].image = "/private/second-source.png";
  second.messages[1].content[1].text = "Read R1 on the second board.";
  value.dataset = jsonl([first, second]);
  for (const meta of [value.meta, value.other]) {
    meta.dataset_sha256 = fingerprint(value.dataset);
    meta.images_sha256["/private/second-source.png"] = fingerprint(secondPng);
  }
  for (const key of ["a", "b"])
    value[key].push(
      ...value[key].map((record) => ({
        ...structuredClone(record),
        key: record.key.replace("case#", "case2#"),
      })),
    );
  return value;
}
