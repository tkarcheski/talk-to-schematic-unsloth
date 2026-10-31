/** CPU import integrity tests, with no network or model execution. */
import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import { readFileSync } from "node:fs";
import { webcrypto } from "node:crypto";
import {
  bundle,
  failure,
  file,
  fingerprint,
  fixture,
  inputs,
  jsonl,
  png,
  secondPng,
} from "./saved_benchmark_fixture.mjs";
const source = readFileSync("schematic_model/web/saved_benchmark.js", "utf8");
const api = vm.runInNewContext(source + ";savedBenchmark;", {
  crypto: webcrypto,
  TextDecoder,
  structuredClone,
  btoa,
});
const rejects = async (value, pattern) =>
  assert.rejects(api.pair(...inputs(value)), pattern);
test("a complete pair binds dataset, reports, images and exact turn keys", async () => {
  const pair = await api.pair(...inputs());
  assert.equal(pair.cases.length, 2);
  assert.equal(pair.lanes[1].report.failed_turns, 1);
  assert.equal(pair.cases[1].question, "And what value is R2?");
  assert.equal(pair.lanes[1].records.get("case#1").output, "");
  assert.equal(pair.warnings.length, 0);
});
test("strict JSON rejects duplicate escaped keys, invalid syntax, deep nesting and non-finite values", () => {
  for (const value of [
    '{"x":1,"x":2}',
    '{"x":1,"\\u0078":2}',
    '{"nested":{"x":1,"x":2}}',
    '{"__proto__":1,"__proto__":2}',
    '{"x":1e999}',
    '{"x":NaN}',
    "[1,]",
    '{"x":1} garbage',
    "[".repeat(50) + "0" + "]".repeat(50),
  ])
    assert.throws(() => api.parse(value), /Invalid JSON/u);
  assert.equal(api.parse('{"a":"quoted \\\"text\\\"","b":false}').b, false);
});
test("changed dataset or prediction bytes cannot be substituted into a report", async () => {
  const value = fixture();
  value.dataset += "\n";
  await rejects(value, /hashes do not match/u);
  const [dataset, lanes] = inputs();
  lanes[0][0] = file("a.jsonl", (await lanes[0][0].text()) + "\n");
  await assert.rejects(api.pair(dataset, lanes), /hashes do not match/u);
});
test("report provenance and totals must agree, even with matching prediction hashes", async () => {
  const value = fixture();
  for (const mutate of [
    (report) => (report.provenance.history = "generated"),
    (report) => (report.successful_turns = 200),
    (report) => (report.failure_rate = 0.2),
  ]) {
    await assert.rejects(
      api.pair(file("dataset.jsonl", value.dataset), [
        bundle(value.a, value.meta, "a", mutate),
        bundle(value.b, value.other, "b"),
      ]),
      /hashes do not match|totals disagree/u,
    );
  }
});
test("missing, extra and duplicate prediction keys fail closed", async () => {
  for (const mutation of [
    (rows) => rows.pop(),
    (rows) => rows.push({ ...rows[0], key: "unknown#0" }),
    (rows) => rows.push(rows[0]),
  ]) {
    const value = fixture();
    mutation(value.a);
    await rejects(value, /keys must exactly|Duplicate/u);
  }
});
test("partial answers and incomplete successes cannot become scored answers", async () => {
  for (const mutation of [
    (record) => (record.output = "partial"),
    (record) => (record.finish_reason = "length"),
    (record) => (record.attempted = false),
  ]) {
    const value = fixture();
    mutation(value.b[1]);
    await rejects(value, /Failed or partial/u);
  }
  const value = fixture();
  value.a[1].finish_reason = "length";
  await rejects(value, /Successful records/u);
});
test("generated-history blocking is tied to the first earlier failed key", async () => {
  const value = fixture();
  value.meta.history = value.other.history = "generated";
  value.b = [failure("case#0"), failure("case#1", false, "case#0")];
  assert.equal((await api.pair(...inputs(value))).history, "generated");
  value.b[1].error.caused_by = "other#0";
  await rejects(value, /contradicts/u);
  value.b[1] = value.a[1];
  await rejects(value, /contradicts/u);
});
test("gold history may not contain unattempted dependency failures", async () => {
  const value = fixture();
  value.b[1] = failure("case#1", false, "case#0");
  await rejects(value, /blocked without/u);
});
test("history, image hashes and generation settings must match between runs", async () => {
  for (const mutation of [
    (meta) => (meta.history = "generated"),
    (meta) => meta.deployment.settings.seed++,
    (meta) =>
      (meta.images_sha256["/private/must-not-fetch/source.png"] = "f".repeat(
        64,
      )),
  ]) {
    const value = fixture();
    value.b = structuredClone(value.a);
    mutation(value.other);
    await rejects(value, /same dataset/u);
  }
  const value = fixture();
  value.other.images_sha256 = {};
  await rejects(value, /Image hash coverage/u);
});
test("same adapter receives an explicit identity warning", async () => {
  const value = fixture();
  value.other = structuredClone(value.meta);
  assert.match(
    (await api.pair(...inputs(value))).warnings[0],
    /same model artifact/u,
  );
});
test("different recorded admission limits cannot masquerade as a model regression", async () => {
  for (const key of ["max_images", "max_text_chars", "max_messages"]) {
    const value = fixture();
    value.other.deployment.settings[key] = 2;
    await rejects(value, /same recorded request admission limits/u);
  }
});
test("legacy admission limits remain unavailable without inventing equality", async () => {
  const keys = ["max_images", "max_text_chars", "max_messages"];
  for (const missingBoth of [false, true]) {
    const value = fixture();
    for (const key of keys) {
      delete value.meta.deployment.settings[key];
      if (missingBoth) delete value.other.deployment.settings[key];
    }
    const pair = await api.pair(...inputs(value));
    assert.equal(pair.warnings.length, missingBoth ? 2 : 1);
    assert.match(
      pair.warnings[0],
      /Run A request admission limits unavailable/u,
    );
    assert.match(
      pair.warnings[0],
      /Equal admission limits cannot be established/u,
    );
    for (const key of keys) {
      assert.equal(pair.lanes[0].admission[key], null);
      assert.equal(
        pair.lanes[1].admission[key],
        missingBoth ? null : value.other.deployment.settings[key],
      );
    }
  }
});
test("recorded malformed admission limits are not treated as legacy omissions", async () => {
  for (const invalid of [null, false, "64", 0, -1, 513]) {
    const value = fixture();
    value.meta.deployment.settings.max_messages = invalid;
    await rejects(value, /Invalid recorded request admission limit/u);
  }
});
test("recorded identities and numeric metrics cannot carry arbitrary paths or invalid values", async () => {
  const value = fixture();
  value.a[0].model_fingerprint = "/private/path";
  await rejects(value, /fingerprint/u);
  const next = fixture();
  next.a[0].usage.completion_tokens = -1;
  await rejects(next, /token usage/u);
});
test("dataset alternation and image references are checked", async () => {
  const value = fixture(),
    rows = JSON.parse(value.dataset);
  rows.messages[2].role = "user";
  value.dataset = jsonl([rows]);
  value.meta.dataset_sha256 = value.other.dataset_sha256 = fingerprint(
    value.dataset,
  );
  await rejects(value, /alternate/u);
});
test("oversized files are rejected before reading", async () => {
  const [dataset, lanes] = inputs();
  let read = false;
  const oversized = {
    size: 65 * 1024 * 1024,
    arrayBuffer: () => {
      read = true;
      throw new Error("read");
    },
  };
  await assert.rejects(api.pair(oversized, lanes), /64 MiB/u);
  assert.equal(read, false);
  assert.ok(dataset.size > 0);
});
test("only a manually selected original PNG with the expected hash is displayed", async () => {
  const result = await api.png(file("source.png", png), [fingerprint(png)]);
  assert.match(result.url, /^data:image\/png;base64,/u);
  await assert.rejects(
    api.png(file("wrong.png", png), ["f".repeat(64)]),
    /does not match/u,
  );
  await assert.rejects(
    api.png(file("source.svg", "<svg/>"), [fingerprint(png)]),
    /original PNG/u,
  );
  const huge = Buffer.from(png);
  huge.writeUInt32BE(100000, 16);
  await assert.rejects(
    api.png(file("huge.png", huge), [fingerprint(huge)]),
    /40-megapixel/u,
  );
});

test("agent tool mode must match and recorded successes must not include errors", async () => {
  const value = fixture();
  value.other.deployment.settings.agent_tools = true;
  await rejects(value, /same dataset/u);
  value.other.deployment.settings.agent_tools = false;
  assert.equal(
    (await api.pair(...inputs(value))).lanes[0].settings.agent_tools,
    false,
  );
  value.a[0].error = { code: "generation_failed" };
  await rejects(value, /Successful records/u);
});

test("image batches merge by hash, reject mixed invalid selections atomically and enforce cache bounds", async () => {
  const files = [file("first.png", png), file("second.png", secondPng)],
    expected = [fingerprint(png), fingerprint(secondPng)];
  const first = await api.pngBatch([files[0]], expected);
  const both = await api.pngBatch(files, expected, first);
  assert.equal(first.size, 1);
  assert.equal(both.size, 2);
  assert.equal((await api.pngBatch(files, expected, both)).size, 2);
  await assert.rejects(
    api.pngBatch([files[0], file("bad.png", "<svg/>")], expected, first),
    /original PNG/u,
  );
  assert.equal(first.size, 1);
  await assert.rejects(
    api.pngBatch(Array(33).fill(files[0]), expected),
    /1–32/u,
  );
  await assert.rejects(
    api.pngBatch([{ size: 65 * 1024 * 1024 }], expected),
    /64 MiB/u,
  );
  const full = new Map(
    Array.from({ length: 32 }, (_, index) => [
      String(index),
      { bytes: 1, pixels: 1 },
    ]),
  );
  await assert.rejects(
    api.pngBatch([files[0]], expected, full),
    /cache is limited/u,
  );
  await assert.rejects(
    api.pngBatch(
      [files[0]],
      expected,
      new Map([["existing", { bytes: 64 * 1024 * 1024, pixels: 1 }]]),
    ),
    /cache is limited/u,
  );
});

test("source image cache also bounds decoded pixel area", async () => {
  await assert.rejects(
    api.pngBatch(
      [file("first.png", png)],
      [fingerprint(png)],
      new Map([["existing", { bytes: 1, pixels: 80000000 }]]),
    ),
    /cache is limited/u,
  );
});
