/** Scripted browser comparison checks. No real inference or GPU loading. */
import assert from "node:assert/strict";
import { chromium, expect } from "@playwright/test";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import path from "node:path";
const root = path.resolve("schematic_model/web");
const out = path.resolve(
  process.env.SCHEMATIC_TEST_OUTPUT || "/tmp/schematic-comparison-preview",
);
await mkdir(out, { recursive: true });
const server = createServer(async (request, response) => {
  try {
    const file =
      request.url === "/"
        ? "index.html"
        : request.url.replace(/^\/assets\//u, "");
    if (!/^[\w./-]+$/u.test(file) || file.includes(".."))
      throw new Error("Invalid path");
    response.setHeader(
      "Content-Type",
      file.endsWith(".js")
        ? "text/javascript"
        : file.endsWith(".css")
          ? "text/css"
          : "text/html",
    );
    response.end(await readFile(path.join(root, file)));
  } catch {
    response.writeHead(404);
    response.end();
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({
  executablePath: process.env.CI
    ? undefined
    : process.env.SCHEMATIC_CHROMIUM || "/usr/bin/chromium",
  headless: true,
  args: ["--disable-gpu"],
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const fixture = {
    id: "fixture",
    title: "Scripted RC fixture",
    publisher: "Test fixtures",
    page: 1,
    attribution: "Test",
    license: "Test only",
    revision: "00000000",
    source_url: "",
    image_url: "/fixture.svg",
    regions: { parts: { R1: [100, 100, 200, 200] } },
    source_evidence: { values: { R1: "10k" } },
    suggested_questions: ["Read R1."],
  };
  await page.route("**/health", (r) =>
    r.fulfill({ json: { loaded: true, busy: false, model: "baseline" } }),
  );
  await page.route("**/api/view-tools", (r) =>
    r.fulfill({ json: { tools: [], instructions: "Test viewer" } }),
  );
  await page.route("**/api/agent/config", (r) =>
    r.fulfill({ json: { enabled: false, tools: [] } }),
  );
  await page.route("**/api/examples", (r) =>
    r.fulfill({ json: { examples: [fixture] } }),
  );
  await page.route("**/api/examples/fixture", (r) =>
    r.fulfill({ json: fixture }),
  );
  await page.route("**/fixture.svg", (r) =>
    r.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300"><rect width="600" height="300" fill="white"/><path d="M50 150H180L190 130L210 170L230 130L250 170L270 130L280 150H550" stroke="black" fill="none"/><text x="180" y="110" font-size="26">R1 10k</text></svg>',
    }),
  );
  await page.route("**/api/comparison/config", (r) =>
    r.fulfill({
      json: {
        enabled: true,
        max_tokens: 1024,
        profiles: [
          { id: "baseline", label: "Baseline", model: "v1" },
          { id: "candidate", label: "Candidate", model: "v3" },
        ],
      },
    }),
  );
  const requests = [];
  let releaseA;
  const waitA = new Promise((resolve) => {
    releaseA = resolve;
  });
  let duplicate = false;
  await page.route("**/api/comparison/complete", async (route) => {
    const body = route.request().postDataJSON();
    requests.push(body);
    if (requests.length === 1) await waitA;
    const a = body.profile === "baseline";
    await route.fulfill({
      json: {
        profile: body.profile,
        model: a ? "v1" : "v3",
        answer: a
          ? "R1 is 10k according to the supplied native evidence."
          : "R1 is 20k. Check the source before using this answer.",
        identity: "endpoint_reported",
        model_details: {
          adapter_sha256: (a || duplicate ? "a" : "b").repeat(64),
        },
        metrics: {
          prompt_tokens: 312,
          completion_tokens: 16,
          inference_seconds: 2,
          tokens_per_second: 8,
          timing_scope: "model_inference_including_prefill",
        },
      },
    });
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await expect(page.locator("#open-comparison")).toBeEnabled();
  await page.locator("#question").fill("Keep regular chat draft");
  await page.locator("#open-comparison").click();
  await expect(page.locator("#compare-send")).toBeEnabled();
  await page.locator("#compare-question").fill("Read R1.");
  await page.locator("#compare-send").click();
  await expect(page.locator("#compare-stage")).toContainText("Running A");
  await expect(page.locator("#compare-stop")).toBeVisible();
  await expect(page.locator("#compare-send")).toBeDisabled();
  await page.locator("#compare-question").fill("Draft next question");
  assert.equal(requests.length, 1, "second lane must not overlap");
  releaseA();
  await expect(page.locator("#compare-use-a")).toBeVisible();
  await expect(page.locator(".comparison-answer").nth(0)).toContainText("10k");
  await expect(page.locator(".comparison-answer").nth(1)).toContainText("20k");
  await expect(page.locator("#compare-question")).toHaveValue(
    "Draft next question",
  );
  assert.deepEqual(requests[0].messages, requests[1].messages);
  const trace = await page
    .locator(".comparison-card pre")
    .first()
    .textContent();
  assert(
    !trace.includes("data:image/"),
    "raw image data must not appear in traces",
  );
  await page.screenshot({
    path: path.join(out, "comparison-desktop.png"),
    fullPage: true,
  });
  await page.locator("#compare-use-a").click();
  await page.locator("#compare-question").fill("And R2?");
  duplicate = true;
  await page.locator("#compare-send").click();
  await expect(page.locator(".comparison-warning")).toContainText(
    "same artifact identity",
  );
  assert.deepEqual(requests[2].messages, requests[3].messages);
  assert.equal(
    requests[2].messages[2].content,
    "R1 is 10k according to the supplied native evidence.",
  );
  await page.locator("#compare-close").click();
  await expect(page.locator("#question")).toHaveValue(
    "Keep regular chat draft",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#open-comparison").click();
  await expect(page.locator("#comparison")).toBeVisible();
  assert(
    await page
      .locator("#comparison")
      .evaluate((element) => element.scrollWidth <= element.clientWidth + 1),
    "mobile comparison must not overflow horizontally",
  );
  await page.screenshot({
    path: path.join(out, "comparison-mobile.png"),
    fullPage: true,
  });
  assert.deepEqual(errors, []);
  console.log(
    "PASS: scripted live comparison, sequential requests, shared input/history, editable drafts, identity warnings and responsive layout. No model-quality claim.",
  );
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
