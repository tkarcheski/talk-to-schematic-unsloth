/** Rendered local import tests. Fixtures are scripted; optional existing model
 * files are reviewed without invoking a model or changing their scores. */
import assert from "node:assert/strict";
import { chromium, expect } from "@playwright/test";
import { createServer } from "node:http";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import {
  failure,
  fingerprint,
  fixture,
  inputs,
  png,
  secondPng,
  twoBoards,
} from "./saved_benchmark_fixture.mjs";
const root = path.resolve("schematic_model/web");
const out = path.resolve(
  process.env.SCHEMATIC_TEST_OUTPUT || "/tmp/schematic-saved-preview",
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
const payload = async (file) => ({
  name: file.name,
  mimeType: "application/json",
  buffer: Buffer.from(await file.arrayBuffer()),
});
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  const errors = [],
    requests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => requests.push(request.url()));
  await page.route("**/health", (route) =>
    route.fulfill({ json: { loaded: false, busy: false, model: null } }),
  );
  await page.route("**/api/view-tools", (route) =>
    route.fulfill({ json: { tools: [], instructions: "Test" } }),
  );
  await page.route("**/api/agent/config", (route) =>
    route.fulfill({ json: { enabled: false, tools: [] } }),
  );
  await page.route("**/api/examples", (route) =>
    route.fulfill({ json: { examples: [] } }),
  );
  await page.route("**/api/comparison/config", (route) =>
    route.fulfill({ json: { enabled: false, profiles: [] } }),
  );
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await expect(page.locator("#open-comparison")).toBeEnabled();
  const chatDraft = await page.locator("#question").inputValue();
  await page.locator("#open-comparison").click();
  await expect(page.locator("#compare-send")).toBeDisabled();
  await page
    .locator("#compare-question")
    .fill("Unfinished live comparison draft");
  await page.locator("#compare-saved").click();
  await expect(page.locator("#saved-comparison")).toBeVisible();
  const storage = await page.evaluate(() =>
    JSON.stringify({
      local: { ...localStorage },
      session: { ...sessionStorage },
    }),
  );
  async function load(value) {
    const [dataset, lanes] = inputs(value);
    await page
      .locator("#saved-files")
      .evaluate((element) => (element.open = true));
    await page.locator("#saved-dataset").setInputFiles(await payload(dataset));
    for (const [index, lane] of lanes.entries())
      await page
        .locator(index ? "#saved-run-b" : "#saved-run-a")
        .setInputFiles(await Promise.all(lane.map(payload)));
    await page.locator("#saved-load").click();
  }
  await page.locator("#saved-label-a").fill("Scripted baseline");
  await page.locator("#saved-label-b").fill("Scripted candidate");
  await load(fixture());
  await expect(page.locator("#saved-import-status")).toContainText(
    "Imported locally",
  );
  await expect(page.locator("#saved-binding")).toContainText("2 turns");
  await expect(page.locator("#saved-summary")).toContainText(
    "1/2 recorded stop",
  );
  await expect(page.locator("#saved-case")).toContainText("Read R1's value.");
  await page.locator("#saved-filter").selectOption("b-failure");
  await expect(page.locator("#saved-count")).toHaveText("1 of 2 turns match");
  await expect(page.locator("#saved-case")).toContainText(
    "And what value is R2?",
  );
  await expect(page.locator('#saved-case [data-lane="0"]')).toContainText(
    "R2 is 20K.",
  );
  await expect(page.locator('#saved-case [data-lane="1"]')).toContainText(
    "No scored answer",
  );
  await expect(page.locator('#saved-case [data-lane="1"]')).toContainText(
    "Finish: error",
  );
  assert(
    !(await page.locator("#saved-comparison").textContent()).includes(
      "/private/",
    ),
    "metadata paths and raw errors must not appear in the view",
  );
  await page.screenshot({
    path: path.join(out, "saved-desktop.png"),
    fullPage: true,
  });
  await page.locator("#saved-image-tools summary").click();
  await page
    .locator("#saved-image-file")
    .setInputFiles({ name: "source.png", mimeType: "image/png", buffer: png });
  await expect(page.locator("#saved-images img")).toBeVisible();
  await expect
    .poll(() =>
      page
        .locator("#saved-images img")
        .evaluate((element) => element.naturalWidth),
    )
    .toBe(1);
  await expect(page.locator("#saved-image-error")).toBeEmpty();
  await page.locator("#saved-image-file").setInputFiles({
    name: "wrong.svg",
    mimeType: "image/svg+xml",
    buffer: Buffer.from("<svg/>"),
  });
  await expect(page.locator("#saved-image-error")).toContainText(
    "original PNG",
  );
  // A matching hash and PNG header do not prove that source pixels can decode.
  const malformedImages = twoBoards();
  const truncatedPng = secondPng.subarray(0, 33);
  for (const meta of [malformedImages.meta, malformedImages.other])
    meta.images_sha256["/private/second-source.png"] =
      fingerprint(truncatedPng);
  await load(malformedImages);
  await expect(page.locator("#saved-import-status")).toContainText(
    "Imported locally",
  );
  await page.locator("#saved-image-file").setInputFiles({
    name: "source.png",
    mimeType: "image/png",
    buffer: png,
  });
  await expect(page.locator("#saved-image-status")).toContainText(
    "1 matching source previews",
  );
  await page.locator("#saved-image-file").setInputFiles([
    { name: "source.png", mimeType: "image/png", buffer: png },
    { name: "truncated.png", mimeType: "image/png", buffer: truncatedPng },
  ]);
  await expect(page.locator("#saved-image-error")).toContainText(
    "could not be decoded",
  );
  await expect(page.locator("#saved-image-status")).toContainText(
    "Existing previews are unchanged",
  );
  await expect(page.locator("#saved-images img")).toHaveAttribute(
    "src",
    "data:image/png;base64," + png.toString("base64"),
  );
  await page.locator("#saved-case-select").selectOption("1");
  await expect(page.locator("#saved-images img")).toHaveCount(0);
  await expect(page.locator("#saved-images")).toContainText("not loaded");

  const legacy = fixture();
  for (const meta of [legacy.meta, legacy.other])
    for (const key of ["max_images", "max_text_chars", "max_messages"])
      delete meta.deployment.settings[key];
  await load(legacy);
  await expect(page.locator("#saved-warning")).toContainText(
    "Run A request admission limits unavailable",
  );
  await expect(page.locator("#saved-warning")).toContainText(
    "Run B request admission limits unavailable",
  );
  await expect(page.locator("#saved-summary pre").first()).toContainText(
    '"max_messages": "unavailable"',
  );
  const unequal = fixture();
  unequal.other.deployment.settings.max_messages = 2;
  await load(unequal);
  await expect(page.locator("#saved-error")).toContainText(
    "same recorded request admission limits",
  );
  await expect(page.locator("#saved-case")).toContainText(
    "And what value is R2?",
  );
  await expect(page.locator("#saved-warning")).toContainText(
    "Run A request admission limits unavailable",
  );
  await load(twoBoards());
  await expect(page.locator("#saved-import-status")).toContainText(
    "Imported locally",
  );
  await expect(page.locator("#saved-summary")).toContainText(
    "Scripted baseline",
  );
  await expect(page.locator("#saved-summary")).toContainText("Operator label");
  await page.locator("#saved-image-file").setInputFiles([
    { name: "first.png", mimeType: "image/png", buffer: png },
    { name: "second.png", mimeType: "image/png", buffer: secondPng },
  ]);
  await expect(page.locator("#saved-image-status")).toContainText(
    "2 matching source previews",
  );
  await page.locator("#saved-case-select").selectOption("1");
  await expect(page.locator("#saved-case")).toContainText("Conversation 2");
  await expect(page.locator("#saved-images img")).toHaveAttribute(
    "src",
    "data:image/png;base64," + secondPng.toString("base64"),
  );
  await expect
    .poll(() =>
      page
        .locator("#saved-images img")
        .evaluate((element) => element.naturalWidth),
    )
    .toBe(1);
  await page.locator("#saved-image-file").setInputFiles([
    { name: "first.png", mimeType: "image/png", buffer: png },
    { name: "bad.png", mimeType: "image/png", buffer: Buffer.from("bad") },
  ]);
  await expect(page.locator("#saved-image-status")).toContainText(
    "Existing previews are unchanged",
  );
  await expect(page.locator("#saved-images img")).toHaveAttribute(
    "src",
    "data:image/png;base64," + secondPng.toString("base64"),
  );
  await page.locator("#saved-clear-images").click();
  await expect(page.locator("#saved-images img")).toHaveCount(0);
  await load(fixture());
  await expect(page.locator("#saved-import-status")).toContainText(
    "Imported locally",
  );
  const changed = fixture();
  changed.dataset += "\n";
  await load(changed);
  await expect(page.locator("#saved-error")).toContainText(
    "hashes do not match",
  );
  await expect(page.locator("#saved-case")).toContainText(
    "And what value is R2?",
  );
  await expect(page.locator("#saved-import-status")).toContainText(
    "previous valid comparison",
  );
  const generated = fixture();
  generated.meta.history = generated.other.history = "generated";
  generated.b = [failure("case#0"), failure("case#1", false, "case#0")];
  await load(generated);
  await expect(page.locator("#saved-binding")).toContainText(
    "independent generated histories",
  );
  await page.locator("#saved-case-select").selectOption("1");
  await expect(page.locator('#saved-case [data-lane="1"]')).toContainText(
    "Blocked by earlier failure",
  );
  await page.locator('#saved-case [data-lane="1"] summary').click();
  await expect(page.locator('#saved-case [data-lane="1"] pre')).toContainText(
    "dependent turns were not attempted",
  );
  const malicious = fixture();
  malicious.a[0].output =
    '<img src="/must-not-request" onerror="window.injected=true">';
  await load(malicious);
  await expect(page.locator("#saved-import-status")).toContainText(
    "Imported locally",
  );
  await page.locator("#saved-filter").selectOption("all");
  await expect(
    page.locator('#saved-case [data-lane="0"] .comparison-answer'),
  ).toHaveText(malicious.a[0].output);
  assert.equal(await page.locator("#saved-case img").count(), 0);
  assert.equal(await page.evaluate(() => window.injected), undefined);
  await load(fixture());
  await expect(page.locator("#saved-import-status")).toContainText(
    "Imported locally",
  );
  await page.locator("#saved-filter").selectOption("b-failure");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#saved-case").scrollIntoViewIfNeeded();
  await page.screenshot({
    path: path.join(out, "saved-mobile.png"),
    fullPage: false,
  });
  assert(
    await page
      .locator("#saved-comparison")
      .evaluate((element) => element.scrollWidth <= element.clientWidth + 1),
    "mobile comparison must not overflow horizontally",
  );
  await page.locator("#saved-close").click();
  await expect(page.locator("#open-comparison")).toBeFocused();
  await expect(page.locator("#question")).toHaveValue(chatDraft);
  await page.locator("#open-comparison").click();
  await expect(page.locator("#compare-question")).toHaveValue(
    "Unfinished live comparison draft",
  );
  await page.locator("#compare-saved").click();
  assert.equal(
    await page.evaluate(() =>
      JSON.stringify({
        local: { ...localStorage },
        session: { ...sessionStorage },
      }),
    ),
    storage,
    "imports must not persist content in browser storage",
  );
  const realFiles = [
    process.env.SCHEMATIC_SAVED_DATASET,
    process.env.SCHEMATIC_SAVED_A,
    process.env.SCHEMATIC_SAVED_B,
  ];
  if (realFiles.some(Boolean)) {
    assert(
      realFiles.every(Boolean),
      "Set all three saved-run input variables together",
    );
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page
      .locator("#saved-files")
      .evaluate((element) => (element.open = true));
    await page
      .locator("#saved-label-a")
      .fill(process.env.SCHEMATIC_SAVED_LABEL_A || "Run A");
    await page
      .locator("#saved-label-b")
      .fill(process.env.SCHEMATIC_SAVED_LABEL_B || "Run B");
    await page.locator("#saved-dataset").setInputFiles(realFiles[0]);
    for (let lane = 0; lane < 2; lane++)
      await page
        .locator(lane ? "#saved-run-b" : "#saved-run-a")
        .setInputFiles([
          realFiles[lane + 1],
          realFiles[lane + 1] + ".meta.json",
          realFiles[lane + 1] + ".report.json",
        ]);
    await page.locator("#saved-load").click();
    await expect(page.locator("#saved-import-status")).toContainText(
      "Imported locally",
    );
    await page.locator("#saved-filter").selectOption("b-failure");
    if (process.env.SCHEMATIC_SAVED_IMAGES) {
      const originals = JSON.parse(process.env.SCHEMATIC_SAVED_IMAGES);
      assert(
        Array.isArray(originals) &&
          originals.length > 0 &&
          originals.length <= 32,
        "Choose 1–32 explicit original PNG paths",
      );
      await page
        .locator("#saved-image-tools")
        .evaluate((element) => (element.open = true));
      await page.locator("#saved-image-file").setInputFiles(originals);
      await expect(page.locator("#saved-image-status")).toContainText(
        `${originals.length} matching source previews`,
      );
      await expect(page.locator("#saved-images img")).toBeVisible();
      console.log(
        "Original PNG previews verified by recorded dataset hashes:",
        originals.length,
      );
      await page
        .locator("#saved-image-tools")
        .evaluate((element) => (element.open = false));
    }
    await page.screenshot({
      path: path.join(out, "saved-existing-predictions.png"),
      fullPage: true,
    });
    console.log(
      "Existing model predictions reviewed (no new inference):",
      await page.locator("#saved-binding").textContent(),
      await page.locator("#saved-count").textContent(),
    );
    console.log(
      "Recorded metrics:",
      (await page.locator("#saved-summary").innerText()).replace(
        /Recorded identity and settings/gu,
        "",
      ),
    );
  }
  assert(
    !requests.some((url) =>
      /must-not|completions|comparison\/complete/u.test(url),
    ),
    "imports must not fetch source references or invoke models",
  );
  assert.deepEqual(errors, []);
  console.log(
    "Saved comparison browser checks passed; screenshots written to the configured local output directory.",
  );
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
