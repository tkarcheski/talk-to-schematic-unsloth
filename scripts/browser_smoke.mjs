/** Real browser checks against a running schematic server; never mocks API calls.
 *
 * node scripts/browser_smoke.mjs --url http://127.0.0.1:8891
 * Add --question 'Show me R1.' only when the real model is ready.
 * Add --pdf FILE to check a local PDF upload.
 */
import { chromium, expect } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

function argumentsFrom(argv) {
  const options = {
    url: "http://127.0.0.1:8891",
    out: "results/browser-smoke",
    executable: "/usr/bin/chromium",
    "expected-examples": "157",
    "answer-timeout": "300000",
  };
  const allowed = new Set([...Object.keys(options), "question", "pdf"]);
  for (let i = 0; i < argv.length; i += 2) {
    const name = argv[i]?.replace(/^--/, "");
    if (
      !argv[i]?.startsWith("--") ||
      !allowed.has(name) ||
      !argv[i + 1] ||
      argv[i + 1].startsWith("--")
    ) {
      throw new Error(
        "Use --url URL --out DIRECTORY --executable CHROMIUM [--expected-examples 120] [--question TEXT] [--pdf FILE] [--answer-timeout MS]",
      );
    }
    options[name] = argv[i + 1];
  }
  const url = new URL(options.url);
  if (
    url.protocol !== "http:" ||
    !["127.0.0.1", "localhost"].includes(url.hostname)
  ) {
    throw new Error("Browser smoke targets a local HTTP schematic server only");
  }
  for (const name of ["expected-examples", "answer-timeout"]) {
    options[name] = Number(options[name]);
    if (!Number.isInteger(options[name]) || options[name] < 1)
      throw new Error(`${name} must be a positive integer`);
  }
  return options;
}

const options = argumentsFrom(process.argv.slice(2));
const destination = path.resolve(options.out);
await mkdir(destination, { recursive: true });
const report = {
  url: options.url,
  actualModelQuestion: options.question || null,
  checks: [],
  consoleErrors: [],
  pageErrors: [],
  screenshots: [],
};
let browser;
try {
  browser = await chromium.launch({
    executablePath: options.executable,
    headless: true,
    args: ["--disable-gpu"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
    deviceScaleFactor: 1,
  });
  page.on("console", (message) => {
    if (message.type() === "error") report.consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => report.pageErrors.push(error.message));
  await page.goto(options.url, {
    waitUntil: "domcontentloaded",
    timeout: 30000,
  });
  await page.locator("#open-library").click();
  await expect(page.locator("#library .board")).toHaveCount(
    options["expected-examples"],
    { timeout: 30000 },
  );
  report.checks.push(
    `Library lists ${options["expected-examples"]} real example sheets`,
  );
  const choices = await page
    .locator("#library .board")
    .evaluateAll((nodes) =>
      nodes.map((node) => ({ id: node.dataset.id, title: node.textContent })),
    );
  const boards = [choices[0], choices[choices.length - 1]];
  if (boards[0].id === boards[1].id)
    throw new Error("At least two distinct examples are required");
  report.boards = [];
  let previousSource;
  for (const board of boards) {
    if (!(await page.locator("#library").evaluate((node) => node.open)))
      await page.locator("#open-library").click();
    await page.locator(`#library .board[data-id="${board.id}"]`).click();
    await expect(page.locator("#doc-title")).toHaveText(
      board.title.replace(/ · sheet \d+$/u, ""),
      { timeout: 30000 },
    );
    await expect(page.locator("#sheet")).toBeVisible();
    await page.waitForFunction(() => {
      const image = document.getElementById("sheet");
      return image.complete && image.naturalWidth > 0;
    });
    const source = await page.locator("#source-link").getAttribute("href");
    expect(source).toMatch(
      /^https:\/\/github\.com\/(adafruit|sparkfun)\/[^/]+\/blob\/[0-9a-f]{40}\/.+\.sch$/,
    );
    if (previousSource) expect(source).not.toBe(previousSource);
    previousSource = source;
    await expect(page.locator("#source-meta")).toContainText(
      /CC-BY-SA-[34].0/u,
    );
    await expect(page.locator("#error")).toBeHidden();
    report.boards.push({ ...board, source });
  }
  report.checks.push(
    "Two distinct boards loaded from the library with pinned source and attribution",
  );

  await page.locator("#stage").focus();
  const fitted = await page.locator("#zoom-level").innerText();
  await page.locator("#stage").press("+");
  await expect(page.locator("#zoom-level")).not.toHaveText(fitted);
  await page.locator("#stage").press("0");
  await expect(page.locator("#zoom-level")).toHaveText(fitted);
  await page.locator("#paper").click();
  await expect(page.locator("#paper")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#paper").click();
  report.checks.push("Keyboard zoom, Fit and the Paper toggle work");

  await page.locator("#mode").selectOption("vision");
  await expect(page.locator("#gate-notice")).toBeVisible();
  await expect(page.locator(".message")).toHaveCount(0);
  await page.locator("#mode").selectOption("evidence");
  await expect(page.locator("#gate-notice")).toBeHidden();
  report.checks.push(
    "Image-only mode shows the accuracy-gate notice; evidence mode hides it",
  );
  if (options.pdf) {
    await page.locator("#open-library").click();
    await page.locator("#upload").setInputFiles(options.pdf);
    await expect(page.locator("#doc-title")).toHaveText(
      path.basename(options.pdf),
      { timeout: 30000 },
    );
    await expect(page.locator("#sheet")).toBeVisible();
    await expect(page.locator("#gate-notice")).toBeVisible();
    await expect(page.locator("#mode")).toBeDisabled();
    const capture = path.join(destination, "pdf-upload.png");
    await page.screenshot({ path: capture });
    report.screenshots.push(capture);
    report.checks.push(
      "Uploaded PDF rendered locally in image-only mode with the gate notice",
    );
  }
  const healthResponse = await page.request.get(
    new URL("/health", options.url).href,
  );
  const health = await healthResponse.json();
  if (!health.loaded) {
    await expect(page.locator("#chat-status")).toHaveText(
      "Waiting for model to load",
    );
    await expect(page.locator("#question")).toBeEnabled();
    await expect(page.locator("#send")).toBeDisabled();
    report.checks.push(
      "Unloaded model is reported accurately; drafting remains available while Send is disabled",
    );
  }

  if (options.question) {
    await expect(page.locator("#question")).toBeEnabled({ timeout: 30000 });
    await page.locator("#question").fill(options.question);
    await page.locator("#send").click();
    await expect(page.locator("#send")).toBeDisabled();
    await expect
      .poll(
        async () => {
          if (await page.locator("#error").isVisible())
            return `error: ${await page.locator("#error").innerText()}`;
          return (await page.locator(".message.assistant").count())
            ? "answered"
            : "pending";
        },
        { timeout: options["answer-timeout"], intervals: [500, 1000, 2000] },
      )
      .not.toBe("pending");
    if (await page.locator("#error").isVisible())
      throw new Error(await page.locator("#error").innerText());
    const answer = (
      await page.locator(".message.assistant .text").last().innerText()
    ).trim();
    expect(answer.length).toBeGreaterThan(0);
    await expect(page.locator("#question")).toHaveValue("");
    await expect(page.locator("#send")).toBeEnabled();
    report.answer = answer;
    report.toolActivity = await page
      .locator(".tool-event summary")
      .allTextContents();
    report.checks.push(
      "Real model returned a complete answer through the browser chat",
    );
  } else {
    report.checks.push(
      "Model generation was not requested; this run verifies UI and source browsing only",
    );
  }

  await page.evaluate(() => document.fonts.ready);
  for (const [name, viewport] of [
    ["desktop", { width: 1440, height: 1100 }],
    ["mobile", { width: 390, height: 844 }],
  ]) {
    await page.setViewportSize(viewport);
    await expect(page.locator("#sheet")).toBeVisible();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth + 1,
      ),
    ).toBe(true);
    const screenshot = path.join(destination, `${name}.png`);
    await page.screenshot({ path: screenshot, fullPage: true });
    report.screenshots.push(screenshot);
  }
  report.checks.push(
    "Desktop and mobile layouts have no horizontal overflow; screenshots saved",
  );
  report.modelStatus = await page.locator("#model-status").innerText();
  expect(report.consoleErrors, "Browser console errors").toEqual([]);
  expect(report.pageErrors, "Uncaught browser errors").toEqual([]);
  report.status = "passed";
} catch (error) {
  report.status = "failed";
  report.error = String(error?.stack || error);
  process.exitCode = 1;
} finally {
  if (browser) await browser.close();
  await writeFile(
    path.join(destination, "report.json"),
    `${JSON.stringify(report, null, 2)}\n`,
  );
  console.log(JSON.stringify(report, null, 2));
}
