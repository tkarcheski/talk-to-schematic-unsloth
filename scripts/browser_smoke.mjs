/** Real browser checks against a running schematic server; never mocks API calls.
 *
 * node scripts/browser_smoke.mjs --url http://127.0.0.1:8891
 * Add --question 'What value is shown for R1?' only when the real model is ready.
 */
import { chromium, expect } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

function argumentsFrom(argv) {
  const options = {
    url: "http://127.0.0.1:8891",
    out: "results/browser-smoke",
    executable: "/usr/bin/chromium",
    "expected-examples": "120",
    "answer-timeout": "300000",
  };
  const allowed = new Set([...Object.keys(options), "question"]);
  for (let i = 0; i < argv.length; i += 2) {
    const name = argv[i]?.replace(/^--/, "");
    if (
      !argv[i]?.startsWith("--") ||
      !allowed.has(name) ||
      !argv[i + 1] ||
      argv[i + 1].startsWith("--")
    ) {
      throw new Error(
        "Use --url URL --out DIRECTORY --executable CHROMIUM [--expected-examples 120] [--question TEXT] [--answer-timeout MS]",
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
  await expect(page.locator("#example option")).toHaveCount(
    options["expected-examples"],
    { timeout: 30000 },
  );
  await expect(page.locator("#example")).toBeEnabled();
  report.checks.push(
    `Catalog has ${options["expected-examples"]} real examples`,
  );

  const choices = await page
    .locator("#example option")
    .evaluateAll((nodes) =>
      nodes.map((node) => ({ id: node.value, title: node.textContent })),
    );
  const boards = [choices[0], choices[Math.floor(choices.length / 2)]];
  if (boards[0].id === boards[1].id)
    throw new Error("At least two distinct examples are required");
  report.boards = [];
  let previousSource;
  for (const board of boards) {
    await page.locator("#example").selectOption(board.id);
    await expect(page.locator("#image-link")).toHaveAttribute(
      "href",
      `/api/examples/${board.id}/image`,
      { timeout: 30000 },
    );
    await expect(page.locator("#schematic-image")).toBeVisible();
    await page.waitForFunction(() => {
      const image = document.getElementById("schematic-image");
      return (
        image.complete && image.naturalWidth > 0 && image.naturalHeight > 0
      );
    });
    const source = await page.locator("#source-link").getAttribute("href");
    expect(source).toMatch(
      /^https:\/\/github\.com\/adafruit\/[^/]+\/blob\/[0-9a-f]{40}\/.+\.sch$/,
    );
    if (previousSource) expect(source).not.toBe(previousSource);
    previousSource = source;
    await expect(page.locator("#source-meta")).toContainText("CC-BY-SA-3.0");
    await expect(page.locator("#error")).toBeHidden();
    report.boards.push({
      ...board,
      source,
      image: await page.locator("#schematic-image").evaluate((image) => ({
        width: image.naturalWidth,
        height: image.naturalHeight,
      })),
    });
  }
  report.checks.push(
    "Two distinct board images loaded with pinned source and attribution",
  );

  await expect(page.locator("#zoom-actual")).toBeEnabled();
  await page.locator("#zoom-actual").click();
  await expect(page.locator("#zoom-level")).toHaveText("100%");
  const actualSize = await page
    .locator("#schematic-image")
    .evaluate((image) => ({
      displayed: image.getBoundingClientRect().width,
      native: image.naturalWidth,
    }));
  expect(Math.abs(actualSize.displayed - actualSize.native)).toBeLessThan(1);
  expect(
    await page
      .locator("#image-stage")
      .evaluate((stage) => stage.scrollWidth > stage.clientWidth),
  ).toBe(true);
  const zoomScreenshot = path.join(destination, "desktop-100-percent.png");
  await page.locator("#image-stage").screenshot({ path: zoomScreenshot });
  report.screenshots.push(zoomScreenshot);
  await page.locator("#image-stage").focus();
  await page.locator("#image-stage").press("+");
  await expect(page.locator("#zoom-level")).toHaveText("135%");
  await page.locator("#image-stage").press("0");
  await expect(page.locator("#zoom-fit")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect
    .poll(() =>
      page
        .locator("#image-stage")
        .evaluate(
          (stage) =>
            stage.scrollWidth <= stage.clientWidth + 1 &&
            stage.scrollHeight <= stage.clientHeight + 1,
        ),
    )
    .toBe(true);
  report.checks.push(
    "100% uses native image resolution inside a bounded scrolling viewport; keyboard zoom and Fit work",
  );

  await page.locator("#mode").selectOption("vision");
  await expect(page.locator("#mode-note")).toContainText("only the image");
  await expect(page.locator(".message")).toHaveCount(0);
  await page.locator("#mode").selectOption("evidence");
  await expect(page.locator("#mode-note")).toContainText(
    "source-extracted values and connections",
  );
  report.checks.push(
    "Evidence mode changes describe the actual prompt evidence and reset history",
  );
  const healthResponse = await page.request.get(
    new URL("/health", options.url).href,
  );
  const health = await healthResponse.json();
  if (!health.loaded) {
    await expect(page.locator("#chat-status")).toHaveText(
      "Waiting for model to load",
    );
    await expect(page.locator("#question")).toBeDisabled();
    await expect(page.locator("#send")).toBeDisabled();
    report.checks.push(
      "Unloaded model is reported accurately and chat inputs remain disabled",
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
    await expect(page.locator("#schematic-image")).toBeVisible();
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
