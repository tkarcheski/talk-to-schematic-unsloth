/* cspell:words valuenow */
/** Scripted browser contract tests. No real model inference is performed. */
import { chromium, expect } from "@playwright/test";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import path from "node:path";
const root = path.resolve("schematic_model/web");
const out = path.resolve(
  process.env.SCHEMATIC_TEST_OUTPUT || "/tmp/schematic-agentic-ui-preview",
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
    const data = await readFile(path.join(root, file));
    response.setHeader(
      "Content-Type",
      file.endsWith(".js")
        ? "text/javascript"
        : file.endsWith(".css")
          ? "text/css"
          : "text/html",
    );
    response.end(data);
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
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const fixture = {
    id: "fixture",
    title: "Power controller · scripted UI fixture",
    page: 1,
    attribution: "UI test fixture",
    license: "Test only",
    revision: "00000000",
    source_url: "",
    image_url: "/fixture.svg",
    regions: { parts: { U1: [370, 270, 620, 670], R1: [100, 100, 200, 200] } },
    source_evidence: { parts: ["U1"] },
    suggested_questions: ["Explain the power path", "Show me U1."],
  };
  const fulfill = (route, json) => route.fulfill({ json });
  await page.route("**/health", (route) =>
    fulfill(route, {
      loaded: true,
      busy: false,
      model: "Self-hosted · UI test fixture",
      model_details: {
        served_model_name: "schematic-v1",
        backend: "unsloth-library",
        backend_version: "2026.8.22",
        adapter_sha256: "a".repeat(64),
        base_model_sha256: "b".repeat(64),
        base_model_name: "Qwen3.5-4B",
        model_fingerprint: "c".repeat(64),
      },
    }),
  );
  await page.route("**/api/view-tools", (route) =>
    fulfill(route, { tools: [], instructions: "Viewer fixture" }),
  );
  await page.route("**/api/agent/config", (route) =>
    fulfill(route, {
      enabled: true,
      provider: "self_hosted",
      endpoint: "http://127.0.0.1:8080/v1",
      network: "approval_required",
      max_steps: 8,
      tools: [
        { name: "show_label", permission: "allow" },
        { name: "schematic_query", permission: "allow" },
        { name: "web_search", permission: "ask" },
      ],
    }),
  );
  await page.route("**/api/examples", (route) =>
    fulfill(route, { examples: [{ ...fixture, publisher: "Test examples" }] }),
  );
  await page.route("**/api/examples/fixture", (route) =>
    fulfill(route, fixture),
  );
  await page.route("**/fixture.svg", (route) =>
    route.fulfill({
      contentType: "image/svg+xml",
      body: `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" viewBox="0 0 1200 650"><rect width="1200" height="650" fill="white"/><g stroke="#111" stroke-width="3" fill="none"><path d="M80 230H444M744 230H1120M80 470H1120M180 230V330M180 370V470M162 330H198M162 370H198M930 230V330M930 370V470M912 330H948M912 370H948"/><rect x="444" y="175" width="300" height="260"/></g><g font-family="monospace" font-size="23" fill="#111"><text x="75" y="210">VIN</text><text x="1010" y="210">3V3</text><text x="520" y="265">U1</text><text x="488" y="310">REGULATOR</text><text x="210" y="355">C1 10µF</text><text x="950" y="355">C2 10µF</text><text x="80" y="505">GND</text><text x="80" y="590">SCRIPTED UI FIXTURE — NOT AN ELECTRICAL DESIGN</text></g></svg>`,
    }),
  );
  const sent = [];
  let decision = null;
  let searchContext = null;
  let requests = 0;
  let releaseFailure;
  const failing = new Promise((resolve) => {
    releaseFailure = resolve;
  });
  let releaseThinking;
  const thinking = new Promise((resolve) => {
    releaseThinking = resolve;
  });
  await page.route("**/api/agent/chat", (route) => {
    requests += 1;
    const body = route.request().postDataJSON();
    sent.push(body);
    const raw = body.messages?.at(-1)?.content;
    const last =
      typeof raw === "string" ? raw.split("User question:\n").at(-1) : raw;
    if (last === "Show me R1.")
      return fulfill(route, {
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "viewer-fixture",
                  type: "function",
                  function: {
                    name: "show_label",
                    arguments: JSON.stringify({ label: "R1" }),
                  },
                },
              ],
            },
          },
        ],
      });
    if (body.messages?.at(-1)?.tool_call_id === "viewer-fixture")
      return fulfill(route, {
        choices: [
          { message: { role: "assistant", content: "R1 is now focused." } },
        ],
      });
    if (last === "Do a web search for that part") {
      searchContext = JSON.parse(raw.split("\n")[1]);
      return fulfill(route, {
        pending_approval: {
          id: "focused-search",
          tool: "web_search",
          arguments: { query: "R1 resistor datasheet" },
          destination: "https://search.example.invalid/search",
        },
      });
    }
    if (last === "Trigger failure")
      return failing.then(() =>
        route.fulfill({
          status: 503,
          json: { error: { message: "Scripted unavailable service" } },
        }),
      );
    if (last === "Metrics unavailable")
      return fulfill(route, {
        trace: [
          {
            id: "trace-fixture",
            kind: "model_response",
            data: {
              content: "No usage supplied by this fixture.",
              reasoning_content:
                "Explicit reasoning supplied by the scripted test provider.",
            },
          },
        ],
        choices: [
          {
            message: {
              role: "assistant",
              content: "No usage supplied by this fixture.",
            },
          },
        ],
      });
    if (body.approval_id) {
      decision = body.decision;
      return fulfill(route, {
        metrics: {
          prompt_tokens: 40,
          completion_tokens: 300,
          total_tokens: 340,
          inference_seconds: 3,
          timing_scope: "model_inference_including_prefill",
        },
        events: [
          {
            tool: "web_search",
            status: body.decision === "allow" ? "completed" : "denied",
            arguments: { query: "regulator datasheet" },
          },
        ],
        choices: [
          {
            message: {
              role: "assistant",
              content:
                body.decision === "allow"
                  ? "Scripted answer: the search completed. Verify the device against your drawing."
                  : "Search denied. I can continue using the schematic evidence.",
            },
          },
        ],
      });
    }
    return thinking.then(() =>
      fulfill(route, {
        metrics: {
          prompt_tokens: 20,
          completion_tokens: 100,
          total_tokens: 120,
          inference_seconds: 2,
          timing_scope: "model_inference_including_prefill",
        },
        events: [
          {
            tool: "schematic_query",
            status: "completed",
            arguments: { query: "U1" },
            result: { label: "U1" },
          },
        ],
        pending_approval: {
          id: "fixture-approval",
          tool: "web_search",
          arguments: { query: "regulator datasheet" },
          destination: "https://search.example.invalid/search",
        },
      }),
    );
  });
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await expect(page).toHaveTitle("Ask the Schematic");
  await expect(page.locator(".brand")).toHaveText("Ask the Schematic");
  await expect(page.locator("#sheet")).toBeVisible();
  await expect(page.locator("#question")).toBeEnabled();
  await page.locator("#open-settings").click();
  await page.locator("#prompt-example").click();
  await expect(page.locator("#system-prompt")).toHaveValue(
    /Review the schematic/u,
  );
  await page.locator("#system-prompt").fill("List issues with evidence.");
  await page.locator("#settings-close").click();
  await page.locator("#question").fill("Find the regulator datasheet");
  await page.locator("#send").click();
  await expect(page.locator("#busy-panel")).toBeVisible();
  await expect(page.locator("#busy-stage")).toHaveText("Thinking…");
  await expect(page.locator("#busy-elapsed")).toHaveText("1s");
  await expect(page.getByRole("progressbar")).toBeVisible();
  await expect(page.locator("#stop")).toBeVisible();
  await page.screenshot({ path: path.join(out, "desktop-thinking.png") });
  await expect(page.locator("#question")).toHaveValue("");
  await page.locator("#question").fill("My next draft");
  await page.locator("#question").press("Enter");
  await expect(page.locator("#question")).toHaveValue("My next draft");
  expect(requests).toBe(1);
  expect(sent[0].messages[0].content).toContain("List issues with evidence.");
  await page.locator("#question").press("ArrowUp");
  await expect(page.locator("#question")).toHaveValue(
    "Find the regulator datasheet",
  );
  await page.locator("#question").press("ArrowDown");
  await expect(page.locator("#question")).toHaveValue("My next draft");
  releaseThinking();
  await expect(
    page.getByRole("button", { name: "Allow once", exact: true }),
  ).toBeVisible();
  await expect(page.locator("#busy-stage")).toHaveText(
    "Waiting for your approval",
  );
  await expect(page.locator("#tool-timeline")).toContainText(
    "https://search.example.invalid/search",
  );
  await expect(page.locator("#tool-timeline")).toContainText(
    "regulator datasheet",
  );
  await page.screenshot({ path: path.join(out, "desktop-approval.png") });
  await page.getByRole("button", { name: "Deny", exact: true }).click();
  await expect(page.locator(".message.assistant")).toContainText(
    "Search denied",
  );
  expect(decision).toBe("deny");
  await expect(page.locator("#question")).toHaveValue("My next draft");
  await expect(page.locator("#metrics-summary")).toHaveText(
    "80.0 tokens/s · 400 output tokens",
  );
  await page.locator("#request-metrics summary").click();
  await expect(page.locator("#metrics-details")).toContainText(
    "Total: 460 tokens",
  );
  await expect(page.locator("#metrics-details")).toContainText(
    "Inference time: 5.00 s",
  );
  await page.screenshot({ path: path.join(out, "desktop-metrics.png") });
  await expect(page.locator("#busy-panel")).toBeHidden();
  const firstTrace = page.locator(".request-trace").first();
  await firstTrace.locator("summary").first().click();
  await expect(firstTrace).toContainText("image bytes omitted");
  await expect(firstTrace).not.toContainText("data:image/");
  await expect(firstTrace).not.toContainText("fixture-approval");
  await expect(firstTrace).toContainText("No reasoning text has been returned");
  await page.locator("#open-settings").click();
  await page.locator("#system-prompt").fill("Answer concisely.");
  await page.locator("#settings-close").click();
  await page.locator("#question").fill("Search once");
  await page.locator("#send").click();
  await page.getByRole("button", { name: "Allow once", exact: true }).click();
  await expect(page.locator(".message.assistant").last()).toContainText(
    "search completed",
  );
  expect(decision).toBe("allow");
  const secondPrompt = sent.filter((body) => body.messages)[1].messages[0]
    .content;
  expect(secondPrompt).toContain("Answer concisely.");
  expect(secondPrompt).not.toContain("List issues with evidence.");
  expect(secondPrompt.match(/User workspace instructions/gu)).toHaveLength(1);
  await page.locator("#open-settings").click();
  await page.locator("#prompt-reset").click();
  await page.locator("#settings-close").click();
  await page.locator("#question").fill("Cancel this search");
  await page.locator("#send").click();
  await expect(
    page.getByRole("button", { name: "Allow once", exact: true }),
  ).toBeVisible();
  expect(
    sent.filter((body) => body.messages)[2].messages[0].content,
  ).not.toContain("User workspace instructions");
  await page.locator("#question").fill("Draft survives cancellation");
  await page.locator("#stop").click();
  await expect(page.locator("#question")).toBeEnabled();
  await expect(page.locator("#error")).toContainText(
    "Server inference may still finish",
  );
  await expect(page.locator("#question")).toHaveValue(
    "Draft survives cancellation",
  );
  await page
    .getByRole("button", { name: "Copy failed question to draft" })
    .last()
    .click();
  await expect(page.locator("#question")).toHaveValue(
    "Draft survives cancellation\n\nCancel this search",
  );
  await page.locator("#question").fill("Trigger failure");
  await page.locator("#send").click();
  await page.locator("#question").fill("Draft survives failure");
  releaseFailure();
  await expect(page.locator("#error")).toContainText(
    "Scripted unavailable service",
  );
  await expect(page.locator("#question")).toHaveValue("Draft survives failure");
  await page.locator("#question").fill("Metrics unavailable");
  await page.locator("#send").click();
  await expect(page.locator("#metrics-summary")).toHaveText(
    "Speed unavailable · Tokens unavailable",
  );
  const reasoningTrace = page.locator(".request-trace").last();
  await reasoningTrace.locator("summary").first().click();
  await expect(reasoningTrace).toContainText(
    "Explicit reasoning supplied by the scripted test provider.",
  );
  await expect(reasoningTrace.locator(".reasoning-availability")).toContainText(
    "Reasoning text returned",
  );
  await page.screenshot({ path: path.join(out, "desktop-request-trace.png") });
  await page.locator("#question").fill("Show me R1.");
  await page.locator("#send").click();
  await page.getByRole("button", { name: "Allow once", exact: true }).click();
  await expect(page.locator(".message.assistant").last()).toContainText(
    "R1 is now focused.",
  );
  await expect(page.locator("#reticle-label")).toHaveText("R1");
  const focusedView = await page.locator("#world").getAttribute("style");
  await page.locator("#question").fill("Do a web search for that part");
  await page.locator("#send").click();
  await expect(
    page.getByRole("button", { name: "Allow once", exact: true }),
  ).toBeVisible();
  expect(searchContext.focused_label.label).toBe("R1");
  expect(searchContext.focused_label.source).toBe("native_region_map");
  await expect(page.locator("#tool-timeline")).toContainText(
    "R1 resistor datasheet",
  );
  await expect(page.locator("#world")).toHaveAttribute("style", focusedView);
  await page.screenshot({ path: path.join(out, "desktop-focused-search.png") });
  await page.getByRole("button", { name: "Deny", exact: true }).click();
  await expect(page.locator("#send")).toBeEnabled();
  await page.locator("#open-settings").click();
  await expect(page.locator("#settings-provider")).toContainText("127.0.0.1");
  await expect(page.locator("#model-version")).toHaveText(
    "Adapter aaaaaaaaaaaa",
  );
  await expect(page.locator("#settings-adapter")).toHaveText("a".repeat(64));
  await expect(page.locator("#settings-fingerprint")).toHaveText(
    "c".repeat(64),
  );
  await expect(page.locator("#settings-backend")).toContainText("2026.8.22");
  await page.screenshot({ path: path.join(out, "desktop-settings.png") });
  await page.locator("#settings-close").click();
  await page.locator("#reset").click();
  await page.locator("#question").fill("line one\nline two");
  await page.locator("#question").press("ArrowUp");
  await expect(page.locator("#question")).toHaveValue("line one\nline two");
  await page
    .locator("#question")
    .evaluate((node) => node.setSelectionRange(0, 0));
  await page.locator("#question").press("ArrowUp");
  await expect(page.locator("#question")).toHaveValue(
    "Do a web search for that part",
  );
  await page.locator("#question").press("ArrowDown");
  await expect(page.locator("#question")).toHaveValue("line one\nline two");
  const divider = page.getByRole("separator", { name: "Resize chat width" });
  await divider.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(divider).toHaveAttribute("aria-valuenow", "450");
  const handle = await divider.boundingBox();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + 100);
  await page.mouse.down();
  await page.mouse.move(handle.x - 116, handle.y + 100, { steps: 5 });
  await page.mouse.up();
  await expect(divider).toHaveAttribute("aria-valuenow", "570");
  const savedKeys = await page.evaluate(() => Object.keys(localStorage));
  expect(savedKeys.every((key) => key.toLowerCase().includes("width"))).toBe(
    true,
  );
  await page.reload();
  await expect(page.locator("#sheet")).toBeVisible();
  await expect(divider).toHaveAttribute("aria-valuenow", "570");
  await page.locator("#open-settings").click();
  await page.locator("#chat-width").focus();
  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#chat-width-value")).toHaveText("330 px");
  await page.locator("#reset-width").click();
  await expect(divider).toHaveAttribute("aria-valuenow", "430");
  await page.locator("#settings-close").click();
  for (const [name, width, height] of [
    ["desktop", 1440, 1000],
    ["mobile", 390, 844],
  ]) {
    await page.setViewportSize({ width, height });
    if (width < 860) await expect(divider).toBeHidden();
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: path.join(out, `${name}.png`),
      fullPage: true,
    });
  }
  expect(errors).toEqual([]);
  console.log(
    `Passed scripted agent browser tests; screenshots: ${out}. No real inference tested.`,
  );
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
