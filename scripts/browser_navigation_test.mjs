/** Scripted multi-turn viewport-consent regressions; no model/GPU inference. */
import { chromium, expect } from "@playwright/test";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(
  process.env.SCHEMATIC_UI_ROOT || "schematic_model/web",
);
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
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const fixture = {
    id: "fixture",
    publisher: "Test fixtures",
    title: "Scripted consent fixture",
    page: 1,
    source_url: "",
    license: "Test only",
    attribution: "Fixture",
    revision: "00000000",
    image_url: "/fixture.svg",
    regions: { parts: { R1: [100, 100, 200, 200], U1: [700, 500, 800, 600] } },
    source_evidence: { parts: ["R1", "U1"] },
    suggested_questions: [],
  };
  const json = (route, body) => route.fulfill({ json: body });
  await page.route("**/health", (route) =>
    json(route, {
      loaded: true,
      busy: false,
      model: "scripted-consent-fixture",
    }),
  );
  await page.route("**/api/view-tools", (route) =>
    json(route, { tools: [], instructions: "Test fixture" }),
  );
  await page.route("**/api/agent/config", (route) =>
    json(route, {
      enabled: true,
      provider: "self_hosted",
      network: "approval_required",
      tools: [
        { name: "show_label", permission: "ask" },
        { name: "web_search", permission: "ask" },
      ],
    }),
  );
  await page.route("**/api/examples", (route) =>
    json(route, { examples: [fixture] }),
  );
  await page.route("**/api/examples/fixture", (route) => json(route, fixture));
  await page.route("**/fixture.svg", (route) =>
    route.fulfill({
      contentType: "image/svg+xml",
      body: '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="800"><rect width="1000" height="800" fill="white"/><text x="100" y="100">R1</text><text x="700" y="500">U1</text></svg>',
    }),
  );
  const received = [];
  const call = (id, label) => ({
    id,
    type: "function",
    function: { name: "show_label", arguments: JSON.stringify({ label }) },
  });
  const calls = (...items) => ({
    choices: [
      { message: { role: "assistant", content: null, tool_calls: items } },
    ],
  });
  const answer = (text) => ({
    choices: [{ message: { role: "assistant", content: text } }],
  });
  let releaseReview;
  await page.route("**/api/agent/chat", async (route) => {
    const body = route.request().postDataJSON();
    received.push(body);
    if (body.approval_id === "first-navigation")
      return json(route, {
        ...calls(call("first-call", "R1")),
        viewer_authorization: {
          tool_call_id: "first-call",
          tool: "show_label",
          arguments: { label: "R1" },
        },
      });
    if (body.approval_id === "changed-target")
      return json(route, {
        ...calls(call("changed-call", "U1")),
        viewer_authorization: {
          tool_call_id: "changed-call",
          tool: "show_label",
          arguments: { label: "U1" },
        },
      });
    if (body.approval_id === "search")
      return json(route, calls(call("search-drift", "U1")));
    const latest = body.messages.at(-1);
    if (latest.role === "tool")
      return json(route, answer("Turn completed without any further action."));
    const raw =
      typeof latest.content === "string"
        ? latest.content
        : latest.content.find((block) => block.type === "text").text;
    const question = raw.split("User question:\n").at(-1);
    if (question === "Show R1")
      return json(route, {
        pending_approval: {
          id: "first-navigation",
          tool: "show_label",
          arguments: { label: "R1" },
          destination: "schematic viewer",
        },
      });
    if (question === "Show R1 again")
      return json(route, {
        pending_approval: {
          id: "changed-target",
          tool: "show_label",
          arguments: { label: "R1" },
          destination: "schematic viewer",
        },
      });
    if (question === "Search for its datasheet")
      return json(route, {
        pending_approval: {
          id: "search",
          tool: "web_search",
          arguments: { query: "R1 public datasheet" },
          destination: "http://127.0.0.1:8894/search",
        },
      });
    if (question === "Explain the full sheet")
      return json(
        route,
        calls({
          id: "sheet-drift",
          type: "function",
          function: { name: "show_full_sheet", arguments: "{}" },
        }),
      );
    if (question === "Review the power path")
      await new Promise((resolve) => {
        releaseReview = resolve;
      });
    // Simulate a defective/legacy provider replaying an earlier authorization.
    return json(route, {
      ...calls(call("first-call", "U1")),
      viewer_authorization: {
        tool_call_id: "first-call",
        tool: "show_label",
        arguments: { label: "U1" },
      },
    });
  });
  const viewport = () =>
    page.locator("#world").evaluate((element) => element.style.transform);
  const send = async (question) => {
    await page.locator("#question").fill(question);
    await page.locator("#send").click();
  };
  const button = (name) =>
    page
      .getByRole("button", { name, exact: true })
      .filter({ visible: true })
      .last();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await expect(page.locator("#sheet")).toBeVisible();
  await expect(page.locator("#question")).toBeEnabled();
  const initial = await viewport();
  await send("Show R1");
  await expect(button("Allow once")).toBeVisible();
  expect(await viewport()).toBe(initial);
  await expect(page.locator("#tool-timeline")).toContainText(
    "move the schematic view",
  );
  await button("Allow once").click();
  await expect(page.locator("#busy-panel")).toBeHidden();
  expect(await viewport()).not.toBe(initial);
  const focused = await viewport();

  for (const question of [
    "Tell me about U1",
    "What is its value?",
    "Compare it with R1",
  ]) {
    await send(question);
    await expect(button("Allow once")).toBeVisible();
    expect(await viewport()).toBe(focused);
    await button("Deny").click();
    await expect(page.locator("#busy-panel")).toBeHidden();
    expect(await viewport()).toBe(focused);
  }
  await send("Explain the full sheet");
  await expect(button("Allow once")).toBeVisible();
  expect(await viewport()).toBe(focused);
  await button("Deny").click();
  await expect(page.locator("#busy-panel")).toBeHidden();
  expect(await viewport()).toBe(focused);
  await send("Search for its datasheet");
  await expect(button("Allow once")).toBeVisible();
  expect(await viewport()).toBe(focused);
  await button("Allow once").click();
  await expect(page.locator("#tool-timeline")).toContainText(
    "move the schematic view",
  );
  expect(await viewport()).toBe(focused);
  await button("Deny").click();
  await expect(page.locator("#busy-panel")).toBeHidden();
  expect(await viewport()).toBe(focused);

  await send("Review the power path");
  await expect.poll(() => Boolean(releaseReview)).toBe(true);
  await page.locator("#question").fill("Keep this draft while reviewing");
  await page.locator("#question").press("ArrowUp");
  await page.locator("#question").press("ArrowDown");
  expect(await viewport()).toBe(focused);
  releaseReview();
  await expect(button("Allow once")).toBeVisible();
  await expect(page.locator("#question")).toBeEnabled();
  await expect(page.locator("#question")).toHaveValue(
    "Keep this draft while reviewing",
  );
  expect(await viewport()).toBe(focused);
  await button("Deny").click();
  await expect(page.locator("#busy-panel")).toBeHidden();
  expect(await viewport()).toBe(focused);
  await send("Show R1 again");
  await expect(button("Allow once")).toBeVisible();
  await button("Allow once").click();
  await expect(page.locator("#tool-timeline")).toContainText('"U1"');
  expect(await viewport()).toBe(focused);
  await button("Deny").click();
  await expect(page.locator("#busy-panel")).toBeHidden();
  expect(await viewport()).toBe(focused);
  await send("Explain startup without changing the view");
  await expect(button("Allow once")).toBeVisible();
  await page.locator("#stop").click();
  await expect(page.locator("#busy-panel")).toBeHidden();
  expect(await viewport()).toBe(focused);
  const lastConversation = received
    .filter((body) => body.messages)
    .at(-1).messages;
  expect(
    lastConversation.some(
      (message) =>
        message.role === "assistant" &&
        message.tool_calls?.some((tool) => tool.function.name === "show_label"),
    ),
  ).toBe(true);
  expect(
    lastConversation
      .filter((message) => message.role === "tool")
      .some((message) => JSON.parse(message.content).status === "denied"),
  ).toBe(true);
  expect(errors).toEqual([]);
  console.log(
    "PASS: historical navigation, discussion, values, comparison, approved search, review and editable drafts cannot move viewport without exact one-shot viewer consent.",
  );
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
