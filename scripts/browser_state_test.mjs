/** CPU state-transition tests for the shipped browser script.
 * DOM, HTTP and image decoding are controlled boundaries, not model integration.
 * Run: node scripts/browser_state_test.mjs
 * Actual browser layout and PDF rendering checks remain in browser_smoke.mjs.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(
  new URL("../schematic_model/web/app.js", import.meta.url),
  "utf8",
);
const comparisonSource = readFileSync(
  new URL("../schematic_model/web/comparison.js", import.meta.url),
  "utf8",
);
assert.match(source, /\ninit\(\);\s*$/u);
const isolatedSource = source.replace(/\ninit\(\);\s*$/u, "");
const drain = async () => {
  for (let index = 0; index < 8; index += 1)
    await new Promise((resolve) => setImmediate(resolve));
};
const TOOLS = {
  tools: [{ type: "function", function: { name: "show_label" } }],
  instructions: "You control the schematic viewer.",
};
const REGIONS = {
  parts: { R1: [100, 100, 200, 200], U1: [400, 400, 600, 600] },
  nets: { SCL: [10, 700, 300, 720] },
};
/** Values created inside the vm realm compare by JSON shape. */
const plain = (value) => JSON.parse(JSON.stringify(value));
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function response(value, status = 200) {
  return {
    ok: status < 400,
    status,
    text: async () => JSON.stringify(value),
  };
}
function toolReply(name, args, id = "call-1") {
  return {
    choices: [
      {
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id,
              type: "function",
              function: { name, arguments: JSON.stringify(args) },
            },
          ],
        },
        finish_reason: "tool_calls",
      },
    ],
  };
}
function answer(text) {
  return {
    choices: [
      { message: { role: "assistant", content: text }, finish_reason: "stop" },
    ],
  };
}
function harness({ decodes = new Map() } = {}) {
  const nodes = new Map();
  const requests = [];
  const resizeCallbacks = [];
  class ClassList {
    constructor() {
      this.names = new Set();
    }
    toggle(name, enabled) {
      if (enabled ?? !this.names.has(name)) this.names.add(name);
      else this.names.delete(name);
    }
    add(name) {
      this.names.add(name);
    }
    remove(name) {
      this.names.delete(name);
    }
    contains(name) {
      return this.names.has(name);
    }
  }
  class Element {
    constructor(id = "") {
      Object.assign(this, {
        id,
        value: "",
        disabled: false,
        hidden: false,
        textContent: "",
        children: [],
        listeners: {},
        attributes: {},
        dataset: {},
        naturalWidth: 0,
        naturalHeight: 0,
        clientWidth: 1000,
        clientHeight: 800,
      });
      this.classList = new ClassList();
      this.style = { setProperty: (name, value) => (this.style[name] = value) };
    }
    addEventListener(name, listener) {
      (this.listeners[name] ??= []).push(listener);
    }
    setAttribute(name, value) {
      this.attributes[name] = value;
    }
    replaceChildren(...children) {
      this.children = [];
      this.append(...children);
    }
    append(...children) {
      for (const child of children) {
        child.parent = this;
        this.children.push(child);
      }
    }
    querySelector(selector) {
      if (selector === ".empty-state")
        return this.children.find((child) => child.className === "empty-state");
      if (selector === "summary")
        return (
          this.children.find((child) => child.tagName === "summary") || null
        );
      return null;
    }
    querySelectorAll() {
      return [];
    }
    remove() {
      if (this.parent)
        this.parent.children = this.parent.children.filter(
          (child) => child !== this,
        );
    }
    scrollIntoView() {}
    focus() {}
    close() {}
    async decode() {
      if (decodes.has(this.src)) await decodes.get(this.src).promise;
      this.naturalWidth = 2000;
      this.naturalHeight = 1000;
    }
    getContext() {
      return { fillRect() {} };
    }
    toDataURL() {
      return this.image || "data:image/png;base64,AAAA";
    }
    getBoundingClientRect() {
      return { left: 0, top: 0 };
    }
  }
  const element = (id) => {
    if (!nodes.has(id)) nodes.set(id, new Element(id));
    return nodes.get(id);
  };
  element("mode").value = "evidence";
  const document = {
    getElementById: element,
    createElement: (tagName) => Object.assign(new Element(), { tagName }),
    body: new Element("body"),
  };
  const context = {
    document,
    crypto: { randomUUID: () => `request-${requests.length}` },
    AbortController,
    structuredClone,
    DOMException,
    ResizeObserver: class {
      constructor(callback) {
        resizeCallbacks.push(callback);
      }
      observe() {}
    },
    FileReader: class {
      readAsDataURL(blob) {
        const finish = () => {
          this.result = blob.url;
          this.onload();
        };
        if (blob.read) blob.read.promise.then(finish, () => this.onerror());
        else finish();
      }
    },
    fetch: (path, options) => {
      const job = { ...deferred(), path, options };
      requests.push(job);
      return job.promise;
    },
    window: {
      setInterval() {},
      setTimeout() {
        return 1;
      },
      clearTimeout() {},
    },
  };
  vm.runInNewContext(
    `${isolatedSource}\n${comparisonSource}\nglobalThis.ui = { comparison, resetComparison, submitComparison, comparisonControls, state, health, controls, fitView, viewForBox, textWords, locateLabel, executeCall, present, reset, submit, openExample, openFile, openPdfPage, actionChip, firstMessages, stopRequest, renderPages, accumulateMetrics, renderMetrics, questionWithViewerContext, setPdfLibrary: (library) => { pdfLibrary = library; }, bind };`,
    context,
  );
  context.ui.bind();
  return { element, requests, resizeCallbacks, ui: context.ui };
}

function pdfFixture(h, { pages = 1, render } = {}) {
  const requested = [];
  const document = {
    numPages: pages,
    async getPage(number) {
      requested.push(number);
      return {
        getViewport: ({ scale }) => ({
          width: 100 * scale,
          height: 50 * scale,
          scale,
          transform: [scale, 0, 0, scale, 0, 0],
        }),
        render: ({ canvas }) => {
          canvas.image = `data:image/png;base64,PAGE${number}`;
          return { promise: render ? render(number) : Promise.resolve() };
        },
        getTextContent: async () => ({ items: [] }),
      };
    },
  };
  h.ui.setPdfLibrary({
    getDocument: () => ({ promise: Promise.resolve(document) }),
  });
  return {
    requested,
    file: {
      name: "schematic.pdf",
      type: "application/pdf",
      size: 1,
      arrayBuffer: async () => new ArrayBuffer(1),
    },
  };
}
function exampleDoc(overrides = {}) {
  return {
    kind: "example",
    id: "example-1",
    title: "Test board",
    page: 1,
    meta: "Test",
    sourceUrl: "https://example.invalid/source",
    image: "data:image/png;base64,AAAA",
    width: 2000,
    height: 1000,
    regions: REGIONS,
    words: [],
    evidence: { bom: [], nets: {} },
    suggestions: ["Show me R1."],
    ...overrides,
  };
}
function seedReady(h, doc = exampleDoc()) {
  h.ui.state.health = { model: "test-model", loaded: true, busy: false };
  h.ui.state.tools = TOOLS;
  h.ui.present(doc);
  h.ui.controls();
}
async function send(h, question) {
  h.element("question").value = question;
  const done = h.ui.submit({ preventDefault() {} });
  await drain();
  return { done };
}
function completion(h, index) {
  const requests = h.requests.filter(
    (request) => request.path === "/v1/chat/completions",
  );
  return requests[index];
}
function body(request) {
  return JSON.parse(request.options.body);
}
async function finish(h, { done }) {
  await drain();
  const healthRequest = h.requests.find(
    (request) => request.path === "/health" && !request.settled,
  );
  if (healthRequest) {
    healthRequest.settled = true;
    healthRequest.resolve(
      response({ model: "test-model", loaded: true, busy: false }),
    );
  }
  await done;
}

test("fit and box views keep the target centered and bounded", () => {
  const h = harness();
  const fit = h.ui.fitView(1000, 800, 2000, 1000);
  assert.ok(fit.scale > 0 && fit.scale < 1);
  assert.ok(fit.x >= 0 && fit.y >= 96);
  const view = h.ui.viewForBox([400, 400, 600, 600], 1000, 800, 2000, 1000);
  const centerX = 0.5 * 2000 * view.scale + view.x;
  const centerY = 0.5 * 1000 * view.scale + view.y;
  assert.ok(Math.abs(centerX - 500) < 1e-6 && Math.abs(centerY - 400) < 1e-6);
  const tiny = h.ui.viewForBox([500, 500, 501, 501], 1000, 800, 2000, 1000);
  assert.ok(tiny.scale <= 4, "a tiny box never zooms past 4x");
});

test("PDF text items become word boxes, including rotated labels", () => {
  const h = harness();
  const identity = [1, 0, 0, -1, 0, 1000];
  const words = h.ui.textWords(
    [
      { str: "R1 10k", width: 60, transform: [10, 0, 0, 10, 100, 900] },
      { str: "SCL", width: 30, transform: [0, 10, -10, 0, 500, 500] },
      { str: "  ", width: 5, transform: [10, 0, 0, 10, 0, 0] },
    ],
    identity,
    1,
    1000,
    1000,
  );
  assert.deepEqual([...words.map((word) => word.text)], ["R1", "10k", "SCL"]);
  const [r1, value, scl] = words.map((word) => word.box);
  assert.ok(r1[0] < r1[2] && r1[1] < r1[3] && r1[2] <= value[0]);
  assert.ok(scl[3] - scl[1] > scl[2] - scl[0], "vertical text is tall");
});

test("late upload reads and decodes cannot replace the active schematic", async (t) => {
  for (const delayed of ["read", "decode"]) {
    await t.test(delayed, async () => {
      const wait = deferred();
      const oldImage = "data:image/png;base64,OLD";
      const newImage = "data:image/png;base64,NEW";
      const h = harness({
        decodes: delayed === "decode" ? new Map([[oldImage, wait]]) : new Map(),
      });
      seedReady(h);
      const old = h.ui.openFile({
        name: "old.png",
        type: "image/png",
        size: 1,
        url: oldImage,
        ...(delayed === "read" ? { read: wait } : {}),
      });
      await drain();
      await h.ui.openFile({
        name: "new.png",
        type: "image/png",
        size: 1,
        url: newImage,
      });
      h.element("question").value = "Keep this next question";
      const view = plain(h.ui.state.view);
      wait.resolve();
      await old;
      assert.equal(h.ui.state.doc.title, "new.png");
      assert.equal(h.ui.state.doc.image, newImage);
      assert.equal(h.element("sheet").src, newImage);
      assert.equal(h.element("sheet").hidden, false);
      assert.equal(h.element("question").value, "Keep this next question");
      assert.deepEqual(plain(h.ui.state.view), view);
      assert.equal(h.element("error").hidden, true);
    });
  }
});

test("an initial PDF render failure is visible and its page can be retried", async () => {
  const h = harness();
  seedReady(h);
  let fail = true;
  const pdf = pdfFixture(h, {
    render: () =>
      fail
        ? Promise.reject(new Error("Page render failed"))
        : Promise.resolve(),
  });
  await h.ui.openFile(pdf.file);
  assert.equal(h.ui.state.doc, null);
  assert.equal(h.element("sheet").hidden, true);
  assert.equal(h.element("error").hidden, false);
  assert.match(h.element("error").textContent, /Page render failed/u);
  assert.match(
    h.element("placeholder-text").textContent,
    /Could not render page 1/u,
  );
  const retry = h
    .element("pages")
    .children.find((button) => button.attributes["aria-label"] === "Page 1");
  assert.ok(retry, "the failed initial page remains reachable for retry");
  assert.equal(h.element("pages").hidden, false);
  fail = false;
  retry.listeners.click[0]();
  await drain();
  assert.deepEqual(pdf.requested, [1, 1]);
  assert.equal(h.ui.state.doc.page, 1);
  assert.equal(h.element("sheet").hidden, false);
  assert.equal(h.element("error").hidden, true);
  assert.equal(h.element("send").disabled, false);
});

test("a stale PDF page failure cannot overwrite a newer upload", async () => {
  const h = harness();
  seedReady(h);
  const wait = deferred();
  const pdf = pdfFixture(h, {
    pages: 2,
    render: (number) => (number === 2 ? wait.promise : Promise.resolve()),
  });
  await h.ui.openFile(pdf.file);
  const second = h
    .element("pages")
    .children.find((button) => button.attributes["aria-label"] === "Page 2");
  second.listeners.click[0]();
  await drain();
  await h.ui.openFile({
    name: "current.png",
    type: "image/png",
    size: 1,
    url: "data:image/png;base64,CURRENT",
  });
  wait.reject(new Error("Obsolete render failed"));
  await drain();
  assert.equal(h.ui.state.doc.title, "current.png");
  assert.equal(h.element("error").hidden, true);
  assert.equal(h.element("pages").hidden, true);
});

test("long PDFs keep bounded navigation while exposing pages beyond 200", async () => {
  const h = harness();
  seedReady(h);
  const pdf = pdfFixture(h, { pages: 10_000 });
  await h.ui.openFile(pdf.file);
  h.ui.controls();
  assert.equal(
    h
      .element("pages")
      .children.find(
        (node) => node.attributes["aria-label"] === "Previous page",
      ).disabled,
    true,
  );
  const click = async (label) => {
    const button = h
      .element("pages")
      .children.find((node) => node.attributes["aria-label"] === label);
    assert.ok(button, `${label} is reachable`);
    assert.equal(button.disabled, false);
    button.listeners.click[0]();
    await drain();
    assert.ok(
      h.element("pages").children.length <= 13,
      "navigation stays bounded",
    );
  };
  await click("Page 10000");
  assert.equal(h.ui.state.doc.page, 10_000);
  h.ui.controls();
  assert.equal(
    h
      .element("pages")
      .children.find((node) => node.attributes["aria-label"] === "Next page")
      .disabled,
    true,
  );
  await click("Previous page");
  assert.equal(h.ui.state.doc.page, 9_999);
  await click("Page 1");
  assert.equal(h.ui.state.doc.page, 1);
  await h.ui.openPdfPage(200);
  await click("Page 201");
  assert.equal(h.ui.state.doc.page, 201);
  assert.equal(
    h
      .element("pages")
      .children.find((node) => node.attributes["aria-label"] === "Page 201")
      .attributes["aria-current"],
    "page",
  );
});

test("keyboard panning changes viewer context and survives resize", () => {
  const h = harness();
  seedReady(h);
  const before = plain(h.ui.state.view);
  let prevented = false;
  h.element("stage").listeners.keydown[0]({
    key: "ArrowRight",
    preventDefault() {
      prevented = true;
    },
  });
  const after = plain(h.ui.state.view);
  assert.equal(after.x, before.x - 60);
  assert.equal(prevented, true);
  assert.equal(h.ui.state.fit, false);
  const context = JSON.parse(
    h.ui.questionWithViewerContext("Read R1").split("\n")[1],
  );
  assert.equal(context.viewport.full_sheet, false);
  for (const callback of h.resizeCallbacks) callback();
  assert.deepEqual(plain(h.ui.state.view), after);
});

test("approval responses with zero model steps preserve measured inference totals", () => {
  const h = harness();
  const measured = {
    model_steps: 1,
    prompt_tokens: 20,
    completion_tokens: 10,
    total_tokens: 30,
    inference_seconds: 2,
    timing_scope: "upstream_wall_time",
  };
  const approval = {
    model_steps: 0,
    prompt_tokens: null,
    completion_tokens: null,
    total_tokens: null,
    inference_seconds: null,
    timing_scope: null,
  };
  h.ui.accumulateMetrics(approval);
  assert.equal(
    h.ui.state.metrics,
    null,
    "an approval alone has no inference measurements",
  );
  h.ui.accumulateMetrics(measured);
  h.ui.accumulateMetrics(approval);
  h.ui.accumulateMetrics(measured);
  h.ui.renderMetrics("Last answer");
  assert.deepEqual(plain(h.ui.state.metrics), {
    responses: 2,
    prompt_tokens: 40,
    completion_tokens: 20,
    total_tokens: 60,
    inference_seconds: 4,
    timing_scope: "upstream_wall_time",
  });
  assert.match(
    h.element("metrics-summary").textContent,
    /5\.0 tokens\/s.*20 output tokens/u,
  );
  h.ui.accumulateMetrics(undefined);
  assert.equal(
    h.ui.state.metrics.completion_tokens,
    null,
    "missing metrics are still unavailable",
  );
  assert.equal(h.ui.state.metrics.inference_seconds, null);
});

test("labels resolve to parts, then nets, then PDF words; absent labels are null", () => {
  const h = harness();
  const doc = exampleDoc({ words: [{ text: "JP1", box: [1, 2, 3, 4] }] });
  assert.equal(h.ui.locateLabel(doc, "r1").kind, "part");
  assert.equal(h.ui.locateLabel(doc, "SCL").kind, "net");
  assert.deepEqual(plain(h.ui.locateLabel(doc, "jp1").box), [1, 2, 3, 4]);
  assert.equal(h.ui.locateLabel(doc, "R999"), null);
});

test("a model tool call moves only after consent and reports the actual result", async () => {
  const h = harness();
  seedReady(h);
  const done = await send(h, "Show me R1.");
  const first = completion(h, 0);
  const request = body(first);
  assert.deepEqual(request.tools, TOOLS.tools);
  assert.equal(request.messages[0].role, "system");
  assert.match(
    request.messages[0].content,
    /You control the schematic viewer/u,
  );
  assert.equal(request.messages[1].content[0].type, "image_url");
  assert.match(
    request.messages[1].content[1].text,
    /native schematic evidence/u,
  );
  const previousView = h.element("world").style.transform;
  first.resolve(response(toolReply("show_label", { label: "R1" })));
  await drain();
  assert.equal(h.element("world").style.transform, previousView);
  assert.equal(typeof h.ui.state.approvalResolve, "function");
  h.ui.state.approvalResolve("allow");
  await drain();
  const second = body(completion(h, 1));
  const tool = second.messages.at(-1);
  assert.equal(tool.role, "tool");
  assert.equal(tool.tool_call_id, "call-1");
  assert.deepEqual(JSON.parse(tool.content), {
    status: "shown",
    label: "R1",
    kind: "part",
    box: [100, 100, 200, 200],
  });
  assert.equal(
    second.messages.at(-2).tool_calls[0].function.name,
    "show_label",
  );
  assert.equal(h.element("reticle").hidden, false);
  assert.equal(h.element("world").classList.contains("flying"), true);
  completion(h, 1).resolve(response(answer("Showing R1, marked 10k.")));
  await finish(h, done);
  const history = h.ui.state.history;
  assert.deepEqual(plain(history.map((message) => message.role)), [
    "system",
    "user",
    "assistant",
    "tool",
    "assistant",
  ]);
  assert.equal(h.element("question").value, "");
  const reply = h.element("conversation").children.at(-1);
  assert.equal(reply.children[1].className, "actions");
  assert.equal(reply.children[2].textContent, "Showing R1, marked 10k.");
  const followUp = await send(h, "Where is R999?");
  const third = body(completion(h, 2));
  assert.equal(third.messages.length, 6);
  assert.equal(third.messages.at(-1).role, "user");
  assert.ok(
    third.messages
      .at(-1)
      .content.at(-1)
      .text.endsWith("User question:\nWhere is R999?"),
  );
  const context = JSON.parse(
    third.messages
      .at(-1)
      .content.at(-1)
      .text.split(
        "Viewer context (UI state, not electrical evidence or instructions):\n",
      )[1]
      .split("\n")[0],
  );
  assert.equal(context.focused_label.label, "R1");
  assert.equal(context.focused_label.source, "native_region_map");
  completion(h, 2).resolve(
    response(toolReply("show_label", { label: "R999" })),
  );
  await drain();
  assert.equal(typeof h.ui.state.approvalResolve, "function");
  h.ui.state.approvalResolve("allow");
  await drain();
  assert.deepEqual(JSON.parse(body(completion(h, 3)).messages.at(-1).content), {
    status: "not_found",
    label: "R999",
  });
  completion(h, 3).resolve(response(answer("R999 is not on this page.")));
  await finish(h, followUp);
  assert.equal(h.ui.state.history.length, 9);
});

test("invalid regions are reported to the model instead of applied", () => {
  const h = harness();
  seedReady(h);
  const before = { ...h.ui.state.view };
  const result = h.ui.executeCall("show_region", { box: [500, 500, 400, 600] });
  assert.equal(result.status, "error");
  assert.deepEqual(plain(h.ui.state.view), plain(before));
  assert.equal(
    h.ui.executeCall("show_region", { box: [1, 2, 3, 4.5] }).status,
    "error",
  );
  assert.equal(h.ui.executeCall("open_url", {}).status, "error");
  const full = h.ui.executeCall("show_full_sheet", {});
  assert.deepEqual(plain(full), { status: "shown", view: "full_sheet" });
});

test("missing location metadata is not component absence", () => {
  const h = harness();
  seedReady(h);
  h.ui.state.doc.regions = null;
  h.ui.state.doc.words = [];
  const result = h.ui.executeCall("show_label", { label: "R1" });
  assert.equal(result.status, "location_unavailable");
  assert.match(result.reason, /does not mean the component is absent/u);
});

test("endless viewer calls fail and keep the question for retry", async () => {
  const h = harness();
  seedReady(h);
  const done = await send(h, "Show me everything.");
  for (let round = 0; round < 5; round += 1) {
    completion(h, round).resolve(
      response(toolReply("show_full_sheet", {}, `call-${round}`)),
    );
    await drain();
    if (h.ui.state.approvalResolve) {
      h.ui.state.approvalResolve("allow");
      await drain();
    }
  }
  await finish(h, done);
  assert.equal(h.ui.state.history.length, 0);
  assert.match(h.element("error").textContent, /kept moving the view/u);
  assert.equal(h.element("question").value, "Show me everything.");
});

test("a reply for a previous schematic is discarded", async () => {
  const h = harness();
  seedReady(h);
  const done = await send(h, "Show me U1.");
  h.ui.state.selection += 1;
  h.ui.present(exampleDoc({ id: "example-2", title: "Other board" }));
  completion(h, 0).resolve(response(answer("Stale answer.")));
  await finish(h, done);
  assert.equal(h.ui.state.history.length, 0);
  const texts = h
    .element("conversation")
    .children.flatMap((child) =>
      child.children.map((part) => part.textContent),
    );
  assert.ok(!texts.includes("Stale answer."));
});

test("uploads are image-only with the qualification notice; examples follow the mode", () => {
  const h = harness();
  seedReady(h);
  assert.equal(h.element("gate-notice").hidden, true);
  assert.equal(h.element("mode").disabled, false);
  h.element("mode").value = "vision";
  h.ui.reset();
  assert.equal(h.element("gate-notice").hidden, false);
  h.element("mode").value = "evidence";
  h.ui.present(
    exampleDoc({ kind: "pdf", regions: null, evidence: null, id: "upload" }),
  );
  assert.equal(h.element("gate-notice").hidden, false);
  assert.equal(h.element("mode").disabled, true);
  assert.equal(h.element("mode").value, "vision");
  assert.match(
    h.element("mode-note").textContent,
    /no native source evidence/u,
  );
});

test("questions wait for the model, the viewer functions and a page", () => {
  const h = harness();
  h.ui.controls();
  assert.equal(h.element("send").disabled, true);
  seedReady(h);
  assert.equal(h.element("send").disabled, false);
  h.ui.state.tools = null;
  h.ui.controls();
  assert.equal(h.element("send").disabled, true);
  assert.equal(
    h.element("chat-status").textContent,
    "Viewer functions unavailable",
  );
});

test("an older health response cannot overwrite a newer one", async () => {
  const h = harness();
  const old = h.ui.health();
  const latest = h.ui.health();
  const [oldRequest, latestRequest] = h.requests.filter(
    (request) => request.path === "/health",
  );
  latestRequest.resolve(response({ model: "m", loaded: true, busy: false }));
  await latest;
  oldRequest.resolve(response({ model: "m", loaded: false, busy: true }));
  await old;
  assert.equal(h.ui.state.health.loaded, true);
  assert.equal(h.element("model-status").textContent, "m · ready");
});

test("model region labels cannot impersonate verified focus and full sheet clears focus", () => {
  const h = harness();
  seedReady(h);
  h.ui.executeCall("show_label", { label: "R1" });
  assert.equal(h.ui.state.focusedLabel.label, "R1");
  h.ui.executeCall("show_region", { label: "R999", box: [100, 100, 200, 200] });
  assert.equal(h.ui.state.focusedLabel, null);
  h.ui.executeCall("show_label", { label: "R1" });
  h.ui.executeCall("show_full_sheet", {});
  assert.equal(h.ui.state.focusedLabel, null);
  h.ui.executeCall("show_label", { label: "R1" });
  h.ui.present(exampleDoc({ id: "new-board" }));
  assert.equal(h.ui.state.focusedLabel, null);
});

function comparisonReady(h, history = "shared") {
  seedReady(h);
  h.ui.comparison.config = {
    enabled: true,
    max_tokens: 64,
    profiles: [
      { id: "baseline", label: "Baseline", model: "v1" },
      { id: "candidate", label: "Candidate", model: "v3" },
    ],
  };
  h.element("compare-a").value = "baseline";
  h.element("compare-b").value = "candidate";
  h.element("compare-history").value = history;
  h.ui.resetComparison();
}
function comparisonRequests(h) {
  return h.requests.filter(
    (request) => request.path === "/api/comparison/complete",
  );
}
function comparisonAnswer(lane, text = lane ? "R1 is 20k." : "R1 is 10k.") {
  return {
    profile: lane ? "candidate" : "baseline",
    model: lane ? "v3" : "v1",
    answer: text,
    identity: "endpoint_reported",
    model_details: { adapter_sha256: (lane ? "b" : "a").repeat(64) },
    metrics: {
      completion_tokens: 5,
      prompt_tokens: 10,
      inference_seconds: 2,
      tokens_per_second: 2.5,
      timing_scope: "upstream_wall_time",
    },
  };
}
async function comparePair(h, question = "Read R1.") {
  h.element("compare-question").value = question;
  const done = h.ui.submitComparison({ preventDefault() {} });
  await drain();
  return { done };
}
async function completePair(h, pending) {
  const start = comparisonRequests(h).length - 1;
  comparisonRequests(h)[start].resolve(response(comparisonAnswer(0)));
  await drain();
  comparisonRequests(h)[start + 1].resolve(response(comparisonAnswer(1)));
  await finish(h, pending);
}

test("comparison freezes input before either lane, preserves drafts, and requires shared history choice", async () => {
  const h = harness();
  comparisonReady(h);
  const originalView = plain(h.ui.state.view);
  const pending = await comparePair(h);
  assert.equal(comparisonRequests(h).length, 1, "B waits for A");
  assert.equal(
    h.element("question").disabled,
    false,
    "normal draft remains editable",
  );
  h.ui.state.customSystemPrompt = "Changed during generation";
  h.ui.state.doc.image = "new-image";
  h.element("compare-question").value = "Unfinished next question";
  await completePair(h, pending);
  const [a, b] = comparisonRequests(h).map(body);
  assert.deepEqual(a.messages, b.messages, "both inputs were frozen before A");
  assert.equal(
    a.messages[1].content[0].image_url.url,
    "data:image/png;base64,AAAA",
  );
  assert.equal(a.tools, undefined);
  assert.equal(h.element("compare-question").value, "Unfinished next question");
  assert.equal(
    h.element("compare-send").disabled,
    true,
    "shared next turn needs explicit context",
  );
  assert.equal(
    h.ui.state.history.length,
    0,
    "comparison cannot change regular chat",
  );
  assert.deepEqual(
    plain(h.ui.state.view),
    originalView,
    "comparison cannot navigate",
  );
  h.element("compare-use-b").onclick();
  assert.equal(h.element("compare-send").disabled, false);
  const next = await comparePair(h, "And R2?");
  await completePair(h, next);
  const [nextA, nextB] = comparisonRequests(h).slice(2).map(body);
  assert.deepEqual(nextA.messages, nextB.messages);
  assert.equal(nextA.messages[2].content, "R1 is 20k.");
  assert.match(nextA.messages[0].content, /Changed during generation/u);
});

test("independent comparison follow-ups contain only their own previous answer", async () => {
  const h = harness();
  comparisonReady(h, "independent");
  await completePair(h, await comparePair(h));
  assert.equal(h.ui.comparison.awaitingChoice, false);
  const next = await comparePair(h, "And R2?");
  await completePair(h, next);
  const [a, b] = comparisonRequests(h).slice(2).map(body);
  assert.equal(a.messages[2].content, "R1 is 10k.");
  assert.equal(b.messages[2].content, "R1 is 20k.");
  assert.deepEqual(a.messages.at(-1), b.messages.at(-1));
});

test("failed or cancelled comparison never advances histories or overwrites an edited draft", async () => {
  for (const cancel of [false, true]) {
    const h = harness();
    comparisonReady(h, "independent");
    const pending = await comparePair(h);
    h.element("compare-question").value = "Keep my next draft";
    if (cancel) h.ui.comparison.controller.abort();
    comparisonRequests(h)[0].reject(
      cancel
        ? new DOMException("Stopped", "AbortError")
        : new Error("Endpoint unavailable"),
    );
    await finish(h, pending);
    assert.equal(
      comparisonRequests(h).length,
      1,
      "unknown/unfinished A must not launch B",
    );
    assert.equal(h.element("compare-question").value, "Keep my next draft");
    assert.deepEqual(plain(h.ui.comparison.histories), [[], []]);
    assert.equal(h.ui.comparison.pending, false);
    assert.equal(h.ui.state.pending, false);
    assert.match(
      h.element("compare-error").textContent,
      /not added to either history/u,
    );
  }
});

test("comparison prevents duplicate profiles and bounds failed attempts", () => {
  const h = harness();
  comparisonReady(h);
  h.element("compare-b").value = "baseline";
  h.ui.comparisonControls();
  assert.equal(h.element("compare-send").disabled, true);
  h.element("compare-b").value = "candidate";
  h.ui.comparison.attempts = 20;
  h.ui.comparisonControls();
  assert.equal(h.element("compare-send").disabled, true);
  assert.equal(h.element("compare-limit").hidden, false);
  h.element("compare-question").value = "Keep draft across reset";
  h.ui.resetComparison();
  assert.equal(h.ui.comparison.attempts, 0);
  assert.equal(h.element("compare-question").value, "Keep draft across reset");
});

function pdfDoc(page, documentId = "pdf:session-one") {
  return exampleDoc({
    kind: "pdf",
    documentId,
    id: `${documentId}:page:${page}`,
    page,
    image: `data:image/png;base64,PAGE${page}`,
    evidence: null,
    regions: null,
  });
}

test("same PDF page changes preserve discussion and draft but clear focused label", async () => {
  const h = harness();
  seedReady(h, pdfDoc(1));
  h.ui.state.pdf = { documentId: "pdf:session-one", pages: 2 };
  const first = await send(h, "Describe page one.");
  completion(h, 0).resolve(response(answer("Page one has a connector.")));
  await finish(h, first);
  const history = plain(h.ui.state.history);
  const conversation = [...h.element("conversation").children];
  h.element("question").value = "Unfinished follow-up";
  h.ui.state.focusedLabel = { label: "U1", source: "pdf_text" };
  h.ui.state.selection += 1;
  h.ui.state.doc = null; // Page rendering temporarily clears the viewer.
  h.ui.present(pdfDoc(2));
  assert.deepEqual(plain(h.ui.state.history), history);
  assert.deepEqual(h.element("conversation").children, conversation);
  assert.equal(h.element("question").value, "Unfinished follow-up");
  assert.equal(h.ui.state.focusedLabel, null);
  assert.equal(h.element("mode").value, "vision");
  const next = await send(h, "What is on this page?");
  const request = body(completion(h, 1));
  const images = request.messages.flatMap((message) =>
    Array.isArray(message.content)
      ? message.content.filter((block) => block.type === "image_url")
      : [],
  );
  assert.deepEqual(images, [
    { type: "image_url", image_url: { url: "data:image/png;base64,PAGE2" } },
  ]);
  assert.match(
    request.messages[1].content[0].text,
    /Historical page image omitted/u,
  );
  assert.match(request.messages[1].content[1].text, /Current source page: 1/u);
  assert.match(
    request.messages.at(-1).content[1].text,
    /Current source page: 2/u,
  );
  assert.equal(request.messages[2].content, "Page one has a connector.");
  completion(h, 1).resolve(response(answer("Page two has an amplifier.")));
  await finish(h, next);
  assert.equal(h.ui.state.history.length, 5);
});

test("reopening a same-named PDF creates a fresh conversation", () => {
  const h = harness();
  seedReady(h, pdfDoc(1));
  h.ui.state.history = [{ role: "assistant", content: "Old file answer" }];
  h.element("question").value = "Old file draft";
  h.ui.present(pdfDoc(1, "pdf:new-upload"));
  assert.equal(h.ui.state.history.length, 0);
  assert.equal(h.element("question").value, "");
});

test("historical viewer actions cannot replay coordinates on a different page", () => {
  const h = harness();
  seedReady(h, pdfDoc(1));
  const chip = h.ui.actionChip({
    name: "show_region",
    args: { box: [100, 100, 200, 200] },
    result: { status: "shown", box: [100, 100, 200, 200] },
  });
  h.ui.present(pdfDoc(2));
  const view = h.element("world").style.transform;
  chip.listeners.click[0]();
  assert.equal(h.element("world").style.transform, view);
  assert.match(h.element("error").textContent, /Open source page 1/u);
});

test("Stop cancels the exact HTTP request before abort and waits for idle health", async () => {
  const h = harness();
  seedReady(h);
  h.ui.state.health.limits = { max_tokens: 4096 };
  const pending = await send(h, "Review this page.");
  const request = completion(h, 0);
  const requestId = body(request).request_id;
  assert.ok(requestId);
  assert.equal(body(request).max_tokens, 4096);
  request.options.signal.addEventListener("abort", () => {
    assert.equal(h.requests.at(-1).path, "/api/jobs/cancel");
    request.reject(new DOMException("Stopped", "AbortError"));
  });
  h.element("question").value = "Draft typed while waiting";
  h.ui.stopRequest();
  const cancel = h.requests.find((job) => job.path === "/api/jobs/cancel");
  assert.deepEqual(body(cancel), { request_id: requestId });
  assert.equal(cancel.options.signal, undefined);
  cancel.resolve(response({ cancel_requested: true, busy: false }));
  await drain();
  assert.equal(h.ui.state.health.busy, true);
  assert.equal(h.element("send").disabled, true);
  const poll = h.requests.find((job) => job.path === "/health");
  poll.resolve(response({ model: "test-model", loaded: true, busy: true }));
  await pending.done;
  assert.equal(h.ui.state.pending, false);
  assert.equal(h.element("send").disabled, true);
  assert.equal(h.element("question").value, "Draft typed while waiting");
  const idle = h.ui.health();
  h.requests
    .at(-1)
    .resolve(response({ model: "test-model", loaded: true, busy: false }));
  await idle;
  assert.equal(h.element("send").disabled, false);
});

test("each HTTP tool round uses a distinct cancellation token", async () => {
  const h = harness();
  seedReady(h);
  const pending = await send(h, "Show R1.");
  const initial = body(completion(h, 0)).request_id;
  completion(h, 0).resolve(response(toolReply("show_label", { label: "R1" })));
  await drain();
  assert.equal(h.ui.state.requestId, null);
  h.ui.state.approvalResolve("allow");
  await drain();
  const resumed = body(completion(h, 1)).request_id;
  assert.ok(resumed);
  assert.notEqual(resumed, initial);
  completion(h, 1).resolve(response(answer("Showing R1.")));
  await finish(h, pending);
  assert.equal(h.ui.state.requestId, null);
});

test("gateway viewer continuation sends the exact result with its single-use token", async () => {
  const h = harness();
  seedReady(h);
  h.ui.state.agent = { enabled: true };
  const gateway = (index) =>
    h.requests.filter((request) => request.path === "/api/agent/chat")[index];
  const pending = await send(h, "Show me R1.");
  const before = h.element("world").style.transform;
  gateway(0).resolve(
    response({
      pending_approval: {
        id: "approval-one",
        tool: "show_label",
        arguments: { label: "R1" },
        destination: "schematic viewer",
      },
    }),
  );
  await drain();
  assert.equal(h.element("world").style.transform, before);
  h.ui.state.approvalResolve("allow");
  await drain();
  assert.equal(body(gateway(1)).approval_id, "approval-one");
  gateway(1).resolve(
    response({
      ...toolReply("show_label", { label: "R1" }),
      continuation_id: "viewer-result-one",
      viewer_authorization: {
        tool_call_id: "call-1",
        tool: "show_label",
        arguments: { label: "R1" },
      },
    }),
  );
  await drain();
  const resumed = body(gateway(2));
  const { request_id, ...continuation } = resumed;
  assert.deepEqual(continuation, {
    continuation_id: "viewer-result-one",
    tool_call_id: "call-1",
    tool: "show_label",
    arguments: { label: "R1" },
    result: {
      status: "shown",
      label: "R1",
      kind: "part",
      box: [100, 100, 200, 200],
    },
  });
  assert.equal(
    new Set([
      body(gateway(0)).request_id,
      body(gateway(1)).request_id,
      request_id,
    ]).size,
    3,
  );
  assert.notEqual(h.element("world").style.transform, before);
  gateway(2).resolve(
    response(answer("Stopped at the configured agent step limit.")),
  );
  await finish(h, pending);
  assert.equal(h.ui.state.history.at(-2).role, "tool");
  assert.equal(
    h.ui.state.history.at(-1).content,
    "Stopped at the configured agent step limit.",
  );
});

test("gateway viewer calls without a continuation fail before moving the view", async () => {
  const h = harness();
  seedReady(h);
  h.ui.state.agent = { enabled: true };
  const before = h.element("world").style.transform;
  const pending = await send(h, "Show me R1.");
  h.requests
    .find((request) => request.path === "/api/agent/chat")
    .resolve(response(toolReply("show_label", { label: "R1" })));
  await finish(h, pending);
  assert.equal(h.element("world").style.transform, before);
  assert.equal(h.ui.state.history.length, 0);
  assert.equal(h.element("question").value, "Show me R1.");
  assert.equal(h.ui.state.approvalResolve, null);
  assert.equal(
    h.requests.filter((request) => request.path === "/api/agent/chat").length,
    1,
  );
});
