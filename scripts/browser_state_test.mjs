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
function harness() {
  const nodes = new Map();
  const requests = [];
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
      this.naturalWidth = 2000;
      this.naturalHeight = 1000;
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
    AbortController,
    DOMException,
    ResizeObserver: class {
      observe() {}
    },
    FileReader: class {
      readAsDataURL(blob) {
        this.result = blob.url;
        this.onload();
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
    `${isolatedSource}\nglobalThis.ui = { state, health, controls, fitView, viewForBox, textWords, locateLabel, executeCall, present, reset, submit, openExample, bind };`,
    context,
  );
  context.ui.bind();
  return { element, requests, ui: context.ui };
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
    third.messages.at(-1).content.endsWith("User question:\nWhere is R999?"),
  );
  const context = JSON.parse(third.messages.at(-1).content.split("\n")[1]);
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
