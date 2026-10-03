/** CPU state-transition tests for the shipped browser script.
 * DOM, HTTP and image decoding are controlled boundaries, not model integration.
 * Run: node scripts/browser_state_test.mjs
 * Actual browser layout/source/model checks remain in browser_smoke.mjs.
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
const drain = () => new Promise((resolve) => setImmediate(resolve));
const readyHealth = { model: "test-model", loaded: true, busy: false };

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
function response(value) {
  return { ok: true, status: 200, text: async () => JSON.stringify(value) };
}
function detail(id) {
  return {
    id,
    title: `Test board ${id}`,
    page: 1,
    image_url: `/test-image/${id}`,
    source_url: `https://example.invalid/source/${id}`,
    revision: "a".repeat(40),
    attribution: "Test attribution",
    license: "Test license",
    suggested_questions: ["Read C1"],
    source_evidence: { bom: [], nets: {} },
  };
}
function harness() {
  const nodes = new Map();
  const requests = [];
  const decodes = [];
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
        style: {},
        naturalWidth: 0,
        naturalHeight: 0,
        clientWidth: 600,
        clientHeight: 480,
        scrollWidth: 600,
        scrollHeight: 480,
        scrollLeft: 0,
        scrollTop: 0,
      });
      const classes = new Set();
      this.classList = {
        toggle(name, enabled) {
          if (enabled) classes.add(name);
          else classes.delete(name);
        },
        remove: (name) => classes.delete(name),
      };
    }
    get options() {
      return this.children;
    }
    addEventListener(name, listener) {
      (this.listeners[name] ??= []).push(listener);
    }
    fire(name, event = {}) {
      assert.equal(this.listeners[name]?.length, 1);
      return this.listeners[name][0](event);
    }
    setAttribute(name, value) {
      this.attributes[name] = value;
    }
    replaceChildren(...children) {
      this.children = [];
      if (this.id === "example") this.value = "";
      this.append(...children);
    }
    append(...children) {
      for (const child of children) {
        child.parent = this;
        this.children.push(child);
      }
      if (this.id === "example" && !this.value)
        this.value = this.children[0]?.value || "";
    }
    querySelector(selector) {
      if (selector === ".empty-state")
        return this.children.find((child) => child.className === "empty-state");
      return null;
    }
    remove() {
      if (this.parent)
        this.parent.children = this.parent.children.filter(
          (child) => child !== this,
        );
    }
    scrollIntoView() {}
    focus() {
      document.activeElement = this;
    }
    decode() {
      const job = { ...deferred(), src: this.src };
      decodes.push(job);
      return job.promise;
    }
  }
  const element = (id) => {
    if (!nodes.has(id)) nodes.set(id, new Element(id));
    return nodes.get(id);
  };
  const document = {
    getElementById: element,
    createElement: () => new Element(),
  };
  const placeholder = new Element();
  placeholder.value = "Loading examples…";
  element("example").append(placeholder);
  element("mode").value = "evidence";
  element("retry-examples").hidden = true;
  const context = {
    document,
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
    window: { setInterval() {} },
  };
  vm.runInNewContext(
    `${isolatedSource}\nglobalThis.ui = { state, health, controls, init };`,
    context,
  );
  return { element, requests, decodes, ui: context.ui };
}
function seedReady(h) {
  h.ui.state.health = { ...readyHealth };
  h.ui.state.example = detail("existing");
  h.ui.state.image = `data:image/png;base64,${Buffer.from("existing").toString("base64")}`;
  h.ui.controls();
}
function choose(h, id) {
  h.element("example").value = id;
  return h.element("example").fire("change");
}
async function supplyImage(h, detailRequest, id) {
  h.requests[detailRequest].resolve(response(detail(id)));
  await drain();
  const imageRequest = h.requests.find(
    (request) => request.path === `/test-image/${id}`,
  );
  assert.ok(imageRequest);
  imageRequest.resolve({
    ok: true,
    blob: async () => ({
      url: `data:image/png;base64,${Buffer.from(id).toString("base64")}`,
    }),
  });
  await drain();
}
function resolveDecode(h, index = h.decodes.length - 1) {
  const image = h.element("schematic-image");
  assert.equal(h.decodes[index].src, image.src);
  image.naturalWidth = 2400;
  image.naturalHeight = 1600;
  h.decodes[index].resolve();
}

test("older health success cannot overwrite a newer ready response", async () => {
  const h = harness();
  seedReady(h);
  const old = h.ui.health();
  const latest = h.ui.health();
  h.requests[1].resolve(response(readyHealth));
  await latest;
  h.requests[0].resolve(response({ ...readyHealth, busy: true }));
  await old;
  assert.equal(h.ui.state.health.busy, false);
  assert.equal(h.element("question").disabled, false);
});

test("older failed health request cannot replace newer ready state", async () => {
  const h = harness();
  seedReady(h);
  const old = h.ui.health();
  const latest = h.ui.health();
  h.requests[1].resolve(response(readyHealth));
  await latest;
  h.requests[0].reject(new Error("Old network failure"));
  await old;
  assert.equal(h.ui.state.health.loaded, true);
  assert.equal(h.element("question").disabled, false);
});

test("newer unavailable state is not replaced by an older success", async () => {
  const h = harness();
  seedReady(h);
  const old = h.ui.health();
  const latest = h.ui.health();
  h.requests[1].reject(new Error("Current network failure"));
  await latest;
  h.requests[0].resolve(response(readyHealth));
  await old;
  assert.equal(h.ui.state.health, null);
  assert.equal(h.element("question").disabled, true);
});

test("image must decode before chat and zoom become available", async () => {
  const h = harness();
  h.ui.state.health = readyHealth;
  const selection = choose(h, "A");
  await supplyImage(h, 0, "A");
  assert.equal(h.ui.state.image, null);
  assert.equal(h.element("question").disabled, true);
  assert.equal(h.element("schematic-image").hidden, true);
  assert.equal(h.element("zoom-in").disabled, true);
  resolveDecode(h);
  await selection;
  assert.equal(h.ui.state.example.id, "A");
  assert.equal(h.element("question").disabled, false);
  assert.equal(h.element("schematic-image").hidden, false);
  assert.equal(h.element("zoom-fit").disabled, false);
});

test("decode failure reports an error and keeps source/chat unavailable", async () => {
  const h = harness();
  h.ui.state.health = readyHealth;
  const selection = choose(h, "A");
  await supplyImage(h, 0, "A");
  h.decodes[0].reject(new Error("Invalid image encoding"));
  await selection;
  assert.equal(h.ui.state.example, null);
  assert.equal(h.ui.state.image, null);
  assert.equal(h.element("question").disabled, true);
  assert.equal(h.element("source").hidden, true);
  assert.equal(h.element("schematic-image").hidden, true);
  assert.match(h.element("error").textContent, /could not be decoded/u);
});

test("stale board HTTP failure cannot replace a newer decoded board", async () => {
  const h = harness();
  const old = choose(h, "A");
  const latest = choose(h, "B");
  await supplyImage(h, 1, "B");
  resolveDecode(h);
  await latest;
  h.requests[0].reject(new Error("Stale board failure"));
  await old;
  assert.equal(h.ui.state.example.id, "B");
  assert.equal(h.element("error").hidden, true);
});

test("stale decode rejection cannot hide a newer board or add its error", async () => {
  const h = harness();
  const old = choose(h, "A");
  await supplyImage(h, 0, "A");
  const latest = choose(h, "B");
  await supplyImage(h, 2, "B");
  resolveDecode(h, 1);
  await latest;
  h.decodes[0].reject(new Error("Old decode cancelled by new source"));
  await old;
  assert.equal(h.ui.state.example.id, "B");
  assert.equal(h.element("schematic-image").hidden, false);
  assert.equal(h.element("error").hidden, true);
});

test("stale decoded board cannot replace the current selection", async () => {
  const h = harness();
  const old = choose(h, "A");
  await supplyImage(h, 0, "A");
  const latest = choose(h, "B");
  await supplyImage(h, 2, "B");
  resolveDecode(h, 1);
  await latest;
  h.decodes[0].resolve();
  await old;
  assert.equal(h.ui.state.example.id, "B");
  assert.equal(h.element("drawing-heading").textContent, "Test board B");
  assert.equal(h.element("source-link").href, detail("B").source_url);
});

test("mode changed while a board loads is retained when decoding finishes", async () => {
  const h = harness();
  const selection = choose(h, "A");
  await supplyImage(h, 0, "A");
  h.element("mode").value = "vision";
  h.element("mode").fire("change");
  resolveDecode(h);
  await selection;
  assert.equal(h.element("mode").value, "vision");
  assert.match(h.element("mode-note").textContent, /only the image/u);
  assert.equal(h.ui.state.history.length, 0);
});

test("pending chat locks selectors; failure keeps draft and existing history", async () => {
  const h = harness();
  seedReady(h);
  const existing = [
    { role: "system", content: "Read the source" },
    { role: "user", content: "Earlier question" },
    { role: "assistant", content: "Earlier answer" },
  ];
  h.ui.state.history = existing;
  h.element("question").value = "Read C1";
  const submit = h.element("chat-form").fire("submit", { preventDefault() {} });
  for (const id of ["example", "mode", "reset", "question", "send"])
    assert.equal(h.element(id).disabled, true, id);
  h.requests[0].reject(new Error("Model unavailable"));
  await drain();
  h.requests[1].resolve(response(readyHealth));
  await submit;
  assert.equal(h.ui.state.history, existing);
  assert.equal(h.element("question").value, "Read C1");
  assert.equal(h.ui.state.pending, false);
  assert.equal(h.element("question").disabled, false);
  assert.equal(h.element("conversation").children.length, 0);
});

test("failed catalog disables placeholder and explicit retry loads a board", async () => {
  const h = harness();
  const initialization = h.ui.init();
  h.requests[0].resolve(response(readyHealth));
  await drain();
  assert.equal(h.element("example").options.length, 0);
  assert.equal(h.element("example").disabled, true);
  h.requests[1].reject(new Error("Catalog unavailable"));
  await initialization;
  assert.equal(h.element("example").options.length, 0);
  assert.equal(h.element("example").disabled, true);
  assert.equal(h.element("retry-examples").hidden, false);
  const retry = h.element("retry-examples").fire("click");
  assert.equal(h.element("retry-examples").hidden, true);
  h.requests[2].resolve(response({ examples: [detail("A")] }));
  await drain();
  await supplyImage(h, 3, "A");
  resolveDecode(h);
  await retry;
  assert.equal(h.ui.state.example.id, "A");
  assert.equal(h.element("example").options.length, 1);
  assert.equal(h.element("error").hidden, true);
  assert.equal(h.element("question").disabled, false);
});

test("focused viewport keyboard zoom retains modifier shortcuts and Fit", () => {
  const h = harness();
  seedReady(h);
  h.element("schematic-image").naturalWidth = 2400;
  h.element("schematic-image").naturalHeight = 1600;
  let prevented = false;
  const key = (value, extra = {}) =>
    h.element("image-stage").fire("keydown", {
      key: value,
      preventDefault() {
        prevented = true;
      },
      ...extra,
    });
  key("+", { ctrlKey: true });
  assert.equal(prevented, false);
  assert.equal(h.ui.state.zoom, 1);
  key("+");
  assert.equal(prevented, true);
  assert.equal(h.element("zoom-level").textContent, "135%");
  key("0");
  assert.equal(h.element("zoom-fit").attributes["aria-pressed"], "true");
  key("1");
  assert.equal(h.element("zoom-level").textContent, "100%");
  assert.equal(h.element("zoom-actual").attributes["aria-pressed"], "true");
});
