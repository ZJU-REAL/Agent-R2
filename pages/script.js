const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------ Section spy ------------------------------ */
(() => {
  const links = [...document.querySelectorAll('.ar-nav a[href^="#"]')];
  const spy = new IntersectionObserver(
    (entries) =>
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        links.forEach((a) => a.classList.toggle("ar-on", a.getAttribute("href") === `#${entry.target.id}`));
      }),
    { rootMargin: "-40% 0px -55% 0px" },
  );
  links.forEach((a) => {
    const el = document.querySelector(a.getAttribute("href"));
    if (el) spy.observe(el);
  });
})();

/* ------------------------------ Segmented helper ------------------------------ */
function segmented(group, onChange) {
  const buttons = [...group.querySelectorAll("button")];
  buttons.forEach((button) =>
    button.addEventListener("click", () => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      onChange(button.dataset.v);
    }),
  );
}

/* ------------------------------ GDPval board (Table 1) ------------------------------ */
(() => {
  const board = document.querySelector("[data-ar-board]");
  if (!board) return;
  // README / paper Table 1: [Office P@1, Office P@3, Non-Office P@1, Non-Office P@3, Full P@1, Full P@3]
  const groups = [
    [
      "Closed-source",
      [
        ["GPT-4.1", [38.0, 51.5, 44.9, 55.5, 39.4, 52.3]],
        ["Gemini-2.5-Pro", [57.0, 59.4, 61.4, 63.3, 57.9, 60.2]],
        ["Claude-Opus-4.5", [73.8, 77.4, 68.2, 72.3, 72.6, 76.3]],
      ],
    ],
    [
      "Open-source",
      [
        ["DeepSeek-v3.1", [48.3, 57.5, 50.9, 61.7, 48.8, 58.4]],
        ["GLM-4.7", [60.8, 67.1, 61.9, 68.3, 61.1, 67.4]],
        ["Qwen3-30B-A3B-Thinking-2507", [33.7, 38.5, 40.4, 44.9, 35.0, 39.8]],
        ["Qwen3-235B-A22B-Thinking-2507", [46.9, 52.9, 52.0, 58.6, 48.0, 54.1], "ar-rival"],
        ["Qwen3-VL-235B-A22B-Thinking", [48.4, 50.9, 54.7, 57.9, 49.7, 52.4]],
      ],
    ],
    [
      "Agent-R² (ours)",
      [
        ["Qwen3-4B-Thinking-2507 (base)", [31.1, 35.7, 37.8, 42.3, 32.4, 37.0]],
        ["Agent-R²-4B-SFT", [45.9, 51.5, 42.8, 47.6, 45.3, 50.7]],
        ["Agent-R²-4B", [51.5, 54.7, 48.9, 55.1, 51.0, 54.8], "ar-ours"],
      ],
    ],
  ];
  const rows = board.querySelector("[data-ar-rows]");
  const note = board.querySelector("[data-ar-note]");
  const subsets = { office: [0, "Office"], nonoffice: [2, "Non-Office"], full: [4, "Full"] };
  const state = { subset: "full", metric: "1" };
  const all = groups.flatMap(([, list]) => list);
  const find = (name) => all.find((r) => r[0] === name)[1];

  // build once, then only update widths and numbers so bars animate
  const cells = [];
  rows.innerHTML = groups
    .map(
      ([label, list]) =>
        `<div class="ar-grp" role="row"><span role="rowheader">${label}</span></div>` +
        list
          .map(
            ([name, , cls]) =>
              `<div class="ar-row ${cls || ""}" role="row"><span class="ar-row-name" role="rowheader">${name}</span>` +
              `<span class="ar-row-bar" aria-hidden="true"><span></span></span><span class="ar-row-val" role="cell"></span></div>`,
          )
          .join(""),
    )
    .join("");
  rows.querySelectorAll(".ar-row").forEach((el, k) => cells.push([el, all[k][1]]));

  function render() {
    const col = subsets[state.subset][0] + (state.metric === "3" ? 1 : 0);
    cells.forEach(([el, vals]) => {
      el.querySelector(".ar-row-bar span").style.setProperty("--v", vals[col] / 80);
      el.querySelector(".ar-row-val").textContent = vals[col].toFixed(1);
    });
    const ours = find("Agent-R²-4B")[col];
    const rival = find("Qwen3-235B-A22B-Thinking-2507")[col];
    const base = find("Qwen3-4B-Thinking-2507 (base)")[col];
    const d = ours - rival;
    const label = `${subsets[state.subset][1]} · Pass@${state.metric}`;
    note.innerHTML =
      `${label}: Agent-R²-4B <b>${ours.toFixed(1)}</b> vs Qwen3-235B-A22B-Thinking-2507 <b>${rival.toFixed(1)}</b> ` +
      (d >= 0 ? `(ahead by ${d.toFixed(1)})` : `(behind by ${(-d).toFixed(1)})`) +
      `, and ${(ours - base).toFixed(1)} points above its 4B base.` +
      (state.subset === "nonoffice" ? " Non-office deliverables (audio, video, other media) are outside the paper's primary focus." : "");
  }
  segmented(board.querySelector("[data-ar-subset]"), (v) => {
    state.subset = v;
    render();
  });
  segmented(board.querySelector("[data-ar-metric]"), (v) => {
    state.metric = v;
    render();
  });
  render();

  // static full table
  const body = board.querySelector("[data-ar-full]");
  body.innerHTML = groups
    .map(
      ([label, list]) =>
        `<tr class="ar-sub"><td colspan="7">${label}</td></tr>` +
        list.map(([name, vals, cls]) => `<tr class="${cls || ""}"><td>${name}</td>${vals.map((v) => `<td>${v.toFixed(1)}</td>`).join("")}</tr>`).join(""),
    )
    .join("");
})();

/* ------------------------------ Copy BibTeX ------------------------------ */
function copyFallback(text) {
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.cssText = "position:fixed;top:0;left:0;opacity:0";
  document.body.append(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}
async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    try {
      await Promise.race([
        navigator.clipboard.writeText(text),
        new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), 800)),
      ]);
      return true;
    } catch {
      /* fall through */
    }
  }
  return copyFallback(text);
}
document.querySelectorAll("[data-ar-copy]").forEach((button) =>
  button.addEventListener("click", async () => {
    const text = document.getElementById(button.dataset.arCopy).textContent;
    button.textContent = (await copyText(text)) ? "Copied ✓" : "Select and copy";
    button.classList.add("ar-done");
    setTimeout(() => {
      button.textContent = "Copy BibTeX";
      button.classList.remove("ar-done");
    }, 1800);
  }),
);

/* ------------------------------ Animated method figure ------------------------------ */
(() => {
  const fig = document.querySelector("[data-ar-fig]");
  if (!fig) return;
  const stage = fig.querySelector("[data-ar-stage]");
  const svg = fig.querySelector("[data-ar-wires]");
  const phases = ["workflow", "files", "task", "refine", "train"];
  const panels = phases.map((p) => fig.querySelector(`[data-ar-panel="${p}"]`));
  const tabs = [...fig.querySelectorAll("[data-ar-phase]")];
  const playButton = fig.querySelector("[data-ar-play]");
  const anchor = (key) => fig.querySelector(`[data-ar-a="${key}"]`);
  const NS = "http://www.w3.org/2000/svg";

  fig.querySelectorAll(".ar-stg").forEach((list) => [...list.children].forEach((child, i) => child.style.setProperty("--i", i)));

  /* ---- the single transient connector: one per phase, from the previous panel into this one ---- */
  const shown = (el) => el && el.getClientRects().length > 0;
  function box(el) {
    const r = el.getBoundingClientRect();
    const s = stage.getBoundingClientRect();
    const l = r.left - s.left;
    const t = r.top - s.top;
    return { l, t, r: l + r.width, b: t + r.height, cx: l + r.width / 2, cy: t + r.height / 2 };
  }
  function route(phase) {
    const side = (fromKey, toKey, a, b) => {
      const f = box(anchor(fromKey));
      const t = box(anchor(toKey));
      const x = (box(panels[a]).r + box(panels[b]).l) / 2;
      const y0 = Math.min(f.cy, box(panels[a]).b - 14);
      return [[f.r, y0], [x, y0], [x, t.cy], [t.l - 3, t.cy]];
    };
    if (!panels.every(shown)) return null;
    if (phase === "files") return side("blueprint", "query", 0, 1);
    if (phase === "task") return side("fc", "sandbox", 1, 2);
    if (phase === "refine") {
      const f = box(anchor("rubric"));
      const t = box(anchor("expert"));
      const y = (box(panels[2]).b + box(panels[3]).t) / 2;
      const x0 = f.l + 28;
      return [[x0, f.b], [x0, y], [t.cx, y], [t.cx, t.t - 3]];
    }
    if (phase === "train") {
      const f = box(anchor("final"));
      const t = box(anchor("sft"));
      const x = (box(panels[3]).r + box(panels[4]).l) / 2;
      return [[f.r, f.cy], [x, f.cy], [x, t.cy], [t.l - 3, t.cy]];
    }
    return null;
  }
  function pathData(pts, radius = 9) {
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[i + 1];
      const l1 = Math.hypot(x1 - x0, y1 - y0) || 1;
      const l2 = Math.hypot(x2 - x1, y2 - y1) || 1;
      const k = Math.min(radius, l1 / 2, l2 / 2);
      d += ` L${x1 + ((x0 - x1) / l1) * k},${y1 + ((y0 - y1) / l1) * k}`;
      d += ` Q${x1},${y1} ${x1 + ((x2 - x1) / l2) * k},${y1 + ((y2 - y1) / l2) * k}`;
    }
    const [lx, ly] = pts[pts.length - 1];
    return `${d} L${lx},${ly}`;
  }
  const make = (tag, attrs) => {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
  };

  let wire = null; // { phase, g, flow, dot, head, len, progress, state }
  function buildWire(phase, progress = 0, state = "") {
    svg.replaceChildren();
    wire = null;
    const pts = route(phase);
    if (!pts) return;
    const g = make("g", { class: state });
    const flowPath = make("path", { class: "ar-w-flow", d: pathData(pts) });
    const [ex, ey] = pts[pts.length - 1];
    const [px, py] = pts[pts.length - 2];
    const ang = (Math.atan2(ey - py, ex - px) * 180) / Math.PI;
    const head = make("path", { class: "ar-w-head", d: "M-7,-4.2 L1,0 L-7,4.2 z", transform: `translate(${ex},${ey}) rotate(${ang})` });
    const dot = make("circle", { class: "ar-w-dot", r: 4, cx: pts[0][0], cy: pts[0][1] });
    g.append(flowPath, head, dot);
    svg.append(g);
    const len = flowPath.getTotalLength();
    flowPath.style.strokeDasharray = `${len} ${len}`;
    flowPath.style.strokeDashoffset = len * (1 - progress);
    wire = { phase, g, flow: flowPath, dot, len, progress, state };
  }
  function clearWire() {
    svg.replaceChildren();
    wire = null;
  }
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  function flow(ms) {
    const run = token;
    buildWire(phases[index]);
    const start = performance.now();
    return new Promise((resolve) => {
      function frame(now) {
        if (run !== token || !wire) return resolve();
        const t = Math.min(1, (now - start) / ms);
        wire.progress = ease(t);
        const at = wire.len * wire.progress;
        const p = wire.flow.getPointAtLength(at);
        wire.flow.style.strokeDashoffset = wire.len - at;
        wire.dot.setAttribute("cx", p.x);
        wire.dot.setAttribute("cy", p.y);
        wire.g.classList.add("ar-live");
        wire.state = "ar-live";
        if (t < 1) return requestAnimationFrame(frame);
        wire.g.classList.remove("ar-live");
        wire.g.classList.add("ar-done");
        wire.state = "ar-done";
        resolve();
      }
      requestAnimationFrame(frame);
    });
  }
  function fadeWire() {
    if (!wire) return;
    wire.g.classList.add("ar-fade");
    wire.state += " ar-fade";
  }

  /* ---- steps ---- */
  const lit = (n) => panels[index].querySelectorAll(`[data-step="${n}"]`).forEach((el) => el.classList.add("ar-lit"));
  function bump(el) {
    if (!el) return;
    el.classList.remove("ar-bump");
    void el.offsetWidth;
    el.classList.add("ar-bump");
  }
  const noop = () => {};
  const sequences = {
    workflow: [
      [700, () => lit(1)],
      [1300, () => lit(2)],
      [1100, () => lit(3)],
      [1500, () => lit(4)],
      [1700, () => bump(anchor("blueprint"))],
    ],
    files: [
      [1000, (ms) => flow(ms)],
      [900, () => lit(1)],
      [1400, () => (lit(2), fadeWire())],
      [1200, () => lit(3)],
      [1500, () => lit(4)],
      [1300, () => lit(5)],
      [1500, noop],
    ],
    task: [
      [1000, (ms) => flow(ms)],
      [1000, () => lit(1)],
      [1400, () => (lit(2), fadeWire())],
      [1400, () => lit(3)],
      [1500, () => lit(4)],
      [1600, noop],
    ],
    refine: [
      [1100, (ms) => flow(ms)],
      [2000, () => lit(1)],
      [1000, () => (lit(2), fadeWire())],
      [1500, () => lit(3)],
      [1200, () => lit(4)],
      [1700, () => bump(anchor("final"))],
    ],
    train: [
      [1000, (ms) => flow(ms)],
      [1400, () => lit(1)],
      [1600, () => (lit(2), fadeWire())],
      [1300, () => lit(3)],
      [1800, () => lit(4)],
      [1400, () => lit(5)],
      [1900, noop],
    ],
  };

  /* ---- player ---- */
  let token = 0;
  let index = 0;
  let playing = !reduceMotion;
  let inView = false;
  let running = false;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  function markTabs() {
    tabs.forEach((tab, k) => tab.setAttribute("aria-pressed", String(k === index)));
    panels.forEach((p, k) => p.classList.toggle("ar-on", k === index));
  }
  // playing layout: earlier panels complete, current one empty and building, later ones blank
  function prepare(i) {
    index = i;
    fig.classList.add("ar-playing");
    panels.forEach((p, k) => {
      p.classList.toggle("ar-run", k === i);
      p.classList.toggle("ar-future", k > i);
      p.querySelectorAll(".ar-lit").forEach((el) => el.classList.remove("ar-lit"));
    });
    clearWire();
    markTabs();
  }
  // static layout: the complete figure, current phase marked
  function settleAll() {
    fig.classList.remove("ar-playing");
    panels.forEach((p) => p.classList.remove("ar-run", "ar-future"));
    clearWire();
    tabs.forEach((tab) => tab.style.setProperty("--t", 0));
    markTabs();
  }
  function stop() {
    token++;
    running = false;
    settleAll();
  }
  async function run(i) {
    const my = ++token;
    running = true;
    tabs.forEach((tab) => tab.style.setProperty("--t", 0));
    prepare(i);
    const steps = sequences[phases[i]];
    const total = steps.reduce((s, [ms]) => s + ms, 0);
    const start = performance.now();
    (function progress(now) {
      if (my !== token) return;
      tabs[i].style.setProperty("--t", Math.min(1, (now - start) / total));
      requestAnimationFrame(progress);
    })(start);
    for (const [ms, action] of steps) {
      action(ms);
      await wait(ms);
      if (my !== token) return;
    }
    if (playing && inView) run((i + 1) % phases.length);
    else stop();
  }
  function select(i) {
    if (playing && inView) run(i);
    else {
      stop();
      index = i;
      markTabs();
    }
  }
  function setPlaying(next) {
    playing = next;
    fig.toggleAttribute("data-paused", !next);
    playButton.setAttribute("aria-label", next ? "Pause animation" : "Play animation");
    if (next && inView) run(index);
    else if (!next) stop();
  }

  tabs.forEach((tab, i) => tab.addEventListener("click", () => select(i)));
  panels.forEach((p, i) =>
    p.addEventListener("click", () => {
      if (index !== i) select(i);
    }),
  );
  playButton.addEventListener("click", () => setPlaying(!playing));

  new IntersectionObserver(
    ([entry]) => {
      inView = entry.isIntersecting;
      if (inView && playing && !running) run(index);
      if (!inView && running) {
        // leave it ready to start again from this phase, with nothing shown ahead of time
        token++;
        running = false;
        prepare(index);
      }
    },
    { threshold: 0.25 },
  ).observe(stage);

  let queued = false;
  new ResizeObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      if (wire) buildWire(wire.phase, wire.progress, wire.state);
    });
  }).observe(stage);

  fig.toggleAttribute("data-paused", !playing);
  if (playing) prepare(0);
  else settleAll();
})();
