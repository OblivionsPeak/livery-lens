/* livery-lens web — UI wiring. All processing is local to the browser. */

import {
  MAX_DIM, extractPalette, assignRoles, medianFilterLabels, zoneGeometry,
  paletteJson, clearcoatStarter,
} from "./pipeline.js";
import {
  U2NET_SIZE, runU2netp, cleanMask, maskArea, heuristicMask,
} from "./segment.js";

const $ = (id) => document.getElementById(id);
const status = $("status"), stageText = $("stageText");
const bar = $("bar"), barFill = $("barFill"), warn = $("warn");
const results = $("results");

let session = null;          // ONNX session, created once
let lastOutputs = null;      // {palette, starter, zonesCanvas, srcName}
let busy = false;

// ------------------------------------------------------------- helpers

function setStage(text) {
  status.classList.add("active");
  stageText.textContent = text;
}
function setProgress(frac) {
  bar.classList.toggle("active", frac != null);
  if (frac != null) barFill.style.width = `${Math.round(frac * 100)}%`;
}
function setWarn(text) {
  warn.classList.toggle("active", !!text);
  warn.textContent = text || "";
}
function doneStatus() {
  status.classList.remove("active");
  setProgress(null);
}
const yieldUI = () => new Promise((r) => setTimeout(r, 0));

function downloadBlob(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------------------------------------------------------- model load

async function ensureSession() {
  if (session) return session;
  if (typeof ort === "undefined") throw new Error("onnxruntime failed to load");
  // Must be an absolute URL: ort dynamic-imports the .mjs from this path, and
  // bare relative specifiers (no "./" or scheme) are illegal in import().
  ort.env.wasm.wasmPaths = new URL("vendor/", document.baseURI).href;
  ort.env.wasm.numThreads = 1; // GitHub Pages has no COOP/COEP headers
  setStage("Downloading segmentation model (one-time, ~16 MB)…");
  const resp = await fetch("vendor/u2netp.onnx");
  if (!resp.ok) throw new Error(`model fetch failed: ${resp.status}`);
  const total = +resp.headers.get("Content-Length") || 0;
  const chunks = [];
  let got = 0;
  const reader = resp.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) setProgress(got / total);
  }
  setProgress(null);
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  setStage("Initializing WASM runtime…");
  await yieldUI();
  session = await ort.InferenceSession.create(buf.buffer, {
    executionProviders: ["wasm"],
  });
  return session;
}

// --------------------------------------------------------- main driver

async function analyze(file) {
  if (busy) return;
  busy = true;
  setWarn(null);
  results.classList.remove("active");
  const t0 = performance.now();
  try {
    setStage("Loading image…");
    const bmp = await createImageBitmap(file);
    const scale = Math.min(1, MAX_DIM / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * scale), h = Math.round(bmp.height * scale);
    const work = new OffscreenCanvas(w, h);
    const wctx = work.getContext("2d");
    wctx.drawImage(bmp, 0, 0, w, h);
    const rgba = wctx.getImageData(0, 0, w, h).data;

    // ---- segmentation
    let mask = null, method = "none";
    const useAI = $("useAI").checked;
    if (useAI) {
      try {
        const sess = await ensureSession();
        setStage("Separating car from background (U2-Net)…");
        await yieldUI();
        const small = new OffscreenCanvas(U2NET_SIZE, U2NET_SIZE);
        const sctx = small.getContext("2d");
        sctx.drawImage(bmp, 0, 0, U2NET_SIZE, U2NET_SIZE);
        const rgba320 = sctx.getImageData(0, 0, U2NET_SIZE, U2NET_SIZE).data;
        const raw = await runU2netp(ort, sess, rgba320, w, h);
        const cleaned = cleanMask(raw, w, h);
        if (maskArea(cleaned) > 0.03 * w * h) { mask = cleaned; method = "u2netp"; }
        else setWarn("U2-Net mask came back nearly empty — using the heuristic fallback.");
      } catch (e) {
        console.error(e);
        setWarn(`AI segmentation unavailable (${e.message}) — using the heuristic fallback.`);
      }
    }
    if (!mask) {
      setStage("Estimating car region (heuristic)…");
      await yieldUI();
      const m = heuristicMask(rgba, w, h);
      if (maskArea(m) > 0.03 * w * h) { mask = m; method = useAI ? "heuristic-fallback" : "heuristic"; }
      else { mask = null; method = "none"; } // full frame
    }

    // ---- palette
    setStage("Clustering colors (k-means in CIELAB)…");
    await yieldUI();
    const { clusters, labelMap, liveryCount } = extractPalette(rgba, w, h, mask, +$("kColors").value);
    assignRoles(clusters);

    // ---- zones
    setStage("Building zone map…");
    await yieldUI();
    const cleanLabels = medianFilterLabels(labelMap, w, h);
    const carArea = mask ? maskArea(mask) : w * h;
    const zones = zoneGeometry(cleanLabels, w, h, clusters, Math.max(carArea, 1));

    // ---- render
    renderResults({ rgba, w, h, mask, method, clusters, cleanLabels, zones, liveryCount, file });
    $("meta").textContent =
      `Segmentation: ${method} · car ≈ ${Math.round(carArea / (w * h) * 100)}% of frame · ` +
      `working resolution ${w}×${h} · ${((performance.now() - t0) / 1000).toFixed(1)}s · ` +
      `processed entirely in your browser`;
    doneStatus();
    results.classList.add("active");
  } catch (e) {
    console.error(e);
    doneStatus();
    setWarn(`Something went wrong: ${e.message}`);
    status.classList.add("active");
    stageText.textContent = "Failed.";
  } finally {
    busy = false;
  }
}

// ------------------------------------------------------------ rendering

function paintCanvas(id, w, h, fill) {
  const cv = $(id);
  cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(w, h);
  fill(img.data);
  ctx.putImageData(img, 0, 0);
  return cv;
}

function renderResults({ rgba, w, h, mask, method, clusters, cleanLabels, zones, file }) {
  // original
  paintCanvas("cvOriginal", w, h, (d) => d.set(rgba));

  // mask overlay: dim background, green boundary
  paintCanvas("cvMask", w, h, (d) => {
    for (let i = 0; i < w * h; i++) {
      const inside = !mask || mask[i];
      const f = inside ? 1 : 0.25;
      d[i * 4] = rgba[i * 4] * f;
      d[i * 4 + 1] = rgba[i * 4 + 1] * f;
      d[i * 4 + 2] = rgba[i * 4 + 2] * f;
      d[i * 4 + 3] = 255;
    }
    if (mask) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const i = y * w + x;
          if (!mask[i]) continue;
          let edge = false;
          for (let dy = -1; dy <= 1 && !edge; dy++)
            for (let dx = -1; dx <= 1 && !edge; dx++) {
              const nx = x + dx, ny = y + dy;
              if (nx < 0 || ny < 0 || nx >= w || ny >= h || !mask[ny * w + nx]) edge = true;
            }
          if (edge) { d[i * 4] = 0; d[i * 4 + 1] = 255; d[i * 4 + 2] = 90; }
        }
      }
    }
  });

  // zone map: cluster color inside livery, dim original outside
  const zonesCv = paintCanvas("cvZones", w, h, (d) => {
    for (let i = 0; i < w * h; i++) {
      d[i * 4] = rgba[i * 4] * 0.25;
      d[i * 4 + 1] = rgba[i * 4 + 1] * 0.25;
      d[i * 4 + 2] = rgba[i * 4 + 2] * 0.25;
      d[i * 4 + 3] = 255;
    }
    for (const c of clusters) {
      for (let i = 0; i < w * h; i++) {
        if (cleanLabels[i] === c.gi) {
          d[i * 4] = c.rgb[0]; d[i * 4 + 1] = c.rgb[1]; d[i * 4 + 2] = c.rgb[2];
        }
      }
    }
  });

  // palette swatches (click to copy hex)
  const sw = $("swatches");
  sw.innerHTML = "";
  for (const c of clusters) {
    const row = document.createElement("div");
    row.className = "sw";
    row.title = "Click to copy hex";
    row.innerHTML =
      `<div class="chip" style="background:${c.hex}"></div>` +
      `<code>${c.hex}</code><span class="role ${c.role}">${c.role}</span>` +
      `<span class="cov">${(c.coverage * 100).toFixed(1)}%</span>`;
    row.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(c.hex); } catch { /* ignore */ }
      const tag = document.createElement("span");
      tag.className = "copied";
      tag.textContent = "copied";
      row.appendChild(tag);
      setTimeout(() => tag.remove(), 900);
    });
    sw.appendChild(row);
  }

  // zone stats
  const zs = $("zoneStats");
  zs.innerHTML = "";
  for (const z of zones) {
    const n = z.regions.length;
    const biggest = n ? ` · largest ${(z.regions[0].area_frac_of_car * 100).toFixed(1)}% of car` : "";
    const row = document.createElement("div");
    row.className = "zonestat";
    row.innerHTML =
      `<span class="dot" style="background:${z.hex}"></span>` +
      `<span>${z.hex} — ${n} region${n === 1 ? "" : "s"}${biggest}</span>`;
    zs.appendChild(row);
  }

  const srcName = file.name || "photo";
  lastOutputs = {
    srcName,
    palette: paletteJson(srcName, method, clusters),
    starter: clearcoatStarter(srcName, clusters),
    zonesCv,
  };
}

// ------------------------------------------------------------ downloads

$("dlPalette").addEventListener("click", () => {
  if (!lastOutputs) return;
  downloadBlob("palette.json", new Blob(
    [JSON.stringify(lastOutputs.palette, null, 2)], { type: "application/json" }));
});
$("dlClearcoat").addEventListener("click", () => {
  if (!lastOutputs) return;
  downloadBlob("clearcoat_starter.json", new Blob(
    [JSON.stringify(lastOutputs.starter, null, 2)], { type: "application/json" }));
});
$("dlZones").addEventListener("click", () => {
  if (!lastOutputs) return;
  lastOutputs.zonesCv.toBlob((b) => downloadBlob("zones.png", b), "image/png");
});

// ------------------------------------------------------------ dropzone

const drop = $("drop"), fileInput = $("file");
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) analyze(fileInput.files[0]);
});
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("hover"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("hover"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) analyze(f);
});
$("kColors").addEventListener("input", () => { $("kVal").textContent = $("kColors").value; });
