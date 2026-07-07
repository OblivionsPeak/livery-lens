/* livery-lens web — car/background segmentation.
 *
 * Primary: u2netp (small U2-Net) via onnxruntime-web WASM, vendored in
 * ./vendor/. Preprocessing follows rembg's conventions: resize to 320x320,
 * divide by the image max, normalize with ImageNet mean/std, CHW float32.
 *
 * Fallback ("no-AI" toggle, or WASM failure): a center-weighted
 * saturation + edge-density heuristic — race-car photos usually frame a
 * colorful car near the center.
 *
 * All functions except loadModel/runU2netp are pure (no DOM) so the Node
 * test harness can exercise them.
 */

import { connectedComponents } from "./pipeline.js";

export const U2NET_SIZE = 320;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// --------------------------------------------------------- preprocessing

// rgba320: Uint8ClampedArray RGBA at 320x320 -> Float32Array [1,3,320,320]
export function buildInputTensor(rgba320) {
  const n = U2NET_SIZE * U2NET_SIZE;
  // rembg divides by the max pixel value of the image
  let mx = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    if (rgba320[o] > mx) mx = rgba320[o];
    if (rgba320[o + 1] > mx) mx = rgba320[o + 1];
    if (rgba320[o + 2] > mx) mx = rgba320[o + 2];
  }
  if (mx === 0) mx = 255;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    for (let c = 0; c < 3; c++) {
      data[c * n + i] = (rgba320[o + c] / mx - MEAN[c]) / STD[c];
    }
  }
  return data;
}

// -------------------------------------------------------- postprocessing

// bilinear resize of a single-channel float image
export function resizeBilinear(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  const xr = sw / dw, yr = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, (y + 0.5) * yr - 0.5);
    const y0 = Math.max(0, Math.floor(sy)), y1 = Math.min(sh - 1, y0 + 1);
    const fy = sy - y0;
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, (x + 0.5) * xr - 0.5);
      const x0 = Math.max(0, Math.floor(sx)), x1 = Math.min(sw - 1, x0 + 1);
      const fx = sx - x0;
      const a = src[y0 * sw + x0], b = src[y0 * sw + x1];
      const c = src[y1 * sw + x0], d = src[y1 * sw + x1];
      out[y * dw + x] = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx +
                        c * fy * (1 - fx) + d * fy * fx;
    }
  }
  return out;
}

// u2netp d0 output (Float32Array 320*320) -> binary mask at target size.
// rembg normalizes the prediction to 0..1 then thresholds alpha > 127/255.
export function maskFromOutput(pred, width, height) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < pred.length; i++) {
    if (pred[i] < mn) mn = pred[i];
    if (pred[i] > mx) mx = pred[i];
  }
  const range = mx - mn || 1;
  const norm = new Float32Array(pred.length);
  for (let i = 0; i < pred.length; i++) norm[i] = (pred[i] - mn) / range;
  const big = resizeBilinear(norm, U2NET_SIZE, U2NET_SIZE, width, height);
  const mask = new Uint8Array(width * height);
  for (let i = 0; i < mask.length; i++) mask[i] = big[i] > 0.5 ? 255 : 0;
  return mask;
}

// ------------------------------------------------------- mask refinement

// binary morphology with a disk structuring element
function morph(mask, width, height, radius, dilate) {
  const out = new Uint8Array(mask.length);
  const offs = [];
  for (let dy = -radius; dy <= radius; dy++)
    for (let dx = -radius; dx <= radius; dx++)
      if (dx * dx + dy * dy <= radius * radius) offs.push([dx, dy]);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let hit = dilate ? 0 : 255;
      for (const [dx, dy] of offs) {
        const nx = x + dx, ny = y + dy;
        const v = (nx < 0 || ny < 0 || nx >= width || ny >= height)
          ? 0 : mask[ny * width + nx];
        if (dilate) { if (v) { hit = 255; break; } }
        else if (!v) { hit = 0; break; }
      }
      out[y * width + x] = hit;
    }
  }
  return out;
}

// port of extract.py clean_mask(): close small holes, keep largest blob
export function cleanMask(mask, width, height) {
  let m = morph(mask, width, height, 4, true);   // close = dilate...
  m = morph(m, width, height, 4, false);         // ...then erode
  const bin = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) bin[i] = m[i] ? 1 : 0;
  const { labels, comps } = connectedComponents(bin, width, height);
  if (comps.length <= 1) return m;
  let biggest = 0;
  comps.forEach((c, i) => { if (c.area > comps[biggest].area) biggest = i; });
  const out = new Uint8Array(m.length);
  for (let i = 0; i < m.length; i++) out[i] = labels[i] === biggest ? 255 : 0;
  return out;
}

export function maskArea(mask) {
  let n = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) n++;
  return n;
}

// --------------------------------------------------- heuristic fallback

/* No-AI fallback: score every pixel by saturation and local edge density,
 * weighted by a center prior, then Otsu-threshold the score map. Crude,
 * but keeps the app functional when WASM is unavailable. */
export function heuristicMask(rgba, width, height) {
  const n = width * height;
  const sat = new Float32Array(n);
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const mx = Math.max(r, g, b);
    sat[i] = mx === 0 ? 0 : (mx - Math.min(r, g, b)) / mx;
    gray[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  // Sobel edge magnitude
  const edge = new Float32Array(n);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const gx = -gray[i - width - 1] - 2 * gray[i - 1] - gray[i + width - 1]
               + gray[i - width + 1] + 2 * gray[i + 1] + gray[i + width + 1];
      const gy = -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1]
               + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
      edge[i] = Math.min(1, Math.hypot(gx, gy) / 1020);
    }
  }
  // box-blur the edge map (radius ~ 2% of long edge) via summed-area table
  const r = Math.max(3, Math.round(Math.max(width, height) * 0.02));
  const sat2 = new Float64Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      sat2[(y + 1) * (width + 1) + x + 1] = edge[y * width + x]
        + sat2[y * (width + 1) + x + 1] + sat2[(y + 1) * (width + 1) + x]
        - sat2[y * (width + 1) + x];
  const density = (x, y) => {
    const x0 = Math.max(0, x - r), x1 = Math.min(width, x + r + 1);
    const y0 = Math.max(0, y - r), y1 = Math.min(height, y + r + 1);
    const s = sat2[y1 * (width + 1) + x1] - sat2[y0 * (width + 1) + x1]
            - sat2[y1 * (width + 1) + x0] + sat2[y0 * (width + 1) + x0];
    return s / ((x1 - x0) * (y1 - y0));
  };
  // score with gaussian center prior
  const cx = width / 2, cy = height / 2;
  const sx = width * 0.38, sy = height * 0.38;
  const score = new Float32Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const w = Math.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2));
      score[y * width + x] = w * (0.55 * sat[y * width + x] + 0.45 * Math.min(1, density(x, y) * 4));
    }
  }
  // Otsu threshold on a 256-bin histogram of the score
  const hist = new Int32Array(256);
  for (let i = 0; i < n; i++) hist[Math.min(255, Math.floor(score[i] * 255))]++;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, best = 0, bestVar = -1;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = n - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) * (mB - mF);
    if (v > bestVar) { bestVar = v; best = t; }
  }
  const thr = best / 255;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) mask[i] = score[i] > thr ? 255 : 0;
  return cleanMask(mask, width, height);
}

// --------------------------------------------------------------- runtime

/* Browser-side: create the ONNX session (ortApi = window.ort) and run
 * u2netp. Kept thin so tests can drive the pure pieces above directly. */
export async function runU2netp(ortApi, session, rgba320, width, height) {
  const input = new ortApi.Tensor("float32",
    buildInputTensor(rgba320), [1, 3, U2NET_SIZE, U2NET_SIZE]);
  const feeds = {};
  feeds[session.inputNames[0]] = input;
  const results = await session.run(feeds);
  const pred = results[session.outputNames[0]].data; // d0: [1,1,320,320]
  return maskFromOutput(pred, width, height);
}
