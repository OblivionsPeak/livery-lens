# Third-party components vendored in `docs/`

Everything the web app needs is committed to this repo — no CDN, no
external requests at runtime. Photos never leave the browser.

## ONNX Runtime Web 1.22.0 (`vendor/ort.min.js`, `vendor/ort-wasm-simd-threaded.{wasm,mjs}`)

- Source: the official `onnxruntime-web@1.22.0` npm tarball
  (`npm pack onnxruntime-web@1.22.0`), files copied verbatim from
  `package/dist/`.
- License: MIT — Copyright (c) Microsoft Corporation.
  <https://github.com/microsoft/onnxruntime/blob/main/LICENSE>
- Only the single-threaded-capable SIMD WASM build is vendored; the app
  runs it with `numThreads = 1` (GitHub Pages does not send the COOP/COEP
  headers required for threaded WASM).

## U2-Net "u2netp" model (`vendor/u2netp.onnx`, 4.6 MB)

- Source: the official rembg model release
  <https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx>
  (rembg by Daniel Gatis, MIT license).
  SHA-256: `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`
- Model license: **Apache License 2.0** — U^2-Net by Xuebin Qin et al.,
  <https://github.com/xuebinqin/U-2-Net> (see its LICENSE file).
  Paper: Qin et al., "U^2-Net: Going Deeper with Nested U-Structure for
  Salient Object Detection", Pattern Recognition 2020.
- This is the *small* (u2netp) variant. The Python CLI uses the full-size
  u2net (~180 MB) via rembg, which segments slightly better; u2netp keeps
  the repo small and the first-load download reasonable. See the README
  for the quality caveat.
