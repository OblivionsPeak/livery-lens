# livery-lens

Experimental spike: extract livery design elements — palette and color-zone
geometry — from a photograph of a real race car, as a **starting point** for
iRacing livery creation (Clearcoat / SimTex Pro toolchain).

Honest goal: a useful 30% head start, not magic. Explicitly out of scope:
sponsor logo extraction, vectorization, 3D unwrapping.

## Setup (Windows, Python 3.12)

```powershell
py -3.12 -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
```

`rembg` downloads its U2-Net model (~180 MB) to `~/.u2net` on first run.
Everything runs on CPU; no torch, no GPU required.

## Usage

```powershell
venv\Scripts\python extract.py path\to\photo.jpg -o outdir
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `-k / --colors` | 7 | max palette size before perceptual merging |
| `--segmenter` | `auto` | `auto` (rembg if installed, else grabcut), `rembg`, `grabcut`, `none` |

## Outputs

| File | Contents |
|---|---|
| `palette.json` | dominant livery colors: hex, coverage fraction of car pixels, saturation, CIELAB, heuristic role (`base` / `accent` / `detail`) |
| `zones.png` | posterized zone map — car pixels painted with their cluster color, background dimmed |
| `zones.json` | per-color connected regions: bounding boxes, centroids, area fractions (the stripe/block geometry) |
| `mask.png` | detected car mask (debug) |
| `report.html` | self-contained visual report: original, mask overlay, zone map, swatches |
| `clearcoat_starter.json` | palette in the proposed Clearcoat import schema (below) |

## How it works

1. **Car/background separation** — `rembg` (U2-Net via onnxruntime) by
   default. Fallback: OpenCV GrabCut seeded with an inset rectangle +
   definite-background corners. Side-by-side on the five examples, rembg
   wins decisively (see `FINDINGS.md`); GrabCut is only acceptable on
   museum shots with dark, uncluttered backgrounds.
2. **Palette** — masked pixels (minus near-black, which is usually tires/
   glass/shadow) are clustered with k-means in CIELAB; clusters closer than
   ΔE76 ≈ 14 are merged. Coverage = fraction of clustered car pixels.
3. **Roles** — heuristic: biggest cluster is `base`; ≥15% coverage or
   (saturation > 0.45 and ≥2.5%) is `accent`; the rest `detail`.
4. **Zones** — every car pixel is labeled with its nearest cluster,
   median-filtered, then connected components ≥0.5% of car area are
   reported with bbox/centroid/area.

## `clearcoat_starter.json` schema (proposal)

Clearcoat has no palette import today; this schema is a proposal matching
its conventions (lowercase `#rrggbb`, a single `baseColor` plus layer
colors). Schema id: `livery-lens/palette@1`.

```json
{
  "schema": "livery-lens/palette@1",
  "source": { "photo": "input.jpg", "generator": "livery-lens extract.py", "generated": "2026-07-06T12:00:00+00:00" },
  "baseColor": "#aab9bf",
  "colors": [
    { "hex": "#aab9bf", "role": "base",   "coverage": 0.433, "lab": [73.9, -3.2, -5.1] },
    { "hex": "#de6d3c", "role": "accent", "coverage": 0.036, "lab": [58.1, 40.2, 47.9] }
  ],
  "notes": "..."
}
```

- `baseColor` — direct candidate for Clearcoat's `doc.baseColor`.
- `colors[]` — sorted by coverage desc; `coverage` is the fraction of
  detected car-body pixels; `lab` is CIELAB (L 0–100, a/b signed).
- Consumers should treat `role` as a hint, not truth.

## Examples

`examples/` contains five Wikimedia Commons photos (attribution and
re-download URLs in `examples/SOURCES.md`; the images themselves are
gitignored) with committed pipeline outputs in each `out/` directory.
Open any `examples/<name>/out/report.html` in a browser.

Honest quality assessment: `FINDINGS.md`.
