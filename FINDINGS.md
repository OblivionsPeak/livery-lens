# Findings — livery-lens spike

Date: 2026-07-06. Five Wikimedia Commons photos, full pipeline
(`--segmenter auto` → rembg), eyeballed against the originals.

## Segmentation: rembg vs GrabCut

- **rembg (U2-Net, onnxruntime, CPU)** isolated the car cleanly in all
  five photos — museum, indoor expo, paddock, and an on-track panning
  shot. A few seconds per image on CPU after the one-time model download.
  This is the keeper.
- **GrabCut (inset-rect seed)** was acceptable only on the two museum
  shots with dark backgrounds. On the on-track Ferrari it included a huge
  swath of grass, which became the "base color" (69% olive green). Kept
  as a zero-download fallback, nothing more.

## Per-example assessment

### gulf-917 (Porsche 917, museum, dark background) — best case
Palette is right: Gulf light blue `#aab9bf` base (43%), white accent,
orange `#de6d3c` accent (3.6%). The zone map renders the orange nose
stripe and "Gulf" band crisply — genuinely usable as stripe-geometry
reference. Weaknesses: a warm `#a68f7f` cluster is bounce light from the
brick floor, not livery; the blue splits into lit/shaded variants.

### martini-935 (Porsche 935 Martini, museum, busy mural background)
Mask is clean despite the background. The Martini stripes survive: red
`#dc0a04` and blue `#238dc0` both recovered as accents, and the zone map
shows the hood/side stripe geometry recognizably. **Role failure:** the
"base" was assigned to `#2b2b2e` (windows, tires, shadow side) instead of
white — dark non-paint surfaces won the coverage contest.

### redbull-rb16 (show car, indoor expo, warm lighting)
Excellent mask in a crowded scene. Navy `#413c8e`, red `#ef6544`, and
yellow `#fce645`/`#e3aa27` all recovered; the zone map reads clearly as
the RB16 livery. **Lighting failure:** warm hall lighting created a big
brownish `#7d645e` cluster (warm-lit carbon/shadow) that took the base
role, and the navy is lighter than the real paint.

### ferrari-sf71h (on-track panning shot) — hardest case
GrabCut failed outright; rembg rescued it. Red `#d73c28` recovered, but
the shadow side splits into a dark maroon `#5b191a` cluster, motion blur
softens zone edges, and some green trackside color still leaks at the
mask boundary (a small `#889340` cluster). Palette useful, zones only
suggestive.

### corvette-c8r (paddock, people and structures behind)
People standing next to the car did not leak into the mask. Corvette
yellow recovered but split into lit `#ead349` / shaded `#e4b629`
variants; red door accent found. Zone map shows the black/gray body
graphics believably.

## Failure modes (recurring)

1. **Illumination splitting** — one paint color becomes 2–3 clusters
   (lit / shaded / reflected). The single biggest quality problem.
2. **Dark-surface base capture** — tires, glass, and shadow-side panels
   form a large dark cluster that can steal the `base` role (martini-935,
   corvette-c8r). The near-black cutoff helps but is not enough.
3. **Colored reflections / ambient cast** — floor bounce (gulf-917), warm
   hall lighting (redbull-rb16) tint the palette away from the true paint.
4. **Mask-boundary leakage** — thin slivers of background at the car edge
   (ferrari grass) create small phantom clusters.
5. **Zones are raster blobs, not shapes** — bounding boxes and centroids
   are honest, but nothing here is a vector stripe you can drop into
   Clearcoat.

## Verdict: does this merit further investment?

**Yes, modestly — for the palette; not yet for geometry.** The palette +
report is already a real 30% starting point: for all five cars the
correct livery colors are in the output and the report makes the good/bad
clusters obvious at a glance. rembg is the right segmentation call.

Highest-value next steps, in order:

1. **Illumination declumping** — re-merge clusters that share hue but
   differ mainly in L (cluster in hue/chroma, or normalize L before
   k-means). Fixes failure mode 1 and partially 3.
2. **Non-paint rejection** — score clusters for "probably tires/glass/
   carbon" (low chroma + low L) and exclude them from role assignment.
   Fixes the base-role errors.
3. **Clearcoat import** — a small "import palette" affordance in
   Clearcoat consuming `livery-lens/palette@1` would close the loop and
   make this immediately usable in the real workflow.
4. Only after those: smarter zone geometry (contour simplification to
   polygons). Vectorizing stripes from a single perspective photo is a
   much bigger project and should stay out of scope until 1–3 prove out.
