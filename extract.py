#!/usr/bin/env python3
"""livery-lens: extract livery design elements from a photo of a real race car.

Produces, in an output directory:
  palette.json           dominant livery colors (LAB k-means, car-weighted)
  zones.png / zones.json posterized color-region segmentation + bounding info
  mask.png               the detected car mask (debug / report)
  report.html            self-contained visual report
  clearcoat_starter.json palette in the proposed Clearcoat import schema

This is a spike: the goal is a useful starting point for iRacing livery
creation, not pixel-perfect extraction. Sponsor logos are out of scope.
"""

import argparse
import base64
import datetime
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

MAX_DIM = 1200          # working resolution (long edge)
SAMPLE_PIXELS = 60_000  # max pixels fed to k-means
MERGE_DELTA_E = 14.0    # CIE76 distance below which clusters are merged
DARK_V_CUTOFF = 32      # HSV value below which pixels are treated as
                        # tires/glass/shadow, not livery paint
MIN_ZONE_FRAC = 0.005   # connected components smaller than this fraction of
                        # the car area are ignored in zones.json


# ---------------------------------------------------------------- image io

def imread_unicode(path: Path) -> np.ndarray:
    """cv2.imread fails on some Windows unicode paths; go through a buffer."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not decode image: {path}")
    return img


def resize_max(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def jpg_data_uri(img: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def png_data_uri(img: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode()


# ------------------------------------------------------------- segmentation

def segment_rembg(img_bgr: np.ndarray) -> np.ndarray | None:
    """Car mask via rembg (U2-Net, onnxruntime). Returns uint8 {0,255} or None."""
    try:
        from rembg import remove  # noqa: deferred import — optional dep
        from PIL import Image
    except ImportError:
        return None
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    out = remove(pil)  # RGBA
    alpha = np.array(out)[:, :, 3]
    mask = (alpha > 127).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None
    return mask


def segment_grabcut(img_bgr: np.ndarray) -> np.ndarray:
    """Car mask via OpenCV GrabCut seeded with an inset rectangle.

    Race-car photos usually frame the car near the center, so an inset rect
    is a workable prior. Corners are marked definite-background.
    """
    h, w = img_bgr.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mx, my = round(w * 0.06), round(h * 0.10)
    mask[my:h - my, mx:w - mx] = cv2.GC_PR_FGD
    # corners: almost always background (sky/crowd/track wall)
    cw, ch = round(w * 0.12), round(h * 0.12)
    for ys, xs in ((slice(0, ch), slice(0, cw)),
                   (slice(0, ch), slice(w - cw, w)),
                   (slice(h - ch, h), slice(0, cw)),
                   (slice(h - ch, h), slice(w - cw, w))):
        mask[ys, xs] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    small = resize_max(img_bgr, 700)  # GrabCut is O(pixels); run reduced
    sm = cv2.resize(mask, (small.shape[1], small.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    cv2.grabCut(small, sm, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    out = np.where((sm == cv2.GC_FGD) | (sm == cv2.GC_PR_FGD), 255, 0)
    out = cv2.resize(out.astype(np.uint8), (w, h),
                     interpolation=cv2.INTER_NEAREST)
    return out


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest blob, close small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def build_mask(img_bgr: np.ndarray, segmenter: str) -> tuple[np.ndarray, str]:
    """Returns (mask uint8 {0,255}, name of method actually used)."""
    h, w = img_bgr.shape[:2]
    if segmenter == "none":
        return np.full((h, w), 255, np.uint8), "none"
    if segmenter in ("rembg", "auto"):
        m = segment_rembg(img_bgr)
        if m is not None:
            m = clean_mask(m)
            if m.sum() / 255 > 0.03 * h * w:  # sanity: mask isn't a sliver
                return m, "rembg"
        if segmenter == "rembg":
            print("warning: rembg unavailable or produced an empty mask; "
                  "falling back to grabcut", file=sys.stderr)
    m = clean_mask(segment_grabcut(img_bgr))
    if m.sum() / 255 < 0.03 * h * w:
        print("warning: grabcut mask too small; using full frame",
              file=sys.stderr)
        return np.full((h, w), 255, np.uint8), "none"
    return m, "grabcut"


# ------------------------------------------------------------------ palette

def cv_lab_to_cie(lab: np.ndarray) -> np.ndarray:
    """OpenCV 8-bit LAB -> real CIELAB floats."""
    lab = lab.astype(np.float64)
    return np.stack([lab[..., 0] * 100.0 / 255.0,
                     lab[..., 1] - 128.0,
                     lab[..., 2] - 128.0], axis=-1)


def bgr_of_lab_center(center_cie: np.ndarray) -> tuple[int, int, int]:
    l, a, b = center_cie
    px = np.array([[[l * 255.0 / 100.0, a + 128.0, b + 128.0]]], np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def hex_of_bgr(bgr) -> str:
    return "#{:02x}{:02x}{:02x}".format(bgr[2], bgr[1], bgr[0])


def saturation_of_bgr(bgr) -> float:
    px = np.array([[list(bgr)]], np.uint8)
    return float(cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0, 1]) / 255.0


def extract_palette(img_bgr, mask, n_colors):
    """K-means in CIELAB over masked, non-near-black pixels.

    Returns (clusters, label_map, livery_mask) where clusters is a list of
    dicts sorted by coverage desc, label_map is int32 (-1 outside livery),
    and livery_mask marks pixels that went into the clustering domain.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    livery_mask = (mask > 0) & (hsv[..., 2] >= DARK_V_CUTOFF)
    if livery_mask.sum() < 500:
        livery_mask = mask > 0  # dark car — keep everything

    lab_img = cv_lab_to_cie(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
    pts = lab_img[livery_mask]
    rng = np.random.default_rng(42)
    sample = pts if len(pts) <= SAMPLE_PIXELS else \
        pts[rng.choice(len(pts), SAMPLE_PIXELS, replace=False)]

    k = min(n_colors, max(2, len(sample) // 200))
    km = KMeans(n_clusters=k, n_init=4, random_state=42).fit(sample)
    centers = km.cluster_centers_

    # merge perceptually-close clusters (CIE76)
    order = list(range(k))
    merged: list[list[int]] = []
    for i in order:
        for group in merged:
            if np.linalg.norm(centers[i] - centers[group[0]]) < MERGE_DELTA_E:
                group.append(i)
                break
        else:
            merged.append([i])

    # assign every livery pixel to its nearest ORIGINAL center, then remap
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
    px_orig = np.argmin(d, axis=1)
    remap = np.empty(k, np.int32)
    for gi, group in enumerate(merged):
        for orig in group:
            remap[orig] = gi
    px_lbl = remap[px_orig]

    clusters = []
    total = len(px_lbl)
    for gi, group in enumerate(merged):
        sel = px_lbl == gi
        cov = sel.sum() / total
        center = pts[sel].mean(axis=0) if sel.any() else centers[group[0]]
        bgr = bgr_of_lab_center(center)
        clusters.append({
            "gi": gi,
            "lab": [round(float(v), 2) for v in center],
            "bgr": bgr,
            "hex": hex_of_bgr(bgr),
            "coverage": float(cov),
            "saturation": round(saturation_of_bgr(bgr), 3),
        })
    clusters.sort(key=lambda c: -c["coverage"])

    # full-resolution label map
    label_map = np.full(img_bgr.shape[:2], -1, np.int32)
    flat = lab_img[livery_mask]
    d_all = np.linalg.norm(flat[:, None, :] - centers[None, :, :], axis=2)
    label_map[livery_mask] = remap[np.argmin(d_all, axis=1)]
    return clusters, label_map, livery_mask


def assign_roles(clusters):
    """Heuristic role guess: base = dominant coverage; accent = punchy,
    meaningful coverage; detail = the rest."""
    for i, c in enumerate(clusters):
        if i == 0:
            c["role"] = "base"
        elif c["coverage"] >= 0.15 or (c["saturation"] > 0.45
                                       and c["coverage"] >= 0.025):
            c["role"] = "accent"
        else:
            c["role"] = "detail"
    return clusters


# -------------------------------------------------------------------- zones

def render_zones(img_bgr, label_map, clusters):
    """Posterized zone image: cluster color inside livery, dim original out."""
    out = (img_bgr * 0.25).astype(np.uint8)
    # light median filter on labels to kill speckle
    lm = cv2.medianBlur((label_map + 1).astype(np.uint8), 7).astype(np.int32) - 1
    for c in clusters:
        out[lm == c["gi"]] = c["bgr"]
    return out, lm


def zone_geometry(label_map, clusters, car_area):
    zones = []
    for c in clusters:
        m = (label_map == c["gi"]).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        regions = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_ZONE_FRAC * car_area:
                continue
            regions.append({
                "bbox": [int(stats[i, cv2.CC_STAT_LEFT]),
                         int(stats[i, cv2.CC_STAT_TOP]),
                         int(stats[i, cv2.CC_STAT_WIDTH]),
                         int(stats[i, cv2.CC_STAT_HEIGHT])],
                "centroid": [round(float(cents[i][0]), 1),
                             round(float(cents[i][1]), 1)],
                "area_frac_of_car": round(area / car_area, 4),
            })
        regions.sort(key=lambda r: -r["area_frac_of_car"])
        zones.append({
            "hex": c["hex"],
            "role": c["role"],
            "coverage": round(c["coverage"], 4),
            "regions": regions[:12],
        })
    return zones


# ------------------------------------------------------------------- report

def build_report(src_name, method, img_bgr, mask, zones_img, clusters):
    overlay = img_bgr.copy()
    overlay[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
    edge = cv2.Canny(mask, 50, 150)
    overlay[cv2.dilate(edge, None) > 0] = (0, 255, 90)

    sw = "".join(
        f'<div class="sw"><div class="chip" style="background:{c["hex"]}"></div>'
        f'<code>{c["hex"]}</code><span class="role {c["role"]}">{c["role"]}'
        f'</span><span>{c["coverage"]*100:.1f}%</span></div>'
        for c in clusters)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>livery-lens — {src_name}</title><style>
 body{{font:14px/1.5 system-ui;margin:24px;background:#14161a;color:#e8e8ec}}
 h1{{font-size:20px}} h2{{font-size:15px;margin:28px 0 8px;color:#9aa3b2}}
 img{{max-width:100%;border-radius:8px;display:block}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
 .sw{{display:flex;align-items:center;gap:10px;padding:6px 10px;background:#1d2026;
      border-radius:8px;margin:4px 0;max-width:420px}}
 .chip{{width:42px;height:28px;border-radius:6px;border:1px solid #333}}
 code{{font:13px ui-monospace,monospace}}
 .role{{font-size:11px;padding:2px 8px;border-radius:99px;text-transform:uppercase}}
 .base{{background:#2d4a7a}} .accent{{background:#7a4a2d}} .detail{{background:#3a3f4a}}
 footer{{margin-top:32px;color:#666;font-size:12px}}
</style></head><body>
<h1>livery-lens report — {src_name}</h1>
<p>Segmentation method: <b>{method}</b></p>
<div class="grid">
 <div><h2>Original</h2><img src="{jpg_data_uri(img_bgr)}"></div>
 <div><h2>Detected car mask</h2><img src="{jpg_data_uri(overlay)}"></div>
 <div><h2>Zone map (posterized)</h2><img src="{png_data_uri(zones_img)}"></div>
 <div><h2>Palette</h2>{sw}</div>
</div>
<footer>Generated by livery-lens extract.py — a spike; expect a starting
point, not a finished livery.</footer></body></html>"""


# ----------------------------------------------------------------- starter

def clearcoat_starter(src_name, clusters):
    """Proposed Clearcoat palette import schema: livery-lens/palette@1.

    Clearcoat has no palette import today; this matches its conventions
    (lowercase #rrggbb hex, a single baseColor plus layer colors).
    """
    base = next((c for c in clusters if c["role"] == "base"), clusters[0])
    return {
        "schema": "livery-lens/palette@1",
        "source": {
            "photo": src_name,
            "generator": "livery-lens extract.py",
            "generated": datetime.datetime.now(datetime.timezone.utc)
                         .isoformat(timespec="seconds"),
        },
        "baseColor": base["hex"],
        "colors": [
            {"hex": c["hex"], "role": c["role"],
             "coverage": round(c["coverage"], 4),
             "lab": c["lab"]}
            for c in clusters
        ],
        "notes": "Coverage is the fraction of detected car-body pixels. "
                 "Roles are heuristic guesses (base/accent/detail).",
    }


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path, help="input race car photo")
    ap.add_argument("-o", "--out", type=Path, required=True,
                    help="output directory")
    ap.add_argument("-k", "--colors", type=int, default=7,
                    help="max palette size before merging (default 7)")
    ap.add_argument("--segmenter", choices=["auto", "rembg", "grabcut", "none"],
                    default="auto",
                    help="car/background separation method (default auto: "
                         "rembg if installed, else grabcut)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    img = resize_max(imread_unicode(args.photo))
    mask, method = build_mask(img, args.segmenter)
    print(f"segmentation: {method} "
          f"({mask.sum() / 255 / mask.size * 100:.0f}% of frame is car)")

    clusters, label_map, _ = extract_palette(img, mask, args.colors)
    assign_roles(clusters)
    zones_img, clean_labels = render_zones(img, label_map, clusters)
    car_area = int((mask > 0).sum())
    zones = zone_geometry(clean_labels, clusters, max(car_area, 1))

    (args.out / "palette.json").write_text(json.dumps({
        "source": args.photo.name,
        "segmenter": method,
        "colors": [{k: c[k] for k in
                    ("hex", "role", "coverage", "saturation", "lab")}
                   for c in clusters],
    }, indent=2), encoding="utf-8")
    (args.out / "zones.json").write_text(
        json.dumps({"image_size": [img.shape[1], img.shape[0]],
                    "zones": zones}, indent=2), encoding="utf-8")
    (args.out / "zones.png").write_bytes(png_bytes(zones_img))
    (args.out / "mask.png").write_bytes(png_bytes(mask))
    (args.out / "clearcoat_starter.json").write_text(
        json.dumps(clearcoat_starter(args.photo.name, clusters), indent=2),
        encoding="utf-8")
    (args.out / "report.html").write_text(
        build_report(args.photo.name, method, img, mask, zones_img, clusters),
        encoding="utf-8")

    print(f"palette ({len(clusters)} colors):")
    for c in clusters:
        print(f"  {c['hex']}  {c['coverage']*100:5.1f}%  {c['role']}")
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
