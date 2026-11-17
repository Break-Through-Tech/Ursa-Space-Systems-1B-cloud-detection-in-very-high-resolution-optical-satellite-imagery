"""Method-agnostic batch runner for the classical cloud baselines (issue #18).

Runs ANY baseline's ``detect(image)`` over every tile of every scene in
``data/labeled_scenes.csv``, scores each tile against its Maxar cloud mask with the
shared ``evaluate()``, and writes a results table to ``results/``. This generalizes
Zainab's DCP-only ``scripts/runner_dcp.py``: the detector is looked up by name from
``src/methods/registry.py``, so RGB (#9), DCP (#16), and HOT (#17) all run through
the one path on identical tiles — which is what the Task-5 comparison needs.

Usage (from the repo root, venv active)::

    python scripts/run_baselines.py --method rgb --export-gis     # the RGB rerun
    python scripts/run_baselines.py --method dcp                  # needs PR #19
    python scripts/run_baselines.py --method all --limit 5        # quick smoke test

Outputs:
  * results/<method>_tiles.csv   one row per (scene, tile): precision, recall, f1, ...
  * results/summary_by_event.csv mean P/R/F1 per (method, event) + an overall row
  * results/masks_gis/*.tif      (with --export-gis) georeferenced predicted masks

Scores measure agreement with Maxar's cloud model, not with hand truth — the masks
are model output (see scripts/scene_finder.py). Shadow (mask value 3) is scored as
not-cloud; nodata (0) is excluded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval import evaluate
from src.methods.registry import METHODS, get_detector
from src.tiling import iter_tiles, read_scene, write_mask_gis

SCENES_CSV = ROOT / "data" / "labeled_scenes.csv"
RESULTS_DIR = ROOT / "results"
GIS_DIR = RESULTS_DIR / "masks_gis"


def run_scene(row, detect_fn, tile, export_gis, method):
    """Tile one scene, score every tile, optionally write a georeferenced mask.

    Returns a list of per-tile result dicts. Any failure (bad/one-band tile, network
    error) is left to the caller to catch so the batch keeps going.
    """
    image, mask, profile = read_scene(row["visual_href"], row["mask_href"])
    full_pred = np.zeros(mask.shape, dtype=np.uint8) if export_gis else None
    rows = []

    for tile_row, tile_col, r0, c0, img_tile, mask_tile in iter_tiles(image, mask, tile):
        pred = detect_fn(img_tile)
        valid = mask_tile != 0
        scores = evaluate(pred, mask_tile == 2, valid)
        cloud_frac = float(pred[valid].mean()) if valid.any() else 0.0
        rows.append({
            "method": method,
            "event": row["event"],
            "item_id": row["item_id"],
            "tile_row": tile_row,
            "tile_col": tile_col,
            "n_valid": int(valid.sum()),
            "cloud_frac": cloud_frac,
            **scores,
        })
        if full_pred is not None:
            full_pred[r0:r0 + pred.shape[0], c0:c0 + pred.shape[1]] = pred

    if export_gis:
        GIS_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = str(row["item_id"]).replace("/", "_")
        out = GIS_DIR / f"{row['event']}_{safe_id}_{method}.tif"
        write_mask_gis(out, full_pred, profile)

    return rows


def run_method(method, scenes, tile, limit, export_gis):
    """Run one baseline across all scenes; write ``results/<method>_tiles.csv``."""
    detect_fn = get_detector(method)
    params = getattr(detect_fn, "params", None)
    print(f"\n=== method: {method}"
          + (f"  (params: {params})" if params else "")
          + f"  scenes: {len(scenes)}  tile: {tile} ===")

    results = []
    for i, row in scenes.iterrows():
        try:
            tile_rows = run_scene(row, detect_fn, tile, export_gis, method)
            results.extend(tile_rows)
            mean_f1 = np.mean([r["f1"] for r in tile_rows]) if tile_rows else float("nan")
            print(f"{i + 1}/{len(scenes)} {row['item_id']}  tiles={len(tile_rows)}  mean_f1={mean_f1:.3f}")
        except Exception as e:  # keep the batch alive, log the failure as a row
            print(f"{i + 1}/{len(scenes)} {row['item_id']}  FAILED: {e}")
            results.append({"method": method, "event": row["event"],
                            "item_id": row["item_id"], "error": str(e)})

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{method}_tiles.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"saved {out}  ({len(results)} rows)")
    return results


def write_summary(all_rows):
    """Mean precision/recall/f1 per (method, event), plus one overall row per method."""
    df = pd.DataFrame(all_rows)
    if "f1" not in df.columns:
        print("no scored tiles — skipping summary")
        return
    scored = df[df["f1"].notna()]
    by_event = (scored.groupby(["method", "event"])[["precision", "recall", "f1"]]
                .mean().round(4).reset_index())
    overall = (scored.groupby("method")[["precision", "recall", "f1"]]
               .mean().round(4).reset_index())
    overall.insert(1, "event", "ALL")
    summary = pd.concat([by_event, overall], ignore_index=True)

    out = RESULTS_DIR / "summary_by_event.csv"
    summary.to_csv(out, index=False)
    print(f"\nsaved {out}")
    print(overall.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="rgb",
                    help="rgb | dcp | all  (default: rgb)")
    ap.add_argument("--tile", type=int, default=512, help="tile size in mask pixels")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N scenes (quick smoke test)")
    ap.add_argument("--scenes", default=str(SCENES_CSV),
                    help="path to the labeled-scenes CSV")
    ap.add_argument("--export-gis", action="store_true",
                    help="write georeferenced predicted masks to results/masks_gis/")
    args = ap.parse_args()

    methods = list(METHODS) if args.method == "all" else [args.method]
    scenes = pd.read_csv(args.scenes)
    if args.limit:
        scenes = scenes.head(args.limit)

    all_rows = []
    for method in methods:
        all_rows.extend(run_method(method, scenes, args.tile, args.limit, args.export_gis))

    write_summary(all_rows)


if __name__ == "__main__":
    main()
