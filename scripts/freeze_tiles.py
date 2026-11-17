"""Freeze the exact evaluation tile list for the Milestone 3 head-to-head (issue #24).

A fair comparison of classical baselines (RGB #9, DCP #16, HOT #17) against pretrained
models requires every method to be scored on the *same* tiles. The tile set is defined
implicitly by ``data/labeled_scenes.csv`` + the deterministic tiling in ``src/tiling.py``.
This script materializes that set into ``data/eval_tiles.csv`` — one row per scorable
tile — so the locked evaluation set is explicit, auditable, and reusable by every method.

It is deliberately lightweight: it never downloads a full-resolution visual COG. For each
scene it reads only the decimated cloud mask (~2176 px) to enumerate tiles, and checks the
visual's *header* (band count) to drop scenes that can't be scored.

The tile list is produced by reusing ``src.tiling.iter_tiles`` — the same function the
batch runner (``scripts/run_baselines.py``) iterates — so the frozen manifest matches
exactly what every method will score at runtime (``tile=512``, all-nodata tiles dropped).

Scenes whose visual COG has fewer than 3 bands (the single-band ``southafrica-flooding22``
scenes) fail ``read_scene`` for every method, so they are excluded here and logged.

Usage (from the repo root, venv active)::

    python scripts/freeze_tiles.py                 # writes data/eval_tiles.csv

Mask value convention (Maxar's cloud model): ``0 = nodata, 1 = clear, 2 = cloud,
3 = shadow``; ``n_valid`` counts non-nodata pixels in the tile (what evaluate() scores).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tiling import iter_tiles

SCENES_CSV = ROOT / "data" / "labeled_scenes.csv"
OUT_CSV = ROOT / "data" / "eval_tiles.csv"


def freeze_scene(row, tile):
    """Enumerate the scorable tiles of one scene from its mask alone.

    Opens the cloud mask (small) to enumerate tiles and the visual COG's *header* to
    confirm it has >= 3 bands. Returns ``(rows, status)`` where ``status`` is "ok",
    "skip:single-band", or "error:<msg>". A zero-cost broadcast stands in for the image
    so ``iter_tiles`` yields exactly the tile indices the runner would.
    """
    with rasterio.open(row["visual_href"]) as v:
        if v.count < 3:  # header read only — no pixels fetched
            return [], f"skip:single-band({v.count})"

    with rasterio.open(row["mask_href"]) as m:
        mask = m.read(1)

    placeholder = np.broadcast_to(mask[:, :, None], mask.shape + (3,))  # no allocation
    rows = []
    for tile_row, tile_col, r0, c0, _img_tile, mask_tile in iter_tiles(
        placeholder, mask, tile
    ):
        rows.append({
            "event": row["event"],
            "item_id": row["item_id"],
            "tile_row": tile_row,
            "tile_col": tile_col,
            "r0": r0,
            "c0": c0,
            "height": mask_tile.shape[0],
            "width": mask_tile.shape[1],
            "n_valid": int((mask_tile != 0).sum()),
            "tile": tile,
        })
    return rows, "ok"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tile", type=int, default=512,
                    help="tile size in mask pixels (must match run_baselines.py)")
    ap.add_argument("--scenes", default=str(SCENES_CSV),
                    help="path to the labeled-scenes CSV")
    ap.add_argument("--out", default=str(OUT_CSV),
                    help="path to write the frozen tile manifest")
    args = ap.parse_args()

    scenes = pd.read_csv(args.scenes)
    print(f"freezing tiles for {len(scenes)} scenes  (tile={args.tile})")

    all_rows = []
    kept = skipped = failed = 0
    for i, row in scenes.iterrows():
        try:
            rows, status = freeze_scene(row, args.tile)
        except Exception as e:  # network / read error — log and keep going
            failed += 1
            print(f"{i + 1}/{len(scenes)} {row['item_id']}  ERROR: {e}")
            continue
        if status == "ok":
            kept += 1
            all_rows.extend(rows)
            print(f"{i + 1}/{len(scenes)} {row['item_id']}  tiles={len(rows)}")
        else:
            skipped += 1
            print(f"{i + 1}/{len(scenes)} {row['item_id']}  SKIP ({status})")

    df = pd.DataFrame(all_rows).sort_values(
        ["event", "item_id", "tile_row", "tile_col"]
    ).reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nscenes kept: {kept}  skipped: {skipped}  failed: {failed}")
    print(f"frozen tiles: {len(df)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
