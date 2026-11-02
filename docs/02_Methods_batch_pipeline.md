# Multi-Scene Tiling + Batch Pipeline (Issue #18)

The shared pipeline that runs **any** classical baseline on the **same** tiles across
the **same** scenes, so RGB (#9), DCP (#16), and HOT (#17) can be compared head-to-head
in Task 5. It generalizes the DCP-only `scripts/runner_dcp.py`.

## Pieces

| File | Role |
|---|---|
| `data/labeled_scenes.csv` | scene list (548 tiles, 8 events) from `scripts/scene_finder.py` — columns `event, item_id, cloud_percent, visual_href, mask_href` |
| `src/tiling.py` | `read_scene()` → aligned RGB+mask; `iter_tiles()` → matching tiles; `write_mask_gis()` → georeferenced prediction |
| `src/methods/registry.py` | `get_detector(name)` → uniform `detect_fn(image) -> uint8 mask` |
| `scripts/run_baselines.py` | batch runner: tile → `detect_fn` → `evaluate()` → results table |
| `src/eval.py` | shared `evaluate(pred, true, valid)` (unchanged, #8) |

## Alignment

The visual COG is ~8× the mask resolution. `read_scene()` reads the visual resampled
onto the mask grid (`out_shape=mask.shape`, `Resampling.average`), so image and mask
are pixel-aligned and a NumPy slice of both at the same offset is a matching tile.

Mask convention (Maxar's model): `0=nodata, 1=clear, 2=cloud, 3=shadow`. Scoring uses
`evaluate(pred, mask == 2, mask != 0)` — shadow counts as not-cloud, nodata excluded.

## Run it

```bash
python scripts/run_baselines.py --method rgb --export-gis   # the RGB reference rerun
python scripts/run_baselines.py --method dcp                # DCP across all scenes
python scripts/run_baselines.py --method all --limit 5      # quick smoke test
```

Outputs: `results/<method>_tiles.csv` (per-tile P/R/F1 + cloud_frac),
`results/summary_by_event.csv` (mean per method×event + an `ALL` row), and with
`--export-gis`, `results/masks_gis/*.tif` (open over the source scene in QGIS/leafmap).

## Adding HOT (issue #17)

1. Implement `src/methods/hot_detect.py` with `detect(image, ...) -> uint8 mask` (1 = cloud),
   taking an `(H, W, 3)` tile — the same shape RGB and DCP receive.
2. In `src/methods/registry.py`: add a `"hot"` branch to `get_detector()` (bake tuned
   thresholds in with `functools.partial`) and add `"hot"` to `METHODS`.
3. `python scripts/run_baselines.py --method hot` — the runner needs no changes.

> Scores measure agreement with Maxar's cloud model, not hand truth. Tune HOT's
> threshold on scenes you don't score (issue #17's "separate tiles") to avoid leakage.
