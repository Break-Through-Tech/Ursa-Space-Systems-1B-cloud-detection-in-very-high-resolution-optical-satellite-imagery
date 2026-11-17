# Milestone 2 — Evaluation patch

All three Task-2 deliverables score on **one shared patch**, so it is pinned in code
(`src/data.py`, `load_belize_patch()`) and reproducible straight from the public
Maxar COG — no image files need to be passed around.

## Patch: `belize_r11384_c1536`

| field | value |
|---|---|
| event | `Belize-Wildfires-June24` (Maxar Open Data) |
| capture | `10300100FDC7C400`, 2024-05-27 |
| tile | `033131012121` — `visual` 8-bit RGB COG, 17408 × 17408 |
| asset URL | `…/events/Belize-Wildfires-June24/ard/16/033131012121/2024-05-27/10300100FDC7C400-visual.tif` |
| window | full-res `Window(col_off=1536, row_off=11384, width=1024, height=1024)` |
| size | 1024 × 1024, 100% valid (no nodata) |
| content | thick opaque cloud (top-left), thin haze veil, clear forest + dirt road (bottom-right) |

> **Offset convention:** `row_off`/`col_off` are **absolute full-resolution pixel
> offsets** into tile `033131012121`, used directly as a `rasterio` `Window`. The
> patch name encodes them: `belize_r<row_off>_c<col_off>`.

> **Note on the original name.** The issues referred to `belize_r3300_c8500`. Those
> offsets land on cloud-free forest (verified against the live COG), so the patch was
> re-pinned to `belize_r11384_c1536`, which contains a real cloud/clear mix — needed
> for precision *and* recall to be meaningful. If the team prefers the old name, only
> the offsets/URL in `src/data.py` need to change; everything else keys off `PATCH_ID`.

## Ground-truth mask (Task 2.1 / issue #8)

The hand-labeled mask **must be drawn on this exact patch** so it aligns pixel-for-pixel
with `detect(...)` output. Export it to:

```
data/masks/belize_r11384_c1536_mask.png   # 1024×1024, 1 = cloud, 0 = clear
```

Label rules (proposed — confirm in #8):
- **1 (cloud):** opaque cloud and clearly cloud-brightened thin cloud/haze.
- **0 (clear):** ground, water, shadow, bright non-cloud surfaces (road, soil).
- **invalid:** nodata, or genuinely unsure pixels — excluded from scoring via the
  `valid` mask returned by `load_belize_patch()`.

## Reproduce

```python
from src.data import load_belize_patch
rgb, valid = load_belize_patch()      # (1024,1024,3) uint8, (1024,1024) bool
```

---

# Milestone 3 — Frozen evaluation set (Task #1 / issue #24)

The M3 head-to-head (classical baselines vs. pretrained models) requires **every method
to be scored on identical tiles**. That tile set is now **frozen** and made explicit here,
so classical and pretrained runs reference the same locked list instead of re-deriving it.

## Scene list — `labeled_scenes.csv`

**548 scenes** across 8 Maxar Open Data events (columns
`event,item_id,cloud_percent,visual_href,mask_href`, written by `scripts/scene_finder.py`).
This file is the frozen source of truth — do not edit it without re-freezing the tile list.

## Tile manifest — `eval_tiles.csv` (the locked evaluation set)

Materialized by `scripts/freeze_tiles.py`, which reuses `src/tiling.py:iter_tiles()` — the
same tiling the batch runner (`scripts/run_baselines.py`) iterates — so the manifest matches
exactly what every method scores at runtime. One row per scorable tile:

| column | meaning |
|---|---|
| `event`, `item_id` | the scene (keys into `labeled_scenes.csv`) |
| `tile_row`, `tile_col` | tile index in the scene's grid |
| `r0`, `c0` | tile's pixel offset in the (mask-grid) scene |
| `height`, `width` | tile size (edge tiles are smaller, not padded) |
| `n_valid` | non-nodata mask pixels in the tile (what `evaluate()` scores over) |
| `tile` | tile size parameter (`512`) |

**Frozen set:** `tile=512`, `skip_empty=True` (all-nodata tiles dropped) →
**538 scenes kept, 10 skipped, 10,072 tiles.**

The 10 skipped scenes are the single-band capture `10200100C3078400`
(`southafrica-flooding22`): their visual COG has < 3 bands, so `read_scene()` fails for
*every* method — they are excluded here so the manifest is exactly the tiles each method can
actually score. (Other, 3-band `southafrica-flooding22` captures remain in the set.)

### Regenerate

```bash
python scripts/freeze_tiles.py          # → data/eval_tiles.csv (deterministic)
```

Lightweight: reads only each scene's decimated cloud mask plus the visual COG's header
(band count) — it never downloads a full-resolution visual.
