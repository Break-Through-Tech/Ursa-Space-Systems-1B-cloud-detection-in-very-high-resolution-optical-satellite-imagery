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
