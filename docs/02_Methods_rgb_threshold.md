# Method: RGB brightness/saturation threshold (baseline)

_Milestone 2, Task 2.2 (issue #9). Copy of the note for Drive `02_Methods`._

## The idea

The simplest cloud cue in a plain RGB image: thick cloud is **bright** and **gray**.
Bright, because cloud tops reflect a lot of visible light; gray (**desaturated**),
because that reflection is roughly equal across red, green, and blue (R ≈ G ≈ B).
Most ground cover is either darker (vegetation, water, shadow) or more colorful
(green canopy, brown soil, blue water), so a brightness-high / saturation-low test
separates a lot of cloud from a lot of ground with no training data at all.

## The formula

Work in HSV-style brightness and saturation, both scaled to `[0, 1]`:

```
brightness  V = max(R, G, B)
saturation  S = (max(R,G,B) − min(R,G,B)) / max(R,G,B)

cloud(pixel) = (V > b) AND (S < s)
```

`b` is the brightness threshold, `s` the saturation threshold (both in `[0, 1]`).
Higher `b` and lower `s` make the detector stricter. Implemented in pure NumPy in
`src/methods/rgb_threshold.py` as `detect(image, b, s) -> uint8 mask` (1 = cloud).
Tuned values come from the Task 2.3 sweep (issue #10).

## Result on the Belize patch

Patch `belize_r11384_c1536` (Belize-Wildfires-June24, tile 033131012121), at the
starting thresholds **b = 0.5, s = 0.2**: ~41% of the patch flagged as cloud. The
thick opaque cloud in the top-left is captured cleanly with a tight edge, and the
clear forest/road in the bottom-right is correctly left out.

Preview: `results/belize_r11384_c1536_rgb_threshold.png` (RGB | mask | overlay).

## Where it fails

Anything that is *also* bright **and** desaturated passes the same test:

- **Thin cloud / haze** — semi-transparent, so the ground color bleeds through and
  raises saturation (and lowers brightness). Partly **missed** — visible as the thin
  wisp at the bottom of the patch that the mask only grazes. This is the baseline's
  biggest blind spot and the motivation for the HOT and dark-channel methods.
- **Bright desaturated surfaces** — snow, sand, bare concrete, light rooftops: bright
  and gray, so **false-positived** as cloud. (Not present in this forest patch, but
  the dominant failure mode on urban/coastal scenes.)
- **Cloud shadow** — dark, so never flagged; shadows stay in the "clear" class even
  though the ground under them is also unusable.
- **Threshold sensitivity** — one global `(b, s)` per scene; it can't adapt to
  illumination changes within a scene, so the best thresholds are scene-dependent.

## Interface (for the rest of Milestone 2)

- `detect(image, b, s)` — issue #9 (this note). `image` is `(H,W,3)` RGB (uint8 or
  float); returns a `(H,W)` uint8 mask.
- `evaluate(pred, true, valid)` — issue #8 — scores this mask vs. ground truth.
- threshold sweep — issue #10 — loops `detect` over a grid of `(b, s)`, scores each.
