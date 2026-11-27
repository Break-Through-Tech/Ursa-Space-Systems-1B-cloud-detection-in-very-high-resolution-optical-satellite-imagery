# Task 4: Head-to-Head Benchmark (Classical vs. Pretrained)

## Setup
All five detectors were scored on the same frozen evaluation tiles (Maxar scenes cut
into 512 px tiles, scored against Maxar's cloud mask with the shared `evaluate()`;
cloud = mask value 2, shadow counts as not-cloud, nodata excluded).

- **Classical + OmniCloudMask** ran on the full set of **548 scenes** (10,072 scored
  tiles, 8 events). Settings: RGB (brightness 0.5, saturation 0.2), DCP (t = 0.6),
  HOT (threshold −0.08), OMCM (Red/Green/NIR from the `-ms` product at 10 m).
- **SAM (zero-shot)** numbers come from the SAM task owner's run (issue #26), covering
  all 8 events. (The benchmark notebook also includes a quick local reproduction on a
  coarse CPU grid for the side-by-side visuals; it is weaker than the headline SAM run
  and is labeled as such.)

Methods join the one method-agnostic runner (`scripts/run_baselines.py`), so every tile
is identical across methods (`src/tiling.iter_tiles`, 512 px). The classical numbers
reproduce `docs/05` exactly, confirming the tiling is deterministic.

## Results

| Method | Scope | Precision | Recall | F1 |
|---|---|---|---|---|
| **OmniCloudMask** | 548 scenes | 0.7987 | 0.8341 | **0.8017** |
| RGB | 548 scenes | 0.8018 | 0.5215 | 0.5936 |
| HOT | 548 scenes | 0.5025 | 0.8730 | 0.5804 |
| DCP | 548 scenes | 0.8468 | 0.3792 | 0.4840 |
| SAM (zero-shot) | all 8 events (#26 run) | 0.7038 | 0.4208 | 0.4592 |

## Winner
OmniCloudMask wins decisively: F1 0.80 overall, about 0.21 above the best classical
baseline (RGB, 0.59), and it has the **best F1 in every one of the 8 events** (0.66 on
HurricaneMilton up to 0.93 on Cyclone-Chido). It is also the only method that is strong
on both precision and recall at once — the purpose-built, NIR-aware model pays off.

## Trade-off
- **OMCM** is balanced (P 0.80 / R 0.83) — no lopsided failure mode.
- **Classical** methods show the familiar split: RGB and DCP are conservative (high
  precision, low recall), HOT is loose (high recall 0.87, low precision 0.50). Among the
  classical three the per-event winner alternates between RGB and HOT depending on scene.
- **SAM** is conservative (P 0.70 / R 0.42): a generic segmenter plus a trivial
  brightness/saturation cloud rule keeps only confident regions, so precision is decent
  but recall is low — it under-segments diffuse/thin cloud that has no sharp boundary.
  Overall it lands just below DCP, last of the five.

## Per-event F1

| Event | RGB | DCP | HOT | OMCM | SAM |
|---|---|---|---|---|---|
| Cyclone-Chido | 0.673 | 0.536 | 0.594 | **0.931** | 0.553 |
| Hurricane-Maria | 0.505 | 0.481 | 0.455 | **0.718** | 0.362 |
| HurricaneHelene | 0.500 | 0.368 | 0.462 | **0.779** | 0.340 |
| HurricaneMilton | 0.441 | 0.464 | 0.629 | **0.659** | 0.374 |
| cyclone-emnati22 | 0.625 | 0.509 | 0.548 | **0.799** | 0.479 |
| ghana-explosion22 | 0.695 | 0.517 | 0.463 | **0.825** | 0.558 |
| southafrica-flooding22 | 0.333 | 0.281 | 0.636 | **0.740** | 0.180 |
| tonga-volcano21 | 0.623 | 0.494 | 0.642 | **0.848** | 0.489 |

OMCM is the best method in every event; no other method wins anywhere.

## Failures
The same 10 scenes that failed in M2 fail again for RGB, DCP, HOT, and OMCM — all from
the single capture `10200100C3078400` (the South-Africa grayscale set), which has one
band where RGB is expected. So each full-set method scored 538 of 548 scenes. This is a
data issue, not a method failure.

## Caveats
- Scores measure agreement with **Maxar's model-made masks**, not hand-labeled truth.
- **OMCM uses Red/Green/NIR** from the `-ms` product, so it is not a pure-RGB entry — it
  has a spectral band the RGB methods (including SAM) do not. Its win partly reflects that
  extra information, which is the point of comparing purpose-built vs. RGB-only methods.
- **SAM is zero-shot** with a deliberately trivial brightness/saturation cloud rule, so
  its score reflects a generic segmenter plus a simple cue, not a cloud-trained model.
- **HOT's** −0.08 threshold was tuned on ~100 of these same scenes, which slightly favors it.

## Next step (Task #5)
Qualitative failure-mode analysis on these masks: smoke vs. cloud, bright surfaces
(snow, sand, rooftops) flagged as cloud, and thin/diffuse cloud — where each method wins
and why. The benchmark notebook (`notebooks/benchmark_head_to_head.ipynb`) renders the
side-by-side per-method masks used for that analysis.
