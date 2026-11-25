# Task 4: Head-to-Head Benchmark (Classical vs. Pretrained)

## Setup
All five detectors were scored on the same frozen evaluation tiles (Maxar scenes cut
into 512 px tiles, scored against Maxar's cloud mask with the shared `evaluate()`;
cloud = mask value 2, shadow counts as not-cloud, nodata excluded).

- **Classical + OmniCloudMask** ran on the full set of **548 scenes** (10,072 scored
  tiles, 8 events). Settings: RGB (brightness 0.5, saturation 0.2), DCP (t = 0.6),
  HOT (threshold −0.08), OMCM (Red/Green/NIR from the `-ms` product at 10 m).
- **SAM (zero-shot)** ran on a **stratified 29-scene subset** (4 scenes per event
  across all 8 events, spanning 20–100 % cloud; 431 scored tiles), with
  `SAM_MODEL_TYPE=vit_b` and `SAM_POINTS_PER_SIDE=8` on CPU. The subset is used because
  SAM's automatic mask generator on CPU is far too slow for all 10k tiles; the subset
  still spans every event for a fair read. The scope asymmetry is noted where it matters.

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
| SAM (zero-shot) | 29-scene subset | 0.4004 | 0.1934 | 0.2087 |

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
- **SAM** is weak on both (P 0.40 / R 0.19). A generic segmenter plus a trivial
  brightness/saturation cloud rule under-segments diffuse/thin cloud and its coarse CPU
  grid (points = 8) misses small regions; the few objects it keeps are often not cloud.

## Per-event F1

| Event | RGB | DCP | HOT | OMCM | SAM |
|---|---|---|---|---|---|
| Cyclone-Chido | 0.673 | 0.536 | 0.594 | **0.931** | 0.387 |
| Hurricane-Maria | 0.505 | 0.481 | 0.455 | **0.718** | 0.113 |
| HurricaneHelene | 0.500 | 0.368 | 0.462 | **0.779** | 0.174 |
| HurricaneMilton | 0.441 | 0.464 | 0.629 | **0.659** | 0.367 |
| cyclone-emnati22 | 0.625 | 0.509 | 0.548 | **0.799** | 0.183 |
| ghana-explosion22 | 0.695 | 0.517 | 0.463 | **0.825** | 0.166 |
| southafrica-flooding22 | 0.333 | 0.281 | 0.636 | **0.740** | 0.048 |
| tonga-volcano21 | 0.623 | 0.494 | 0.642 | **0.848** | 0.255 |

## Failures
The same 10 scenes that failed in M2 fail again for RGB, DCP, HOT, and OMCM — all from
the single capture `10200100C3078400` (the South-Africa grayscale set), which has one
band where RGB is expected. So each full-set method scored 538 of 548 scenes. In the SAM
subset, 2 of the 4 South-Africa scenes are from that capture and fail the same way (SAM
scored 27 of 29). This is a data issue, not a method failure.

## Caveats
- Scores measure agreement with **Maxar's model-made masks**, not hand-labeled truth.
- **OMCM uses Red/Green/NIR** from the `-ms` product, so it is not a pure-RGB entry — it
  has a spectral band the RGB methods (including SAM) do not. Its win partly reflects that
  extra information, which is the point of comparing purpose-built vs. RGB-only methods.
- **SAM is handicapped three ways** relative to the others: zero-shot with a deliberately
  trivial cloud rule, a reduced subset, and a coarse `points_per_side=8` grid for CPU
  feasibility. Its F1 is a floor, not SAM's ceiling.
- **HOT's** −0.08 threshold was tuned on ~100 of these same scenes, which slightly favors it.

## Next step (Task #5)
Qualitative failure-mode analysis on these masks: smoke vs. cloud, bright surfaces
(snow, sand, rooftops) flagged as cloud, and thin/diffuse cloud — where each method wins
and why. The benchmark notebook (`notebooks/benchmark_head_to_head.ipynb`) renders the
side-by-side per-method masks used for that analysis.
