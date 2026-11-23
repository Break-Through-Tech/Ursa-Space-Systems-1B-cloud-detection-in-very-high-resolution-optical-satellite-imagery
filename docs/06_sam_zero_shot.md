# SAM Zero-Shot Baseline (Milestone 3)

Segment Anything (SAM, Kirillov et al., ICCV 2023) is the general-purpose
foundation-model comparison point. It runs on the RGB visual tiles only — no NIR,
no cloud-specific training — via `src/methods/sam_detect.py`.

## How it works

1. `SamAutomaticMaskGenerator` proposes object-level regions over the tile.
2. Each region is kept as cloud iff its **mean brightness > `b` (0.5)** and
   **mean saturation < `s` (0.2)** — the same cue as the RGB baseline, applied
   per SAM region instead of per pixel.
3. Kept regions are unioned into the binary mask.

SAM supplies *what counts as one object*; the spectral rule still decides *whether
that object is cloud*. Checkpoint/model come from `SAM_CHECKPOINT` +
`SAM_MODEL_TYPE` (see the module docstring for the Colab setup).

## Run it

```bash
pip install git+https://github.com/facebookresearch/segment-anything
export SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth SAM_MODEL_TYPE=vit_h
python scripts/run_baselines.py --method sam
```

## Where SAM helps vs. the purpose-built baselines

- **Coherent boundaries.** Region-level output gives closed, object-shaped masks;
  the per-pixel RGB threshold speckles and leaks through thin cloud.
- **Implicit context.** SAM "knows" a bright uniform expanse is one thing, so a
  cloud deck isn't chopped by texture noise the way a pixel rule chops it.

## Where SAM hurts (cf. Ren et al., "Segment Anything, from Space?", WACV 2024)

- **No cloud prior.** SAM segments *any* salient object — snow, sand, bright
  rooftops, coastline — so the brightness/saturation rule still has to reject
  them, and it can't: those regions pass the same test clouds do.
- **Diffuse cloud is invisible to it.** Haze and thin cloud have no sharp
  boundary, so SAM's point-grid proposals tend to miss or under-segment them —
  exactly the cases the spectral index methods (HOT, DCP) are built to catch.
- **Resolution ceiling.** SAM was trained mostly on natural photos at ~1 MP;
  sub-pixel thin cloud and small cumulus in 512 px tiles sit below its
  granularity.

Net expectation: SAM should beat RGB on boundary quality for thick cloud but
lose recall on thin/hazy cloud — the head-to-head `results/summary_by_event.csv`
row for `sam` vs `rgb`/`hot`/`dcp` is the check.
