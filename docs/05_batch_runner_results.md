# Task 5: Batch Runner Results

## Setup
All three classical baselines were run on 548 Maxar scenes, cut into 512 px tiles.
Settings: RGB (brightness 0.5, saturation 0.2), DCP (t = 0.6), HOT (threshold -0.08).

## Results

| Method | Precision | Recall | F1 |
|---|---|---|---|
| RGB | 0.8018 | 0.5215 | **0.5936** |
| HOT | 0.5025 | 0.8730 | 0.5804 |
| DCP | 0.8468 | 0.3792 | 0.4840 |

## Winner
After running all 548 Maxar scenes, the method with the best F1 came out to be RGB, closely tying with HOT. RGB and HOT are only about 0.01 apart, while DCP trails by about 0.1.

## Trade-off
RGB and DCP seem to be conservative. They each have high precision and low recall, so the low recall is what pulls their F1 down. On the other hand, HOT raises more false alarms, as it resulted in a high recall and a low precision.

HOT's -0.08 threshold is loose, so it catches almost every cloud but also flags things that aren't clouds. In Tonga, it even seems to flag the black, empty edge of the tile.

## Failures
10 scenes failed, once for each method, giving 30 FAILED lines. All 10 come from one capture, 10200100C3078400 (the South Africa grayscale set), and each image has only 1 band where the code expects RGB. So every method scored 538 of 548. This seems to be a data issue, not a method failure.

## Caveats
- The scores measure agreement with Maxar's model-made masks, not hand-labeled truth.
- HOT's threshold was tuned on about 100 of these same scenes, which slightly favors HOT.

## If we had more time
- Tuning DCP's threshold could raise its recall and produce a better result.
- Combining RGB and HOT could balance RGB's precision with HOT's recall.

## Next step
The team is moving on to pretrained models. These F1 scores (RGB 0.5936, HOT 0.5804, DCP 0.4840) are the bar the pretrained models need to beat on the same 548 scenes.
