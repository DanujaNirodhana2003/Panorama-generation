# M3 — SIFT + FLANN + RANSAC N-Image Pipeline

## Overview
Milestone 3 upgraded every component of the M2 baseline:

| Component | M2 | M3 |
|---|---|---|
| Feature detector | ORB | SIFT |
| Descriptor type | Binary (32-byte) | Float (128-dim) |
| Matcher | BruteForce | FLANN KD-Tree (5 trees, 50 checks) |
| Outlier filtering | Top-N by distance | Lowe's ratio test (threshold 0.70) |
| Outlier rejection | None | RANSAC (reprojection threshold 5.0 px) |
| Compositing | Hard overwrite | Hard overwrite (unchanged, addressed in M4) |
| N-image support | ❌ | ✅ (sequential iterative stitching) |

## Measured Metrics (image1.jpeg ↔ image2.jpeg)

| Metric | Value |
|---|---|
| Total good matches (after ratio test) | ~280 (varies per run) |
| **Inlier ratio (after RANSAC)** | **76.56 %** |
| **RMSE (inliers only)** | **0.54 px** |
| **Wall time (per pair)** | **~0.65 s** |

## Known Limitations → addressed in M4
1. **Hard overwrite seams** — visible exposure boundary at image overlap edge.
2. **No stretch/skew check** — small-overlap pairs can produce catastrophically wide canvases.
3. **Iterative drift** — composed H[0→N] accumulates pairwise errors on 4+ image sequences.

## Outputs
- `outputs/m3/panorama_sift.jpg` — two-image panorama
- `outputs/m3/feature_matches_sift.jpg` — SIFT match visualization
