# M2 — ORB + Brute-Force Baseline

## Overview
Milestone 2 established the first working panorama stitcher using classical binary feature matching.

| Component | Choice |
|---|---|
| Feature detector | ORB (Oriented FAST + Rotated BRIEF) |
| Descriptor type | Binary (32-byte, Hamming distance) |
| Matcher | BruteForce with crossCheck |
| Outlier rejection | Sort by distance, take top-N (no RANSAC) |
| Compositing | Hard pixel overwrite (img1 pasted over warped img2) |

## Known limitations identified at M2
1. **ORB descriptors** are less distinctive than SIFT on planar-texture regions (low gradient areas).
2. **No RANSAC** — all top-N matches include outliers, leading to inaccurate homographies.
3. **Hard overwrite** compositing creates a visible seam at the image boundary.
4. **Two-image only** — no support for N-image sequences.

## Results
| Metric | Value |
|---|---|
| Output | `outputs/m2/baseline_panorama.jpg` |
| Match visualization | `outputs/m2/feature_matches.jpg` |

## Code
Original pseudocode planning notes are archived in `docs/M2_baseline_logic_pseudocode.py`.
The M3 implementation is in `src/pipeline.py` (factored from `src/baseline.py`).
