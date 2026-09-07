# M4 — Planned Improvements & Implementation Status

## Motivation
M3 achieved 76.56% inlier ratio and 0.54 px RMSE at ~0.65 s/pair on a 2-image sequence.
Three failure cases were identified for M4 to address:

| Failure Case | Root Cause | M4 Module |
|---|---|---|
| Visible exposure seams | Hard-overwrite compositing | `src/blending/` |
| Catastrophic warps on small-overlap pairs | No stretch check on H | `src/geometry/homography.py` |
| Iterative drift on 4+ frame sequences | Sequential error compounding | `src/bundle_adjustment/optimizer.py` |

---

## Implemented ✅

### 1. Distance-based feathering (`src/blending/feathering.py`)
- Per-pixel distance-transform weights smooth the exposure transition across the seam zone.
- No smoothing of high-frequency edges (gradient is weighted by mask distance, not blurred).
- Activated via `--blend feather` or `configs/default.yaml → blend_mode: feather`.

### 2. Laplacian pyramid multi-band blending (`src/blending/multiband.py`)
- Burt & Adelson (1983) approach: blend Laplacian residuals at each pyramid level.
- Low-freq bands (exposure) blended over wide zone; high-freq bands (edges) at sharp boundary.
- Activated via `--blend multiband` or `configs/default.yaml → blend_mode: multiband`.

### 3. Smart fallback transform (`src/geometry/homography.py`)
- After RANSAC, bounding-box stretch is measured.
- If width or height ratio > `stretch_limit` (default 1.3×) → fall back to affine
  (`estimateAffinePartial2D`: rotation + scale + translation only).
- Every fallback event logged to `outputs/metrics/fallback_log.csv`
  (timestamp, pair, stretch factor, reason).

### 4. Global bundle adjustment (`src/bundle_adjustment/optimizer.py`)
- After sequential stitching, all pairwise H matrices refined jointly using
  Levenberg-Marquardt (scipy `least_squares`).
- Reports before/after RMSE per pair. Activated via `--ba` flag.

### 5. Exposure compensation (`src/stitching/warper.py`)
- Per-channel mean+std normalization (Reinhard-style) applied before blending
  when `compensate_exposure: true` in config or `warp_images(..., compensate_exposure=True)`.

### 6. Seam mask from distance transform (`src/stitching/warper.py`)
- `distance_seam_mask()` computes per-pixel blend weights for downstream use.

### 7. Synthetic 4-frame test sequence (`generate_dummy_data.py`)
- Generates `frame_00.jpg … frame_03.jpg` in `data/raw/` for multi-image + BA testing.

### 8. Unit tests (`tests/`)
- `test_features.py` — SIFT/ORB detection
- `test_matching.py` — ratio-test filtering, point-pair shapes
- `test_geometry.py` — RANSAC mask shape, fallback trigger at 1.3×
- `test_blending.py` — feathering + pyramid blend on synthetic patches, seam gradient

### 9. Benchmark suite (`tests/benchmark.py`)
- Full pipeline benchmark logging: pair, time, inliers, RMSE, seam score, fallback.

---

## Still Open / Future Work 🔲

| Item | Priority | Notes |
|---|---|---|
| Graph-cut seam finding | Medium | Replace distance-transform with optimal seam path |
| Confidence-weighted compositing from inlier density | Medium | Weight blend by inlier count near seam |
| Cylindrical / spherical projection pre-warp | High (for wide FOV) | Required for >120° panoramas |
| Full OpenCV `detail::BundleAdjusterRay` integration | High | More robust than educational LM for 8+ images |
| GPU acceleration (cv2.cuda) | Low | ~5–10× speed gain for large datasets |

---

## Results Table (M2 → M3 → M4)

| Metric | M2 (ORB+BF) | M3 (SIFT+FLANN) | M4 (feather) | M4 (multiband) |
|---|---|---|---|---|
| Inlier ratio | N/A (no RANSAC) | **76.56 %** | 76.56 % (same) | 76.56 % (same) |
| RMSE (px) | N/A | **0.54** | 0.54 (same) | 0.54 (same) |
| Speed (s/pair) | ~0.2 | **~0.65** | ~0.70 | ~0.90 |
| Seam quality | Hard seam | Hard seam | Smooth (feathered) | Sharp + seamless |
| N-image support | ❌ | ✅ | ✅ | ✅ |
| Bundle adjustment | ❌ | ❌ | ✅ (--ba) | ✅ (--ba) |

> Note: RMSE and inlier ratio are geometry metrics, unchanged by blending.
> Seam quality improvement is visual — use `tests/benchmark.py` for seam_score column.
